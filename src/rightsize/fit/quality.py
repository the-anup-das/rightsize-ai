"""How much a GGUF file type costs in quality, in llama.cpp's own units (F3, F4).

The measure is the perplexity a file type adds over 16-bit on Llama-3-8B, as printed by the
``llama-quantize`` we pin (data/quality/gguf_types.yaml). It is one model and one text, so it
ranks quantizations against each other well and says little about any one task; the KL
gate (F8) measures a specific model once it exists.

The i-quants have no printed perplexity. For those we interpolate along the measured types
by *effective* bits per weight - llama.cpp's own table, the same units on both sides; mixing
the i-quants' nominal bits with the K-quants' effective bits once ranked IQ4_XS below Q3_K_L.
The result is an upper bound: an i-quant made with an importance matrix usually beats a
K-quant of the same size, which is what it is for.
"""

from __future__ import annotations

import math
from functools import lru_cache

from rightsize._data import load_yaml

#: Bands in llama.cpp's units, for people rather than for the ranker, which uses the number.
#: Q6_K adds 0.02, Q5_K_M 0.06, Q4_K_M 0.18, Q4_K_S 0.27, Q3_K_M 0.66 and Q2_K 3.5 on
#: Llama-3-8B. The edges are a judgement: "good" runs to 0.3, about 5% on a model whose
#: 16-bit perplexity is near 6, which takes in both 4-bit K-quants.
BANDS = (
    (0.06, "near-lossless"),
    (0.3, "good"),
    (0.7, "noticeable"),
    (float("inf"), "damaged"),
)


@lru_cache(maxsize=1)
def _types() -> dict[str, dict]:
    return {k.upper(): v for k, v in load_yaml("quality/gguf_types.yaml")["file_types"].items()}


def _measured() -> dict[str, float]:
    return {k: float(v["ppl_delta"]) for k, v in _types().items() if v.get("ppl_delta") is not None}


def _bpw() -> dict[str, float]:
    return {k: float(v["bpw"]) for k, v in _types().items() if v.get("bpw") is not None}


def ppl_delta(quant: str) -> tuple[float | None, str]:
    """(perplexity added over 16-bit, how we know). None when it cannot be said honestly."""
    q = quant.upper()
    measured = _measured()
    if q in measured:
        return max(measured[q], 0.0), "measured by llama.cpp on Llama-3-8B"
    if quant.lower() in _format_rows():
        return format_delta(quant)
    bits = _bpw().get(q)
    if bits is None:
        return None, "unknown file type"
    # log-linear in bits: the cost of each bit removed grows as the bits run out
    points = sorted(
        (_bpw()[name], max(delta, 1e-4))
        for name, delta in measured.items()
        if name in _bpw() and name not in ("F16", "BF16", "F32")
    )
    if bits < points[0][0]:
        return None, (
            f"{bits:.2f} bits is below every measured type (lowest {points[0][0]:.2f}); "
            "too far out to extrapolate"
        )
    for (b0, d0), (b1, d1) in zip(points, points[1:], strict=False):
        if b0 <= bits <= b1:
            t = (bits - b0) / (b1 - b0) if b1 != b0 else 0.0
            delta = math.exp(math.log(d0) + t * (math.log(d1) - math.log(d0)))
            return delta, (
                f"at most this: interpolated at {bits:.2f} effective bits between measured "
                "types, and an i-quant made with an importance matrix usually does better"
            )
    return points[-1][1], "at or above the best measured type"


def band(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return next(label for limit, label in BANDS if delta <= limit)


@lru_cache(maxsize=1)
def _format_rows() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for r in load_yaml("quality/formats_quality.yaml")["formats"]:
        rows.setdefault(str(r["format"]).lower(), []).append(r)
    return rows


@lru_cache(maxsize=1)
def _ladders() -> dict[str, list[tuple[float, str]]]:
    """Per model: llama.cpp's types measured on it, as (KL divergence, type), by KLD."""
    out: dict[str, list[tuple[float, str]]] = {}
    for r in load_yaml("quality/formats_quality.yaml")["ladder"]:
        if str(r["type"]).upper() in _measured():
            out.setdefault(r["model"], []).append((float(r["kld"]), str(r["type"]).upper()))
    return {m: sorted(v) for m, v in out.items()}


def _bridge(kld: float, ladder: list[tuple[float, str]]) -> tuple[float, str]:
    """The perplexity cost of the GGUF type with this KL divergence on the same model,
    log-log between the two nearest measured types; the ends are held, not extrapolated."""
    measured = _measured()
    if kld <= ladder[0][0]:
        return measured[ladder[0][1]], f"at most GGUF {ladder[0][1]}"
    if kld >= ladder[-1][0]:
        return measured[ladder[-1][1]], f"at least GGUF {ladder[-1][1]}"
    for (k0, t0), (k1, t1) in zip(ladder, ladder[1:], strict=False):
        if k0 <= kld <= k1:
            d0, d1 = max(measured[t0], 1e-4), max(measured[t1], 1e-4)
            frac = (math.log(kld) - math.log(k0)) / (math.log(k1) - math.log(k0))
            return math.exp(math.log(d0) + frac * (math.log(d1) - math.log(d0))), (
                f"between GGUF {t0} and {t1}"
            )
    raise AssertionError("unreachable")


def format_delta(name: str) -> tuple[float | None, str]:
    """A format's cost on llama.cpp's scale: the GGUF type with the same measured KL
    divergence on the same model (quality/formats_quality.yaml), averaged over the models
    it was measured on."""
    rows = _format_rows().get(name.lower())
    if not rows:
        return None, "unknown format"
    deltas, where = [], []
    for r in rows:
        ladder = _ladders().get(r["model"])
        if not ladder:
            continue
        delta, span = _bridge(float(r["kld"]), ladder)
        deltas.append(delta)
        where.append(f"{span} by KL divergence on {r['model'].split('/')[-1]}")
    if not deltas:
        return None, "measured, but with no llama.cpp types on the same model to compare"
    mean = math.exp(sum(math.log(max(d, 1e-4)) for d in deltas) / len(deltas))
    return mean, "; ".join(where) + " (measured here, see quality/formats_quality.yaml)"
