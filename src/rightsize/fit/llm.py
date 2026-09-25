"""LLM fit estimator, GGUF path (F3, first slice).

formula_id ``llm.gguf.analytic.v0``:
  weights_gb   = params x effective_bpw / 8
  kv_gb        = per-layer KV as llama.cpp allocates it (fit/kv.py): plain GQA is
                 2 x layers x kv_heads x head_dim x ctx x batch x kv_bytes; sliding-window,
                 MLA and hybrid models follow data/runtimes/llama.cpp/kv_cache.yaml
  overhead_gb  = runtime constant + fraction x weights  (llama.cpp ~0.75 GB + 2%)
  vram_gb      = weights + kv + overhead
  tok/s        ~ bandwidth / (weights + kv) bytes read per decoded token, x efficiency
Confidence 0.6 (analytic, uncalibrated). Every number is explainable in ``breakdown``.
"""

from __future__ import annotations

from rightsize._data import load_yaml
from rightsize.fit import kv as _kv
from rightsize.types import Device, FitResult, Mode, ModelFacts, QuantSpec, RuntimeSpec, Verdict

FORMULA_ID = "llm.gguf.analytic.v0"



def overhead_constants(runtime: str) -> tuple[float, float]:
    """(fixed GB, fraction of weights) a runtime adds, from data/runtimes/overheads.yaml,
    which scripts/refit_constants.py refits from measurements. Unknown runtimes get
    llama.cpp's, since most local runtimes wrap it."""
    rows = {r["name"]: r for r in load_yaml("runtimes/overheads.yaml")["runtimes"]}
    row = rows.get(runtime.lower()) or rows["llama.cpp"]
    return float(row["fixed_gb"]), float(row["fraction"])

_HEADROOM = 0.10  # verdict "tight" inside this fraction of usable memory
_DECODE_EFFICIENCY = 0.70  # fraction of peak bandwidth a llama.cpp decode loop reaches
_SLOW_TOK_S = 5.0


def gguf_bpw(quant: str) -> tuple[float, str]:
    """Effective bits-per-weight for a GGUF file type or tensor type, and where it came from.

    llama.cpp's own measurement comes first (data/quality/gguf_types.yaml, effective bits on
    Llama-3.1-8B for every type). Before it was consulted, the i-quants were sized by their
    nominal bits - IQ4_XS at 4.25 against a real 4.46, five percent small - while the
    quality ranking used the effective figure, so two tables disagreed about one number.
    """
    q = quant.upper()
    measured = (load_yaml("quality/gguf_types.yaml").get("file_types") or {}).get(q) or {}
    if measured.get("bpw"):
        return float(measured["bpw"]), "measured by llama.cpp"
    table = load_yaml("quants/gguf_bpw.yaml")
    ft = table.get("file_types") or {}
    if q in ft:
        return float(ft[q]), "file_types"
    tt = table["tensor_types"]
    if q in tt:
        return float(tt[q]), "tensor_types"
    # Q4_K_M / Q4_K_S style names fall back to their base tensor type when unmeasured.
    base = q.rsplit("_", 1)[0] if q.endswith(("_M", "_S", "_L", "_XL", "_XS", "_XXS")) else q
    if base in tt:
        return float(tt[base]), "tensor_types(base)"
    raise KeyError(f"unknown GGUF quant type {quant!r}")


def predicted_file_gb(facts: ModelFacts, quant: str) -> float:
    if not facts.params_total:
        raise ValueError("params_total unknown; cannot predict size")
    bpw, _ = gguf_bpw(quant)
    return facts.params_total * bpw / 8 / 1e9


def kv_cache_gb(facts: ModelFacts, ctx: int, batch: int = 1, kv_bytes: float = 2.0) -> float:
    """Context-dependent memory as llama.cpp allocates it. Delegates to fit.kv, which knows
    sliding-window, MLA and hybrid layouts; plain GQA models come out as before."""
    return _kv.kv_cache_gb(facts, ctx, batch, kv_bytes)


def read_per_token(facts: ModelFacts) -> int:
    """Weights a decode step reads: active parameters, less the input embedding table when
    it is separate from the output head. Looking a token up reads one row of the table; the
    output head is read in full. With tied embeddings they are one matrix, read in full."""
    params = facts.params_active or facts.params_total or 0
    extra = facts.extra or {}
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if vocab and hidden and extra.get("tie_word_embeddings") is False:
        params -= vocab * hidden
    return max(params, 0)


def estimate(
    facts: ModelFacts,
    quant: QuantSpec | str,
    device: Device,
    *,
    runtime: RuntimeSpec | str = "llama.cpp",
    mode: Mode = Mode.infer,
    ctx: int = 8192,
    batch: int = 1,
    kv_bytes: float = 2.0,
) -> FitResult:
    if mode is not Mode.infer:
        # training memory does not depend on the serving quantization; see fit/finetune.py
        from rightsize.fit.finetune import estimate_finetune

        return estimate_finetune(facts, device, mode, batch=batch)
    qname = quant.variant or quant.method if isinstance(quant, QuantSpec) else str(quant)
    rname = runtime.name if isinstance(runtime, RuntimeSpec) else str(runtime)
    fixed, frac = overhead_constants(rname)

    bpw, bpw_source = gguf_bpw(qname)
    weights = facts.params_total * bpw / 8 / 1e9
    kv = kv_cache_gb(facts, ctx, batch, kv_bytes)
    overhead = fixed + frac * weights
    vram = weights + kv + overhead
    usable = device.memory_gb * device.usable_fraction

    notes = [f"bpw {bpw:.3f} from {bpw_source}", f"ctx {ctx}, kv {kv_bytes:g} B/elem"]
    lay = _kv.layout(facts)
    if lay.kind != "gqa":
        notes.append(lay.rule)
    notes.extend(lay.notes)
    if vram <= usable * (1 - _HEADROOM):
        verdict = Verdict.fits
    elif vram <= usable:
        verdict = Verdict.tight
    elif weights * 0.5 + kv + overhead <= usable and device.system_ram_gb:
        verdict = Verdict.offload
        notes.append("does not fit fully; partial offload to system RAM possible")
    else:
        verdict = Verdict.no_fit

    speed = None
    if device.bandwidth_gbps:
        active_gb = read_per_token(facts) * bpw / 8 / 1e9
        speed = _DECODE_EFFICIENCY * device.bandwidth_gbps / (active_gb + kv)
        if verdict is Verdict.offload:
            speed = speed * 0.15  # CPU-bound once layers spill; refit by F9
            notes.append("speed assumes partial offload (large penalty)")
        if speed < _SLOW_TOK_S:
            notes.append(f"below {_SLOW_TOK_S:g} tok/s; fits but will feel slow")
        if facts.params_active:
            # The efficiency constant was measured on a dense model. Routing tokens to
            # experts costs llama.cpp bandwidth efficiency, so for MoE this is a ceiling.
            notes.append("mixture of experts: speed is an upper bound; expert routing is "
                         "usually less bandwidth-efficient than the dense model it was fit on")
    else:
        # Say so rather than showing a blank column. Some devices have no published
        # bandwidth at all - NVIDIA gives laptop GPUs a bus width but no bandwidth,
        # because the memory speed is the laptop maker's choice - so measuring beats
        # hunting for a number that does not exist.
        notes.append(
            "no speed estimate: memory bandwidth unknown for this device. "
            "Measure it with 'rightsize bench', or pass one you trust."
        )

    return FitResult(
        verdict=verdict,
        vram_gb=round(vram, 2),
        ram_gb=0.0,
        breakdown={
            "weights": round(weights, 3),
            "kv_cache": round(kv, 3),
            **{k: v for k, v in _kv.kv_breakdown(facts, ctx, batch, kv_bytes).items() if v},
            "overhead": round(overhead, 3),
            "usable_memory": round(usable, 2),
        },
        speed=round(speed, 1) if speed else None,
        speed_unit="tok/s" if speed else None,
        confidence=0.6,
        formula_id=FORMULA_ID,
        notes=notes,
    )
