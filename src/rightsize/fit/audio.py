"""Audio fit estimator: Whisper-family speech recognition, and other audio models (F3).

formula_id ``audio.whisper.v0`` for Whisper models:
  weights   = params x bits per weight / 8     (f16, int8, q5_0, ... at the runtime's type)
  overhead  = the runtime's measured overhead at a reference size, scaled by
              (params / reference params) ** size_exponent
  batch     = the measured cost of each extra item in a batch, scaled the same way
  vram      = weights + overhead + batch       (the GPU rows include the CUDA context)
The rows and the exponent are in data/runtimes/whisper/memory.yaml; the tests replay them.

formula_id ``audio.weights.v0`` for anything else (TTS, other speech recognizers): weights
plus a fixed runtime allowance, with low confidence, because nothing measured backs it.
"""

from __future__ import annotations

from rightsize._data import load_yaml
from rightsize.fit.formats import lookup
from rightsize.types import Device, FitResult, ModelFacts, Verdict

FORMULA_ID = "audio.whisper.v0"
GENERIC_ID = "audio.weights.v0"
RUNTIMES = ("whisper.cpp", "faster-whisper", "transformers", "openai-whisper")
_HEADROOM = 0.10
# Generic path: the CUDA context on a GPU, the Python runtime on a CPU, plus a share of
# the weights for activations. Round figures, not measurements.
_GENERIC_FIXED = {"gpu": 0.6, "cpu": 0.3}
_GENERIC_ACTIVATION_SHARE = 0.2


def _table() -> dict:
    return load_yaml("runtimes/whisper/memory.yaml")


def default_runtime(device: Device, quant: str | None) -> str:
    """whisper.cpp for ggml types and on Apple silicon (Metal); faster-whisper otherwise."""
    q = (quant or "").lower()
    if device.vendor == "apple" or q.startswith(("q4", "q5", "q8")):
        return "whisper.cpp"
    return "faster-whisper"


def _is_whisper(facts: ModelFacts) -> bool:
    return str((facts.extra or {}).get("model_type") or "").lower() == "whisper"


def _where(device: Device) -> str:
    return "cpu" if device.vendor == "cpu" else "gpu"


def _overhead_gb(runtime: str, where: str, params: int, batch: int) -> tuple[float, str]:
    """Measured overhead for this runtime, scaled to this model size."""
    table = _table()
    rows = [r for r in table["measurements"]["rows"] if r["runtime"] == runtime]
    local = [r for r in rows if r["device"] == where] or rows
    if not local:
        raise KeyError(f"no measurements for runtime {runtime!r}")
    exp = table["size_exponent"]
    singles = [r for r in local if r["batch"] == 1]
    per_row = []
    for r in singles:
        w = r["params"] * lookup(r["precision"], gguf="tensor").bpw / 8 / 1e9
        per_row.append((r["memory_mb"] / 1000 - w) * (params / r["params"]) ** exp)
    overhead = sum(per_row) / len(per_row)
    how = f"{runtime} on {local[0]['device']}, measured at {local[0]['model']}"
    if batch > 1:
        extra = []
        for r in local:
            if r["batch"] > 1:
                one = next((s for s in singles if s["precision"] == r["precision"]), None)
                if one:
                    step = (r["memory_mb"] - one["memory_mb"]) / 1000 / (r["batch"] - 1)
                    extra.append(step * (params / r["params"]) ** exp)
        if extra:
            overhead += (batch - 1) * sum(extra) / len(extra)
        else:
            how += "; no batched measurement, batch cost not included"
    return overhead, how


def _verdict(vram: float, usable: float) -> Verdict:
    if vram <= usable * (1 - _HEADROOM):
        return Verdict.fits
    return Verdict.tight if vram <= usable else Verdict.no_fit


def estimate(
    facts: ModelFacts,
    quant: str | None,
    device: Device,
    *,
    runtime: str | None = None,
    batch: int = 1,
) -> FitResult:
    params = facts.params_total or 0
    if not params:
        raise ValueError("parameter count unknown; cannot size this model")
    usable = device.memory_gb * device.usable_fraction
    where = _where(device)

    if not _is_whisper(facts):
        fmt = lookup(quant or "fp16", gguf="tensor")
        weights = params * fmt.bpw / 8 / 1e9
        overhead = _GENERIC_FIXED[where] + _GENERIC_ACTIVATION_SHARE * weights
        vram = weights + overhead
        return FitResult(
            verdict=_verdict(vram, usable),
            vram_gb=round(vram, 2),
            breakdown={"weights": round(weights, 3), "overhead": round(overhead, 3),
                       "usable_memory": round(usable, 2)},
            confidence=min(0.3, facts.confidence),
            formula_id=GENERIC_ID,
            notes=[f"{fmt.id} at {fmt.bpw:g} bpw: {fmt.source}",
                   "weights plus a fixed allowance; no measurement backs this family yet"],
        )

    rt = runtime if runtime in RUNTIMES else default_runtime(device, quant)
    fmt = lookup(quant or ("f16" if rt == "whisper.cpp" else "fp16"), gguf="tensor")
    weights = params * fmt.bpw / 8 / 1e9
    overhead, how = _overhead_gb(rt, where, params, batch)
    vram = weights + overhead
    measured = any(r["runtime"] == rt and r["device"] == where
                   for r in _table()["measurements"]["rows"])
    notes = [
        f"{rt}, {fmt.id} at {fmt.bpw:g} bpw: {fmt.source}",
        f"overhead from {how}, scaled by model size",
        "beam size 5, as measured; greedy decoding needs a little less",
        "no speed estimate for speech recognition yet",
    ]
    if runtime and runtime not in RUNTIMES:
        notes.append(f"runtime {runtime!r} has no Whisper data; used {rt}")
    return FitResult(
        verdict=_verdict(vram, usable),
        vram_gb=round(vram, 2),
        breakdown={"weights": round(weights, 3), "overhead": round(overhead, 3),
                   "usable_memory": round(usable, 2)},
        confidence=min(0.6 if measured else 0.4, facts.confidence),
        formula_id=FORMULA_ID,
        notes=notes,
    )
