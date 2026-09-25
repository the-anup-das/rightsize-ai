"""LLM fit estimator, GGUF path (F3, first slice).

formula_id ``llm.gguf.analytic.v0``:
  weights_gb   = params x effective_bpw / 8
  kv_gb        = 2 x layers x kv_heads x head_dim x ctx x batch x kv_bytes
  overhead_gb  = runtime constant + fraction x weights  (llama.cpp ~0.75 GB + 2%)
  vram_gb      = weights + kv + overhead
  tok/s        ~ bandwidth / (weights + kv) bytes read per decoded token, x efficiency
Confidence 0.6 (analytic, uncalibrated). Every number is explainable in ``breakdown``.
"""

from __future__ import annotations

from rightsize._data import load_yaml
from rightsize.types import Device, FitResult, Mode, ModelFacts, QuantSpec, RuntimeSpec, Verdict

FORMULA_ID = "llm.gguf.analytic.v0"

# Runtime overhead constants (calibration data; refit by F9). GB fixed + fraction of weights.
_OVERHEAD = {
    "llama.cpp": (0.75, 0.02),
    "ollama": (0.90, 0.03),
    "lm studio": (0.80, 0.02),
}
_HEADROOM = 0.10  # verdict "tight" inside this fraction of usable memory
_DECODE_EFFICIENCY = 0.70  # fraction of peak bandwidth a llama.cpp decode loop reaches
_SLOW_TOK_S = 5.0


def gguf_bpw(quant: str) -> tuple[float, str]:
    """Effective bits-per-weight for a GGUF file type or tensor type, and where it came from."""
    table = load_yaml("quants/gguf_bpw.yaml")
    q = quant.upper()
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
    if not (facts.num_layers and facts.num_kv_heads and facts.head_dim):
        raise ValueError("num_layers, num_kv_heads and head_dim are required for the KV cache")
    layers = facts.num_layers
    sw = (facts.extra or {}).get("sliding_window")
    # Sliding-window layers cap their KV at the window; without a per-layer map assume all
    # layers are windowed when the model declares one (conservative for hybrids, noted).
    eff_ctx = min(ctx, sw) if sw else ctx
    return 2 * layers * facts.num_kv_heads * facts.head_dim * eff_ctx * batch * kv_bytes / 1e9


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
        raise NotImplementedError("fine-tune modes land with the full F3; this slice is inference")
    qname = quant.variant or quant.method if isinstance(quant, QuantSpec) else str(quant)
    rname = runtime.name if isinstance(runtime, RuntimeSpec) else str(runtime)
    fixed, frac = _OVERHEAD.get(rname.lower(), _OVERHEAD["llama.cpp"])

    bpw, bpw_source = gguf_bpw(qname)
    weights = facts.params_total * bpw / 8 / 1e9
    kv = kv_cache_gb(facts, ctx, batch, kv_bytes)
    overhead = fixed + frac * weights
    vram = weights + kv + overhead
    usable = device.memory_gb * device.usable_fraction

    notes = [f"bpw {bpw:.3f} from {bpw_source}", f"ctx {ctx}, kv {kv_bytes:g} B/elem"]
    if (facts.extra or {}).get("sliding_window"):
        notes.append("sliding-window model: KV capped at the window for all layers (approximate)")
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
        active = facts.params_active or facts.params_total
        active_gb = active * bpw / 8 / 1e9
        speed = _DECODE_EFFICIENCY * device.bandwidth_gbps / (active_gb + kv)
        if verdict is Verdict.offload:
            speed = speed * 0.15  # CPU-bound once layers spill; refit by F9
            notes.append("speed assumes partial offload (large penalty)")
        if speed < _SLOW_TOK_S:
            notes.append(f"below {_SLOW_TOK_S:g} tok/s; fits but will feel slow")
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
            "overhead": round(overhead, 3),
            "usable_memory": round(usable, 2),
        },
        speed=round(speed, 1) if speed else None,
        speed_unit="tok/s" if speed else None,
        confidence=0.6,
        formula_id=FORMULA_ID,
        notes=notes,
    )
