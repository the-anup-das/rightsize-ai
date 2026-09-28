"""Vision and embedding fit estimator (F3).

formula_id ``encoder.weights.v0``. These models run one forward pass per input with no KV
cache, so memory is the weights plus one batch of activations:

  weights      = params x bits per weight / 8
  activations  = batch x tokens x hidden x 2 bytes x 16
                 tokens: the sequence length for text, (image size / patch)^2 for a ViT
  overhead     = the CUDA context on a GPU (0.6 GB), the Python runtime on a CPU (0.3 GB)

The factor of 16 covers the few activations alive at once in a transformer layer (input,
attention projections, the MLP's intermediate). It is a round figure, not a measurement,
so confidence is low; a CNN with no token count gets a flat 0.25 GB per image.
"""

from __future__ import annotations

from rightsize.fit.formats import lookup
from rightsize.types import Device, Family, FitResult, ModelFacts, Verdict

FORMULA_ID = "encoder.weights.v0"
_HEADROOM = 0.10
_LIVE_ACTIVATIONS = 16
_FIXED = {"gpu": 0.6, "cpu": 0.3}
_CNN_PER_IMAGE_GB = 0.25


def estimate(
    facts: ModelFacts,
    quant: str | None,
    device: Device,
    *,
    batch: int | None = None,
    seq_len: int = 512,
) -> FitResult:
    params = facts.params_total or 0
    if not params:
        raise ValueError("parameter count unknown; cannot size this model")
    extra = facts.extra or {}
    embedding = facts.family is Family.embedding
    batch = batch or (32 if embedding else 1)
    fmt = lookup(quant or "fp16", gguf="tensor")
    weights = params * fmt.bpw / 8 / 1e9
    hidden = extra.get("hidden_size") or 0

    notes = [f"{fmt.id} at {fmt.bpw:g} bpw: {fmt.source}"]
    image, patch = extra.get("image_size"), extra.get("patch_size")
    if embedding and hidden:
        tokens = seq_len
        notes.append(f"batch {batch} x {seq_len} tokens")
    elif image and patch and hidden:
        tokens = (int(image) // int(patch)) ** 2 + 1
        notes.append(f"batch {batch} images at {image}px, {tokens} patches")
    else:
        tokens = 0
    if tokens:
        acts = batch * tokens * hidden * 2 * _LIVE_ACTIVATIONS / 1e9
    else:
        acts = batch * _CNN_PER_IMAGE_GB
        notes.append("no token count in the config; a flat allowance per input")
    where = "cpu" if device.vendor == "cpu" else "gpu"
    overhead = _FIXED[where]
    vram = weights + acts + overhead
    usable = device.memory_gb * device.usable_fraction
    if vram <= usable * (1 - _HEADROOM):
        verdict = Verdict.fits
    else:
        verdict = Verdict.tight if vram <= usable else Verdict.no_fit
    if embedding:
        notes.append(
            "the index is separate: int8 output vectors store 4x smaller than "
            "float32 and binary ones 32x, a saving on disk and in the vector store"
        )
    notes.append("no speed estimate for encoders yet")
    return FitResult(
        verdict=verdict,
        vram_gb=round(vram, 2),
        breakdown={
            "weights": round(weights, 3),
            "activations": round(acts, 3),
            "overhead": round(overhead, 3),
            "usable_memory": round(usable, 2),
        },
        confidence=min(0.4, facts.confidence),
        formula_id=FORMULA_ID,
        notes=notes,
    )
