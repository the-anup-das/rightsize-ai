"""Fit engine: memory and speed estimators per model family (F3).

``estimate`` picks the estimator for the model's family: LLMs (fit/llm.py), diffusion
pipelines (fit/diffusion.py), audio (fit/audio.py), vision and embedding models
(fit/encoder.py). Each result names the formula that produced it.

Plan and TODO: docs/plans/F03-fit-engine.md
"""

from __future__ import annotations

from typing import Any

from rightsize.fit.finetune import estimate_finetune
from rightsize.fit.llm import bits_that_fit, gguf_bpw, kv_cache_gb, predicted_file_gb
from rightsize.fit.quality import ppl_delta
from rightsize.types import Device, Family, FitResult, Mode, ModelFacts

#: The keyword arguments each family's estimator takes. Others are ignored, so one call
#: shape (the CLI's, the MCP tool's) works for every family.
FAMILY_ARGS: dict[Family, frozenset[str]] = {
    Family.llm: frozenset({"runtime", "mode", "ctx", "batch", "kv_bytes", "all_logits"}),
    Family.diffusion: frozenset(
        {"offload", "resolution", "batch", "frames", "text_encoder_quant", "vae_slicing"}
    ),
    Family.audio: frozenset({"runtime", "batch"}),
    Family.vision: frozenset({"batch"}),
    Family.embedding: frozenset({"batch", "seq_len"}),
}

#: The quantization each family is sized at when none is given.
DEFAULT_QUANT = {
    Family.llm: "Q4_K_M",
    Family.diffusion: "bf16",
    Family.audio: None,  # the runtime's own default: f16 for whisper.cpp, fp16 otherwise
    Family.vision: "fp16",
    Family.embedding: "fp16",
}


def estimate(
    facts: ModelFacts, quant: Any = None, device: Device | None = None, **kwargs: Any
) -> FitResult:
    """Memory, and speed where there is a model for it, for one model on one device."""
    if device is None:
        raise TypeError("estimate() needs a device")
    family = facts.family
    mode = Mode(kwargs.get("mode") or Mode.infer)
    if family is not Family.llm and mode is not Mode.infer:
        from rightsize.errors import NotImplementedYet

        raise NotImplementedYet(
            f"fine-tuning memory for {family.value} models", "docs/plans/F03-fit-engine.md"
        )
    args = {k: v for k, v in kwargs.items() if k in FAMILY_ARGS[family] and v is not None}
    if quant is None:
        # a GGUF repo is sized at the file its facts were read from
        quant = ((facts.extra or {}).get("gguf") or {}).get("quant") or DEFAULT_QUANT[family]
    if family is Family.diffusion:
        from rightsize.fit import diffusion

        return diffusion.estimate(facts, quant, device, **args)
    if family is Family.audio:
        from rightsize.fit import audio

        return audio.estimate(facts, quant, device, **args)
    if family in (Family.vision, Family.embedding):
        from rightsize.fit import encoder

        return encoder.estimate(facts, quant, device, **args)
    from rightsize.fit import llm

    return llm.estimate(facts, quant, device, **args)


__all__ = [
    "DEFAULT_QUANT",
    "FAMILY_ARGS",
    "bits_that_fit",
    "estimate",
    "estimate_finetune",
    "gguf_bpw",
    "kv_cache_gb",
    "ppl_delta",
    "predicted_file_gb",
]
