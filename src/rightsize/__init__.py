"""Rightsize: quantize and fit any model to your hardware.

Public API (see docs/plans/F06-surfaces.md):

    rightsize.recommend(...)            -> list[Plan]   (F4)
    rightsize.recommend_for_model(...)  -> list[Plan]   (F4)
    rightsize.estimate(...)             -> FitResult    (F3)
    rightsize.detect()                  -> Device       (F2)

Lightweight rule: importing this package must not import pydantic, httpx,
yaml or any extra. Types are resolved lazily through ``__getattr__`` so that
``import rightsize`` costs only the standard library. ``from rightsize import
Plan`` pays for pydantic on first use.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Names that live in rightsize.types and are re-exported lazily.
_TYPE_EXPORTS = frozenset(
    {
        "Device",
        "Family",
        "FitResult",
        "JobEstimate",
        "ModelFacts",
        "ModelRef",
        "Offer",
        "Mode",
        "Plan",
        "PlanStep",
        "Provenance",
        "QuantSpec",
        "RuntimeSpec",
        "Verdict",
    }
)
_ERROR_EXPORTS = frozenset({"MissingExtraError", "NotImplementedYet", "RightsizeError"})

__all__ = [
    "__version__",
    "recommend",
    "recommend_for_model",
    "estimate",
    "detect",
    *sorted(_ERROR_EXPORTS),
    *sorted(_TYPE_EXPORTS),
]


def __getattr__(name: str):
    # PEP 562 lazy attribute hook: keeps `import rightsize` stdlib-only.
    if name in _TYPE_EXPORTS:
        from rightsize import types as _types

        return getattr(_types, name)
    if name in _ERROR_EXPORTS:
        from rightsize import errors as _errors

        return getattr(_errors, name)
    raise AttributeError(f"module 'rightsize' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def _not_yet(feature: str, plan: str):
    from rightsize.errors import NotImplementedYet

    raise NotImplementedYet(feature, plan)


def recommend(task: str = "chat", target_device="detect", **kwargs):
    """Hardware-first: the best models for a device, one ranked Plan each (F4).

        rightsize.recommend("coding", "RTX 3060 12GB")
        rightsize.recommend("chat", "detect", finetune_device="T4", mode="qlora")

    Each Plan's ``trace`` explains the fit, the score and every rule that fired, with its
    source; ``plan.render()`` gives the commands. See rules.recommend for every option."""
    from rightsize.rules.recommend import recommend as _recommend

    return _recommend(task, target_device, **kwargs)


def recommend_for_model(model, target_device="detect", **kwargs):
    """Model-first: every quantization of one model that works on a device, best first."""
    from rightsize.rules.recommend import recommend_for_model as _for_model

    return _for_model(model, target_device, **kwargs).plans


def estimate(
    model,
    quant=None,
    device="detect",
    *,
    ctx: int = 8192,
    runtime: str | None = None,
    revision: str = "main",
    bandwidth_gbps: float | None = None,
    mode: str = "infer",
    batch: int | None = None,
    offload: str = "none",
    resolution=None,
    frames: int | None = None,
    text_encoder_quant: str | None = None,
    vae_slicing: bool = False,
    seq_len: int | None = None,
    file: str | None = None,
):
    """Memory, and speed where there is a model for it, for one model on one device (F3).

        rightsize.estimate("Qwen/Qwen3-4B", "Q4_K_M", "RTX 4090")
        rightsize.estimate("black-forest-labs/FLUX.1-dev", "nf4", "RTX 4070",
                           text_encoder_quant="nf4", offload="model")
        rightsize.estimate("openai/whisper-large-v3", "int8", "RTX 3060 12GB")

    ``model`` is a Hub id or a ``ModelFacts``; its family (LLM, diffusion, audio, vision,
    embedding) picks the estimator. A repo of GGUFs is read from the header of one file,
    ``file`` or else its Q4_K_M, and sized from its own file for each ``quant`` it holds.
    ``quant`` is a GGUF type or a format such as bf16, fp8, nf4, int8, int4, awq,
    mlx-4bit; left out, each family has its default (Q4_K_M for LLMs, bf16 for
    diffusion), and a GGUF repo the file it was read from. ``device`` is a ``Device``, a
    preset or catalogue name,
    ``"@hf-username"`` for the hardware saved on that profile, or ``"detect"``.

    LLMs: ``ctx``, ``runtime``, ``mode`` (``"lora"``, ``"qlora"``, ``"full"`` for the
    memory to fine-tune), and ``bandwidth_gbps`` for a device with no known bandwidth.
    Diffusion: ``resolution`` ("1024x1024" or a (width, height) pair), ``batch``,
    ``offload`` (none, model, sequential), ``frames`` for video, ``text_encoder_quant``,
    ``vae_slicing``. Audio: ``runtime`` (whisper.cpp, faster-whisper, transformers),
    ``batch``. Embeddings: ``batch``, ``seq_len``. Arguments for another family are
    ignored.
    """
    from rightsize.catalog import facts
    from rightsize.fit import estimate as _estimate
    from rightsize.hardware import resolve
    from rightsize.types import Mode, ModelFacts

    fx = model if isinstance(model, ModelFacts) else facts(model, revision, file=file)
    dev = resolve(device)
    if bandwidth_gbps:
        dev = dev.model_copy(update={"bandwidth_gbps": bandwidth_gbps})
    return _estimate(
        fx, quant, dev, runtime=runtime, ctx=ctx, mode=Mode(mode), batch=batch,
        offload=offload, resolution=_resolution(resolution), frames=frames,
        text_encoder_quant=text_encoder_quant, vae_slicing=vae_slicing or None,
        seq_len=seq_len,
    )


def _resolution(value) -> tuple[int, int] | None:
    """(width, height) from "1024x1024", "1024", or a pair; None passes through."""
    if value is None or isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    text = str(value).lower().replace(" ", "")
    width, _, height = text.partition("x")
    return int(width), int(height or width)


def detect():
    """This machine as a ``Device``, with a measured bandwidth if ``rightsize bench`` has
    recorded one (F2)."""
    from rightsize.hardware import resolve

    return resolve("detect")
