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
        "ModelFacts",
        "ModelRef",
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
    quant: str = "Q4_K_M",
    device="detect",
    *,
    ctx: int = 8192,
    runtime: str = "llama.cpp",
    revision: str = "main",
    bandwidth_gbps: float | None = None,
    mode: str = "infer",
):
    """Memory and speed for one model at one quantization on one device (F3).

        rightsize.estimate("Qwen/Qwen3-4B", "Q4_K_M", "RTX 4090")

    ``model`` is a Hub id or a ``ModelFacts``. ``device`` is a ``Device``, a preset or
    catalogue name, ``"@hf-username"`` for the hardware saved on that profile, or
    ``"detect"``. ``bandwidth_gbps`` fills in a device we have no bandwidth for, which is
    the difference between a speed estimate and none. ``mode`` is ``"infer"``, or
    ``"lora"``, ``"qlora"`` or ``"full"`` for the memory to fine-tune it.
    """
    from rightsize.catalog import facts
    from rightsize.fit import estimate as _estimate
    from rightsize.hardware import resolve
    from rightsize.types import Mode, ModelFacts

    fx = model if isinstance(model, ModelFacts) else facts(model, revision)
    dev = resolve(device)
    if bandwidth_gbps:
        dev = dev.model_copy(update={"bandwidth_gbps": bandwidth_gbps})
    return _estimate(fx, quant.upper(), dev, runtime=runtime, ctx=ctx, mode=Mode(mode))


def detect():
    """This machine as a ``Device``, with a measured bandwidth if ``rightsize bench`` has
    recorded one (F2)."""
    from rightsize.hardware import resolve

    return resolve("detect")
