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


def recommend(*args, **kwargs):
    """Hardware-first planning. Lands with F4. See docs/plans/F04-rules-engine.md."""
    _not_yet("recommend", "docs/plans/F04-rules-engine.md")


def recommend_for_model(*args, **kwargs):
    """Model-first planning. Lands with F4. See docs/plans/F04-rules-engine.md."""
    _not_yet("recommend_for_model", "docs/plans/F04-rules-engine.md")


def estimate(
    model,
    quant: str = "Q4_K_M",
    device="detect",
    *,
    ctx: int = 8192,
    runtime: str = "llama.cpp",
    revision: str = "main",
    bandwidth_gbps: float | None = None,
):
    """Memory and speed for one model at one quantization on one device (F3).

        rightsize.estimate("Qwen/Qwen3-4B", "Q4_K_M", "RTX 4090")

    ``model`` is a Hub id or a ``ModelFacts``. ``device`` is a ``Device``, a preset or
    catalogue name, ``"@hf-username"`` for the hardware saved on that profile, or
    ``"detect"``. ``bandwidth_gbps`` fills in a device we have no bandwidth for, which is
    the difference between a speed estimate and none.
    """
    from rightsize.catalog import facts
    from rightsize.fit import estimate as _estimate
    from rightsize.hardware import resolve
    from rightsize.types import ModelFacts

    fx = model if isinstance(model, ModelFacts) else facts(model, revision)
    dev = resolve(device)
    if bandwidth_gbps:
        dev = dev.model_copy(update={"bandwidth_gbps": bandwidth_gbps})
    return _estimate(fx, quant.upper(), dev, runtime=runtime, ctx=ctx)


def detect():
    """This machine as a ``Device``, with a measured bandwidth if ``rightsize bench`` has
    recorded one (F2)."""
    from rightsize.hardware import resolve

    return resolve("detect")
