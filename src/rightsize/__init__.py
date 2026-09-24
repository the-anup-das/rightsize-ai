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


def estimate(*args, **kwargs):
    """Memory and speed estimate for one (model, quant, runtime, device). Lands with F3."""
    _not_yet("estimate", "docs/plans/F03-fit-engine.md")


def detect():
    """Detect the local machine as a Device. Lands with F2. See docs/plans/F02-hardware.md."""
    _not_yet("detect", "docs/plans/F02-hardware.md")
