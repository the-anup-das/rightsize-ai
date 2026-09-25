"""Hardware database, presets and local detection (F2).

Plan and TODO: docs/plans/F02-hardware.md
"""

from __future__ import annotations

from rightsize.hardware.db import catalog, get, presets
from rightsize.hardware.probe import detect

__all__ = ["catalog", "detect", "from_hf", "get", "presets"]


def resolve(spec) -> Device:  # noqa: F821 - Device resolved lazily
    """Accept a Device, a preset name, or 'detect'."""
    from rightsize.types import Device

    if isinstance(spec, Device):
        return spec
    if spec is None or str(spec).lower() == "detect":
        return detect()
    if str(spec).startswith("@"):  # a Hugging Face profile's saved hardware
        return from_hf(str(spec))[0]
    return get(str(spec))


def from_hf(username: str, **kw):
    """Devices saved on a Hugging Face profile. Imported lazily: it needs the network."""
    from rightsize.hardware.hf import from_hf as _impl

    return _impl(username, **kw)
