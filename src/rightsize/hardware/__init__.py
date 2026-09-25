"""Hardware database, presets and local detection (F2).

Plan and TODO: docs/plans/F02-hardware.md
"""

from __future__ import annotations

from rightsize.hardware.db import get, presets
from rightsize.hardware.probe import detect

__all__ = ["detect", "get", "presets"]


def resolve(spec) -> Device:  # noqa: F821 - Device resolved lazily
    """Accept a Device, a preset name, or 'detect'."""
    from rightsize.types import Device

    if isinstance(spec, Device):
        return spec
    if spec is None or str(spec).lower() == "detect":
        return detect()
    return get(str(spec))
