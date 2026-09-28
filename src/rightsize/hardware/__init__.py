"""Hardware database, presets and local detection (F2).

Plan and TODO: docs/plans/F02-hardware.md
"""

from __future__ import annotations

from rightsize.hardware.db import catalog, get, presets
from rightsize.hardware.probe import detect

__all__ = ["catalog", "detect", "from_hf", "get", "presets"]


def resolve(spec) -> Device:  # noqa: F821 - Device resolved lazily
    """Accept a Device, a preset name, '@hf-username', or 'detect'."""
    from rightsize.types import Device

    if isinstance(spec, Device):
        device = spec
    elif spec is None or str(spec).lower() == "detect":
        device = detect()
    elif str(spec).startswith("@"):  # a Hugging Face profile's saved hardware
        device = from_hf(str(spec))[0]
    else:
        device = get(str(spec))
    return _with_measured_bandwidth(device)


def _with_measured_bandwidth(device):
    """Prefer a bandwidth measured on this machine over anything in a table.

    ``rightsize bench`` records what the device actually achieved. That beats a published
    figure, which describes the model of card rather than this one, and it is the only
    number available for hardware nobody publishes a bandwidth for.
    """
    from rightsize.execution.bench import measured_bandwidth

    gbps = measured_bandwidth(device.name)
    if gbps is None or gbps == device.bandwidth_gbps:
        return device
    return device.model_copy(update={"bandwidth_gbps": gbps})


def from_hf(username: str, **kw):
    """Devices saved on a Hugging Face profile. Imported lazily: it needs the network."""
    from rightsize.hardware.hf import from_hf as _impl

    return _impl(username, **kw)
