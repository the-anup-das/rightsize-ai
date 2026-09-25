"""Opt-in predicted-vs-measured calibration, off by default (F9).

    rightsize calibrate            # compare predictions with what the GPU holds right now
    rightsize telemetry on         # keep those comparisons in a local file
    rightsize telemetry export F   # write them out, to read and share if you choose

Nothing is ever sent. scripts/refit_constants.py turns records into proposed constants.

Plan and TODO: docs/plans/F09-calibration.md
"""

from __future__ import annotations

from rightsize.telemetry.consent import (
    WHAT_IS_RECORDED,
    disable,
    enable,
    enabled,
    status,
    store_path,
)

__all__ = ["WHAT_IS_RECORDED", "disable", "enable", "enabled", "status", "store_path"]
