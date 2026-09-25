"""Cloud GPU rental fallback and job cost (F7).

    from rightsize.cloud import cheapest, estimate_job
    offers = cheapest(40)                          # single-GPU offers with >= 40 GB, cheapest first
    estimate_job(14_800_000_000, 10_000_000, offers[0])

Prices come from SkyPilot's open catalog, read at run time and cached for a day. Advisory
only: nothing is launched (that is F10).

Plan and TODO: docs/plans/F07-cloud-fallback.md
"""

from __future__ import annotations

from rightsize.cloud.job import estimate_job, tensor_tflops
from rightsize.cloud.offers import PROVIDERS, cheapest, device_for, offers

__all__ = ["PROVIDERS", "cheapest", "device_for", "estimate_job", "offers", "tensor_tflops"]
