"""Cloud fallback (F7): offers from the price catalog, job estimates, and plans that rent.

The CSV rows below are copied from SkyPilot's catalog (catalogs/v8, 2026-09-25), one per
quirk the parser has to survive: RunPod's GpuInfo in GiB, Lambda's in MiB, PrimeIntellect
counting an 8-GPU box twice, GCP with no GpuInfo at all, Vast's zero spot price, a TPU.
"""

from __future__ import annotations

import json

import httpx
import pytest

from rightsize.cloud import cheapest, estimate_job, offers
from rightsize.cloud.offers import fetch_csv, parse
from rightsize.types import Mode, Offer

HEADER = "InstanceType,AcceleratorName,AcceleratorCount,vCPUs,MemoryGiB,Region,SpotPrice,Price,AvailabilityZone,GpuInfo\n"  # noqa: E501

RUNPOD = HEADER + (
    "1x_A100-80GB_SECURE,A100-80GB,1.0,8.0,117.0,CA,1.59,1.59,CA-MTL-1,\"{'Gpus': [{'Name': 'A100 PCIe', 'Count': 1, 'MemoryInfo': {'SizeInMiB': 80}}], 'TotalGpuMemoryInMiB': 80}\"\n"  # noqa: E501
    "1x_RTX4090_SECURE,RTX4090,1.0,16.0,62.0,CA,0.74,0.74,CA-MTL-1,\"{'Gpus': [{'Name': 'RTX 4090', 'Count': 1}], 'TotalGpuMemoryInMiB': 24}\"\n"  # noqa: E501
    "1x_L40S_SECURE,L40S,1.0,16.0,62.0,CA,1.09,1.09,CA-MTL-1,\"{'TotalGpuMemoryInMiB': 48}\"\n"
)
LAMBDA = HEADER + (
    "gpu_8x_a100,A100,8.0,124.0,1800.0,us-east-1,,15.92,,\"{'TotalGpuMemoryInMiB': 40960}\"\n"
    "gpu_1x_h100_pcie,H100,1.0,26.0,200.0,us-east-1,,3.29,,\"{'TotalGpuMemoryInMiB': 81920}\"\n"
)
PRIMEINTELLECT = HEADER + (
    "lambdalabs__8xH100_80GB,H100,8.0,208.0,1800.0,US,,23.92,,\"{'TotalGpuMemoryInMiB': 5242880}\"\n"  # noqa: E501
)
GCP = HEADER + (
    ",A100,1,,,us-central1,1.8563,3.10156,us-central1-a,\n"
    ",T4,1,,,us-central1,0.2088,0.35,us-central1-a,\n"
    ",tpu-v5litepod-1,1,,,us-west4,,1.20,us-west4-a,\n"
)
VAST = HEADER + (
    '1x-L40S-32-65536,L40S,1,32,64,"Bulgaria, BG, EU",0.00,0.60,,"{\'TotalGpuMemoryInMiB\': 46068}"\n'  # noqa: E501
)


def _pool() -> list[Offer]:
    out: list[Offer] = []
    for provider, text in [
        ("runpod", RUNPOD),
        ("lambda", LAMBDA),
        ("primeintellect", PRIMEINTELLECT),
        ("gcp", GCP),
        ("vast", VAST),
    ]:
        out.extend(parse(provider, text, "2026-09-25"))
    return out


def _find(pool, provider, gpu, spot=False) -> Offer:
    return next(o for o in pool if o.provider == provider and o.gpu == gpu and o.spot == spot)


def test_gpu_memory_survives_every_provider_quirk() -> None:
    pool = _pool()
    assert _find(pool, "runpod", "A100-80GB").vram_gib == 80, "RunPod states GiB"
    assert _find(pool, "lambda", "A100").vram_gib == 40, "Lambda states MiB for 8 GPUs"
    assert _find(pool, "primeintellect", "H100").vram_gib == 80, "double count ignored"
    assert _find(pool, "gcp", "A100").vram_gib == 40, "no GpuInfo: the smallest A100"
    assert _find(pool, "vast", "L40S").vram_gib == pytest.approx(45, abs=0.1)


def test_non_gpus_and_zero_prices_are_dropped() -> None:
    pool = _pool()
    assert not any(o.gpu.startswith("tpu-") for o in pool)
    assert not any(o.provider == "vast" and o.spot for o in pool), "a 0.00 spot is no price"
    assert _find(pool, "gcp", "T4", spot=True).usd_per_hour == pytest.approx(0.2088)


def test_offers_know_their_device_and_compute_capability() -> None:
    o = _find(_pool(), "runpod", "RTX4090")
    assert o.device == "RTX 4090 24GB" and o.compute_capability == pytest.approx(8.9)


def test_cheapest_is_single_gpu_big_enough_and_one_per_card() -> None:
    found = cheapest(40, pool=_pool(), top=10)
    assert all(o.gpu_count == 1 and o.vram_gb >= 40 for o in found)
    assert [o.usd_per_hour for o in found] == sorted(o.usd_per_hour for o in found)
    assert ("lambda", "A100") not in {(o.provider, o.gpu) for o in found}, "8-GPU boxes skipped"
    assert (found[0].provider, found[0].gpu) == ("vast", "L40S"), "45 GiB at $0.60"


def test_cheapest_can_require_a_compute_capability() -> None:
    found = cheapest(10, pool=_pool(), min_compute_capability=8.0, top=20)
    assert found and all(o.compute_capability >= 8.0 for o in found)
    assert "T4" not in {o.gpu for o in found}


def test_job_estimate_arithmetic() -> None:
    """6 FLOPs per parameter per token over dense tensor TFLOPS at 35%: 14.8B parameters on
    10M tokens is 8.88e17 FLOPs; an L40S (362.05 TFLOPS) does 1.267e14 a second."""
    offer = _find(_pool(), "vast", "L40S")
    job = estimate_job(14_800_000_000, 10_000_000, offer)
    assert job.hours == pytest.approx(8.88e17 / (362.05e12 * 0.35) / 3600, rel=0.01)
    assert job.usd == pytest.approx(job.hours * 0.60, rel=0.01)
    assert job.confidence == 0.3


def test_a_gpu_without_datasheet_throughput_gets_a_price_but_no_time() -> None:
    offer = Offer(
        provider="x",
        instance_type="y",
        gpu="RTX5880-Ada",
        gpu_count=1,
        vram_gib=48,
        usd_per_hour=0.35,
        source_url="https://example.org",
        fetched_at="2026-09-25",
    )
    job = estimate_job(1_000_000_000, 10_000_000, offer)
    assert job.hours is None and job.usd is None and job.confidence == 0


def test_prices_are_cached_and_read_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path))
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=RUNPOD)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        text, _ = fetch_csv("runpod", client=client)
        assert "A100-80GB" in text and calls
    assert fetch_csv("runpod", offline=True)[0] == text, "offline reads the cache"
    assert fetch_csv("lambda", offline=True) is None, "offline with no cache: nothing"
    pool, missing = offers(["runpod", "lambda"], offline=True)
    assert missing == ["lambda"] and pool


def test_a_fine_tune_that_does_not_fit_is_planned_on_a_rental() -> None:
    from rightsize.hardware import resolve
    from rightsize.rules.recommend import load_candidates, rank

    small = resolve("RTX 3060 12GB")
    qwen = tuple(c for c in load_candidates() if c.repo == "Qwen/Qwen3-14B")
    without = rank(qwen, resolve("RTX 4090"), finetune_device=small, mode=Mode.lora)
    assert not without.plans and "needs" in without.rejected[0].reason
    result = rank(
        qwen, resolve("RTX 4090"), finetune_device=small, mode=Mode.lora, cloud_pool=_pool()
    )
    plan = result.plans[0]
    offer = plan.cloud_fallback["offer"]
    assert plan.steps[0].device.name == f"{offer['provider']} {offer['gpu']}"
    assert plan.steps[0].fit.verdict.value == "fits"
    assert "rent a" in plan.trace[-1] and offer["source_url"] in plan.trace[-1]
    job = plan.cloud_fallback["job_per_10m_tokens"]
    assert job["usd"] is not None, "among offers with a time estimate, the cheapest job wins"


def test_cli_cloud_lists_offers_for_a_memory_need(monkeypatch, capsys) -> None:
    import rightsize.cloud as cloud
    from rightsize.cli import main

    monkeypatch.setattr(cloud, "offers", lambda providers=None, offline=False: (_pool(), []))
    assert main(["--json", "cloud", "--vram", "40", "--top", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["offers"]) == 3 and all(o["vram_gib"] >= 37 for o in out["offers"])
