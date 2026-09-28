"""Rental GPU prices from SkyPilot's open catalog (F7).

SkyPilot keeps one CSV of instance prices per provider, refreshed from the providers' own
pricing APIs, at github.com/skypilot-org/skypilot-catalog. Rightsize reads those files at
run time into ~/.cache/rightsize/cloud/ (one day by default) and never ships a copy.

GPU memory needs care. The catalog's GpuInfo "TotalGpuMemoryInMiB" is MiB for most
providers and GiB for RunPod; it is the instance's total at AWS but one GPU's at Lambda;
some PrimeIntellect rows count it twice; elsewhere it is absent. And one name can mean two
cards ("A100" is 40 GB at Lambda and GCP, 80 GB at Vast). So both readings of GpuInfo are
tried and the one matching a known size of that card wins, then a plausible one, then a
size in the name, then the smallest variant rightsize knows: an offer is never sized larger
than the card might be.
"""

from __future__ import annotations

import ast
import csv
import datetime as dt
import functools
import io
import os
import re
import time
from pathlib import Path

from rightsize.types import GB, GIB, Offer

CATALOG_URL = (
    "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/"
    "{provider}/vms.csv"
)
#: Providers read by default. AWS is left for asking by name: its file is 5 MB, and its
#: single-GPU prices are rarely the lowest.
PROVIDERS = (
    "runpod",
    "lambda",
    "vast",
    "nebius",
    "hyperstack",
    "cudo",
    "fluidstack",
    "paperspace",
    "shadeform",
    "primeintellect",
    "do",
    "azure",
    "gcp",
)
_NOT_A_GPU = re.compile(r"^(tpu-|inferentia|trainium|virtex|gaudi)", re.IGNORECASE)
_SIZE = re.compile(r"-(\d+)GB?(?:-|$)", re.IGNORECASE)
_FORM = re.compile(r"-(SXM\d?|PCIE|NVLINK|NVL|MEGA|WK|WS|SE)\b", re.IGNORECASE)
#: How the catalog spells some cards, against rightsize's names.
_SPELLING = {
    "RTXA6000": "RTX A6000",
    "RTX-A6000": "RTX A6000",
    "A6000": "RTX A6000",
    "RTXA5000": "RTX A5000",
    "A5000": "RTX A5000",
    "RTXA4000": "RTX A4000",
    "RTX-A4000": "RTX A4000",
    "A4000": "RTX A4000",
    "RTXA4500": "RTX A4500",
    "RTX6000-Ada": "RTX 6000 Ada",
    "RTX6000Ada": "RTX 6000 Ada",
    "RTX4000-Ada": "RTX 4000 Ada",
    "RTX2000-Ada": "RTX 2000 Ada",
    "RTX5880-Ada": "RTX 5880 Ada",
}


def _cache_dir() -> Path:
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    return base / "cloud"


def fetch_csv(
    provider: str, *, ttl_s: float = 24 * 3600, offline: bool = False, client=None
) -> tuple[str, str] | None:
    """(csv text, fetched date) for one provider, from the cache when it is fresh enough.
    None when there is neither a cache nor a network answer. RIGHTSIZE_OFFLINE=1 is the
    same as ``offline=True``."""
    offline = offline or os.environ.get("RIGHTSIZE_OFFLINE") == "1"
    path = _cache_dir() / f"{provider}.csv"
    fresh = path.exists() and (offline or time.time() - path.stat().st_mtime < ttl_s)
    if fresh:
        day = dt.date.fromtimestamp(path.stat().st_mtime).isoformat()
        return path.read_text(encoding="utf-8"), day
    if offline:
        return None
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        r = client.get(CATALOG_URL.format(provider=provider))
        if r.status_code != 200:
            return None
        text = r.text
    except httpx.HTTPError:
        return (path.read_text(encoding="utf-8"), "stale cache") if path.exists() else None
    finally:
        if own:
            client.close()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass
    return text, dt.date.today().isoformat()


def _gpuinfo_gib(info: str, count: int, sizes: tuple[float, ...] = ()) -> float | None:
    """Per-GPU memory from GpuInfo, or None when it is missing or implausible.

    ``sizes`` are the memory sizes rightsize knows for this card; a reading that matches
    one is taken over one that is merely plausible."""
    if not info:
        return None
    try:
        total = ast.literal_eval(info).get("TotalGpuMemoryInMiB")
    except (ValueError, SyntaxError, AttributeError):
        return None
    if not total:
        return None

    def gib(x: float) -> float:
        return x / 1024 if x >= 1024 else x  # MiB for most providers, GiB for RunPod

    readings = [gib(float(total) / count), gib(float(total))] if count > 1 else [gib(total)]
    for g in readings:
        if any(abs(g - s) < 1.0 for s in sizes):
            return g
    plausible = [g for g in readings if 6 <= g <= 300]  # PrimeIntellect's count twice
    return plausible[0] if plausible else None


def _base_name(gpu: str) -> str:
    name = _FORM.sub("", _SIZE.sub("-", gpu)).strip("-")
    name = _SPELLING.get(name, name)
    return re.sub(r"^RTX(\d)", r"RTX \1", name)


@functools.lru_cache(maxsize=512)
def _variants(gpu: str) -> tuple:
    """Every rightsize device a catalog name could be: "A100" is the 40 and 80 GB cards."""
    from rightsize.hardware import catalog

    base = _base_name(gpu).lower()
    return tuple(
        dev for name, dev in catalog().items() if re.sub(r"\s+\d+GB$", "", name).lower() == base
    )


@functools.lru_cache(maxsize=512)
def device_for(gpu: str, vram_gib: float | None = None):
    """The rightsize device a catalog name refers to: the variant with this memory when
    it is known, else the smallest one."""
    variants = _variants(gpu)
    if not variants:
        return None
    if vram_gib:
        same = [d for d in variants if abs(d.memory_gib - vram_gib) < 1]
        if same:
            return same[0]
    return min(variants, key=lambda d: d.memory_gib)


def parse(provider: str, text: str, fetched_at: str) -> list[Offer]:
    """Every GPU offer in one provider's CSV: on demand, and spot where it is priced."""
    out: list[Offer] = []
    source = CATALOG_URL.format(provider=provider)
    for row in csv.DictReader(io.StringIO(text)):
        gpu = (row.get("AcceleratorName") or "").strip()
        if not gpu or _NOT_A_GPU.match(gpu):
            continue
        try:
            count = int(float(row.get("AcceleratorCount") or 0))
        except ValueError:
            continue
        if count < 1:
            continue
        stated = _SIZE.search(gpu)
        sizes = tuple(d.memory_gib for d in _variants(gpu))
        vram = _gpuinfo_gib(row.get("GpuInfo") or "", count, sizes) or (
            float(stated.group(1)) if stated else None
        )
        dev = device_for(gpu, vram)
        vram = vram or (dev.memory_gib if dev else None)
        if not vram:
            continue
        # the chip family gives the compute capability; the name only when memory matches
        # (Vast's 94 GiB "H100" is an H100 NVL, not the 80 GB card)
        named = dev.name if dev and abs(dev.memory_gib - vram) < 1 else None
        for price_key, spot in (("Price", False), ("SpotPrice", True)):
            try:
                price = float(row.get(price_key) or 0)
            except ValueError:
                continue
            if price <= 0:
                continue
            out.append(
                Offer(
                    provider=provider,
                    instance_type=row.get("InstanceType") or "",
                    gpu=gpu,
                    gpu_count=count,
                    vram_gib=vram,
                    usd_per_hour=price,
                    spot=spot,
                    region=row.get("Region") or None,
                    device=named,
                    compute_capability=dev.compute_capability if dev else None,
                    source_url=source,
                    fetched_at=fetched_at,
                )
            )
    return out


def offers(
    providers: tuple[str, ...] | list[str] | None = None,
    *,
    offline: bool = False,
    ttl_s: float = 24 * 3600,
) -> tuple[list[Offer], list[str]]:
    """Every offer from these providers, and the providers that could not be read."""
    found: list[Offer] = []
    missing: list[str] = []
    for p in providers or PROVIDERS:
        got = fetch_csv(p, ttl_s=ttl_s, offline=offline)
        if got is None:
            missing.append(p)
            continue
        found.extend(parse(p, *got))
    return found, missing


def cheapest(
    min_vram_gb: float,
    *,
    providers: tuple[str, ...] | list[str] | None = None,
    spot: bool = False,
    min_compute_capability: float | None = None,
    top: int = 5,
    offline: bool = False,
    pool: list[Offer] | None = None,
) -> list[Offer]:
    """The cheapest single-GPU offers with at least this much memory, one per provider and
    card, lowest price first. ``min_vram_gb`` is decimal GB, what the fit engine predicts.

    ``pool`` skips the fetch (tests, or a caller that already has the offers)."""
    if pool is None:
        pool, _ = offers(providers, offline=offline)
    best: dict[tuple[str, str], Offer] = {}
    for o in pool:
        if o.gpu_count != 1 or o.spot != spot or o.vram_gib * GIB / GB < min_vram_gb:
            continue
        if min_compute_capability and not (
            o.compute_capability and o.compute_capability >= min_compute_capability
        ):
            continue
        key = (o.provider, o.gpu)
        if key not in best or o.usd_per_hour < best[key].usd_per_hour:
            best[key] = o
    return sorted(best.values(), key=lambda o: (o.usd_per_hour, o.provider))[:top]
