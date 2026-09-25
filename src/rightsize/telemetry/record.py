"""Calibration records: one predicted-vs-measured comparison each, built from an allowlist (F9).

A record is assembled field by field from values rightsize chose to keep, never by copying
an object that might carry more. Every sub-model forbids extra keys, so a field nobody
reviewed cannot slip in. The tests feed paths, user names and a private repo through and
check none of them comes out.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rightsize.telemetry.consent import enabled, store_path
from rightsize.types import Device, FitResult, ModelFacts

SCHEMA_VERSION = 1


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordDevice(_Closed):
    vendor: str
    name: str = Field(description="a catalog or preset name, or a generic class")
    memory_gib: float
    compute_capability: float | None = None
    os: str


class RecordRuntime(_Closed):
    name: str
    version: str | None = None


class RecordModel(_Closed):
    repo: str | None = Field(description="the Hub id when the repo is public, else None")
    params: int | None = None
    model_type: str | None = None
    quant: str | None = None
    ctx: int | None = None


class RecordNumbers(_Closed):
    vram_gb: float | None = None
    weights_gb: float | None = None
    kv_gb: float | None = None
    overhead_gb: float | None = None
    tok_s: float | None = None
    formula_id: str | None = None


class CalibrationRecord(_Closed):
    schema_version: int = SCHEMA_VERSION
    rightsize_version: str
    recorded_on: str = Field(description="the date only")
    source: Literal["lm studio", "ollama", "manifest", "manual"]
    device: RecordDevice
    runtime: RecordRuntime
    model: RecordModel
    predicted: RecordNumbers
    measured: RecordNumbers


def _device_name(device: Device) -> str:
    """The device's name when rightsize knows it by that name; else only its class, so a
    name built from local details never lands in a record."""
    from rightsize.hardware import catalog, get, presets

    known = set(presets()) | set(catalog())
    if device.name in known:
        return device.name
    try:  # a detected card reads "NVIDIA GeForce RTX 4070 Ti SUPER": map it to the catalog
        match = get(device.name)
    except (KeyError, ValueError):
        match = None
    if match is not None and abs(match.memory_gib - device.memory_gib) < 1:
        return match.name
    return f"{device.vendor} {device.memory_gib:g} GiB"


def build(
    *,
    source: str,
    device: Device,
    runtime: str,
    runtime_version: str | None = None,
    facts: ModelFacts | None,
    quant: str | None,
    ctx: int | None,
    predicted: FitResult | None,
    measured_vram_gb: float | None,
    measured_tok_s: float | None = None,
) -> CalibrationRecord:
    from rightsize import __version__

    public = bool(facts) and (facts.extra or {}).get("private") is False
    extra = (facts.extra or {}) if facts else {}
    pred = RecordNumbers()
    if predicted is not None:
        b = predicted.breakdown
        pred = RecordNumbers(
            vram_gb=predicted.vram_gb, weights_gb=b.get("weights"), kv_gb=b.get("kv_cache"),
            overhead_gb=b.get("overhead"), tok_s=predicted.speed,
            formula_id=predicted.formula_id,
        )
    return CalibrationRecord(
        rightsize_version=__version__,
        recorded_on=dt.date.today().isoformat(),
        source=source,
        device=RecordDevice(
            vendor=device.vendor, name=_device_name(device), memory_gib=device.memory_gib,
            compute_capability=device.compute_capability, os=device.os,
        ),
        runtime=RecordRuntime(name=runtime, version=runtime_version),
        model=RecordModel(
            repo=facts.ref.repo if public else None,
            params=facts.params_total if facts else None,
            model_type=extra.get("model_type"),
            quant=quant,
            ctx=ctx,
        ),
        predicted=pred,
        measured=RecordNumbers(vram_gb=measured_vram_gb, tok_s=measured_tok_s),
    )


def append(record: CalibrationRecord) -> bool:
    """Add a record to the local store, if recording is on. True when it was written."""
    if not enabled():
        return False
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
    return True


def read(path: str | Path | None = None) -> list[CalibrationRecord]:
    p = Path(path) if path else store_path()
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(CalibrationRecord.model_validate_json(line))
    return out


def export(dest: str | Path) -> int:
    """Copy the records to a file the user can read and share. Returns how many."""
    records = read()
    Path(dest).write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )
    return len(records)


def as_dict(record: CalibrationRecord) -> dict[str, Any]:
    return json.loads(record.model_dump_json())
