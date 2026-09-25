"""Hardware presets from the data seed (F2, first slice)."""

from __future__ import annotations

import re

from rightsize._data import load_yaml
from rightsize.types import Device, Provenance


def _to_device(rec: dict) -> Device:
    prov = rec.get("provenance")
    return Device(
        name=rec["name"],
        vendor=rec.get("vendor", "other"),
        memory_gb=rec["memory_gb"],
        system_ram_gb=rec.get("system_ram_gb"),
        bandwidth_gbps=rec.get("bandwidth_gbps"),
        compute_arch=rec.get("compute_arch"),
        backends=list(rec.get("backends", [])),
        os=rec.get("os", "unknown"),
        usable_fraction=rec.get("usable_fraction", 1.0),
        provenance=Provenance(
            source_url=prov["source_url"], fetched_at=str(prov["fetched_at"]), note=prov.get("note")
        )
        if prov
        else None,
    )


def presets() -> dict[str, Device]:
    data = load_yaml("hardware/presets.yaml")
    return {rec["name"]: _to_device(rec) for rec in data["presets"]}


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(nvidia|geforce|amd|radeon|apple|intel|laptop gpu)\b", " ", s)
    s = re.sub(r"(\d+)\s*gb\b", r"\1gb", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def get(name: str) -> Device:
    """Fuzzy preset lookup: 'RTX 4090', '4090', 'GeForce RTX 4090 24GB' resolve to one record."""
    table = presets()
    if name in table:
        return table[name]
    q = _norm(name)
    q_tokens = set(q.split())
    scored: list[tuple[int, str]] = []
    for key in table:
        k_tokens = set(_norm(key).split())
        if q_tokens <= k_tokens or k_tokens <= q_tokens:
            scored.append((len(q_tokens & k_tokens), key))
    if not scored:
        raise KeyError(f"no preset matches {name!r}; known: {sorted(table)}")
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise KeyError(f"{name!r} is ambiguous: {[k for _, k in scored[:3]]}")
    return table[scored[0][1]]
