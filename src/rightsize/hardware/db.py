"""Hardware database: the ingested accelerator catalogue, curated presets, lookup (F2).

Three files, three jobs. ``hardware/gpus.yaml`` is generated from Hugging Face's public SKU
table (scripts/ingest_hf_hardware.py) and gives breadth: 251 accelerators with their memory
options, compute capability and TFLOPS. ``hardware/bandwidth.yaml`` supplies the one number
that table lacks, from bus width and memory speed. ``hardware/presets.yaml`` stays a short
curated list of complete, opinionated setups, with the OS and usable fraction filled in.

``get()`` searches presets first, then the catalogue, so a name nobody curated still
resolves - just without the hand-tuned fields.
"""

from __future__ import annotations

import re
from functools import lru_cache

from rightsize._data import load_yaml
from rightsize.types import Device, Provenance

#: What a runtime can actually allocate, by vendor, when nobody has measured the device.
#: Discrete cards lose the display and driver reserve; Apple caps the GPU's share of
#: unified memory. Curated presets override this per device.
_USABLE = {"nvidia": 0.92, "amd": 0.92, "intel": 0.92, "apple": 0.7, "qualcomm": 0.8}
_BACKENDS = {
    "nvidia": ["cuda", "vulkan"],
    "amd": ["rocm", "vulkan"],
    "intel": ["sycl", "vulkan"],
    "apple": ["metal"],
    "qualcomm": ["qnn"],
}


def _to_device(rec: dict) -> Device:
    prov = rec.get("provenance")
    return Device(
        name=rec["name"],
        vendor=rec.get("vendor", "other"),
        memory_gib=rec["memory_gib"],
        system_ram_gib=rec.get("system_ram_gib"),
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


@lru_cache(maxsize=1)
def bandwidth() -> dict[str, dict]:
    """Bandwidth per device name and alias, each entry carrying how it was arrived at."""
    data = load_yaml("hardware/bandwidth.yaml")
    formula = data.get("formula_id")
    out: dict[str, dict] = {}
    for rec in data["devices"]:
        if rec.get("bandwidth_gbps") is not None:
            entry = {"gbps": float(rec["bandwidth_gbps"]), "how": "stated by the vendor"}
        else:
            # bus width x data rate / 8, the arithmetic the vendor's own spec sheet implies
            gbps = rec["bus_width_bits"] * rec["memory_speed_gbps"] / 8
            entry = {
                "gbps": round(gbps, 1),
                "how": (
                    f"{rec['bus_width_bits']}-bit {rec.get('memory_type', '')} at "
                    f"{rec['memory_speed_gbps']} Gbps"
                ).strip(),
                "formula_id": formula,
            }
        entry["provenance"] = rec.get("provenance")
        for key in (rec["name"], *rec.get("aliases", [])):
            out[_norm(key)] = entry
    return out


def _bandwidth_for(*names: str) -> tuple[float | None, dict | None]:
    table = bandwidth()
    for n in names:
        if (hit := table.get(_norm(n))) is not None:
            return hit["gbps"], hit
    return None, None


@lru_cache(maxsize=1)
def catalog() -> dict[str, Device]:
    """Every accelerator in the ingested table, one entry per memory option.

    Named "<model> <size>GB" to match how people write them and how the curated presets are
    keyed. Bandwidth is joined in where we have it; without it the fit engine still gives a
    memory verdict, just no tok/s.
    """
    data = load_yaml("hardware/gpus.yaml")
    out: dict[str, Device] = {}
    for rec in data["devices"]:
        vendor = rec.get("vendor", "other")
        gbps, hit = _bandwidth_for(rec["name"])
        prov = (hit or {}).get("provenance") or rec.get("provenance")
        for mem in rec.get("memory_gib") or []:
            name = f"{rec['name']} {mem:g}GB"
            out[name] = Device(
                name=name,
                vendor=vendor,
                memory_gib=mem,
                bandwidth_gbps=gbps,
                compute_arch=rec.get("gfx_version") or _arch(rec),
                backends=list(_BACKENDS.get(vendor, [])),
                usable_fraction=_USABLE.get(vendor, 0.9),
                provenance=Provenance(
                    source_url=prov["source_url"],
                    fetched_at=str(prov["fetched_at"]),
                    note=(hit or {}).get("how"),
                )
                if prov
                else None,
            )
    return out


def _arch(rec: dict) -> str | None:
    cc = rec.get("compute_capability")
    return f"sm_{str(cc).replace('.', '')}" if cc else None


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(nvidia|geforce|amd|radeon|apple|intel|laptop gpu)\b", " ", s)
    s = re.sub(r"(\d+)\s*gb\b", r"\1gb", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def get(name: str) -> Device:
    """Fuzzy lookup: 'RTX 4090', '4090', 'GeForce RTX 4090 24GB' resolve to one record.

    Curated presets win, because they carry the OS and usable fraction someone checked.
    Anything else falls through to the ingested catalogue.
    """
    for table in (presets(), catalog()):
        try:
            return _search(table, name)
        except KeyError as exc:
            last = exc
    raise last


def _search(table: dict[str, Device], name: str) -> Device:
    if name in table:
        return table[name]
    q = _norm(name)
    q_tokens = set(q.split())
    scored: list[tuple[int, int, str]] = []
    for key in table:
        k_tokens = set(_norm(key).split())
        if q_tokens <= k_tokens or k_tokens <= q_tokens:
            # Most query tokens matched wins; ties go to the name that adds least of its
            # own, so "RTX 5080" is the desktop card rather than "RTX 5080 Mobile".
            scored.append((len(q_tokens & k_tokens), -len(k_tokens - q_tokens), key))
    if not scored:
        # The catalogue has hundreds of names, so suggest rather than dump the lot.
        near = [k for k in table if q_tokens & set(_norm(k).split())][:6]
        raise KeyError(f"no device matches {name!r}" + (f"; did you mean {near}?" if near else ""))
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        raise KeyError(f"{name!r} is ambiguous: {[k for *_, k in scored[:3]]}")
    return table[scored[0][2]]
