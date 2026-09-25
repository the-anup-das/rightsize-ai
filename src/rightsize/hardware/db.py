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


#: Ingested from Wikipedia's GPU lists first, then the hand-typed file on top: those were
#: taken from vendor pages and win any disagreement.
_BANDWIDTH_FILES = ("hardware/bandwidth_wikipedia.yaml", "hardware/bandwidth.yaml")


@lru_cache(maxsize=1)
def bandwidth() -> dict[str, dict]:
    """Bandwidth per device name and alias, each entry carrying how it was arrived at."""
    out: dict[str, dict] = {}
    for path in _BANDWIDTH_FILES:
        try:
            data = load_yaml(path)
        except FileNotFoundError:  # the generated file is optional
            continue
        formula = data.get("formula_id")
        for rec in data["devices"]:
            entry = _bandwidth_entry(rec, formula)
            keys = [rec["match"]] if rec.get("match") else []
            keys += [_norm(k) for k in (rec["name"], *rec.get("aliases", []))]
            for key in keys:
                out[key] = entry
    return out


def _bandwidth_entry(rec: dict, formula: str | None) -> dict:
    if rec.get("memory_speed_gbps"):
        # bus width x data rate / 8, the arithmetic the vendor's own spec sheet implies.
        # Preferred over any stated figure: Wikipedia's bandwidth column and its own bus
        # width disagree on some rows, and the arithmetic is the one that checks out.
        gbps = rec["bus_width_bits"] * rec["memory_speed_gbps"] / 8
        return {
            "gbps": round(gbps, 1),
            "vendor": rec.get("vendor"),
            "how": (
                f"{rec['bus_width_bits']:g}-bit {rec.get('memory_type', '')} at "
                f"{rec['memory_speed_gbps']:g} Gbps"
            ).replace("  ", " ").strip(),
            "formula_id": formula,
            "stated_gbps": rec.get("stated_gbps"),
            "provenance": rec.get("provenance"),
        }
    return {
        "gbps": float(rec["bandwidth_gbps"]),
        "how": "stated by the source",
        "vendor": rec.get("vendor"),
        "provenance": rec.get("provenance"),
    }


def _bandwidth_for(
    name: str, vendor: str | None = None, mem: float | None = None
) -> tuple[float | None, dict | None]:
    """Bandwidth for one model, refusing a record that belongs to a different vendor.

    ``_norm`` drops the vendor word, so "Apple M4" and an old Mobility Radeon M4 both
    normalise to "m4". Ingested records carry the list they came from; a mismatch means the
    name collided, not that we found the device.
    """
    table = bandwidth()
    # "A100" alone is ambiguous: the 40GB PCIe card does 1555 GB/s, the 80GB SXM 2039. Ask
    # for the size we actually have before falling back to the bare model name.
    hit = table.get(_norm(f"{name} {mem:g}GB")) if mem else None
    if hit is None:
        hit = table.get(_norm(name))
    if hit is None:
        return None, None
    if vendor and hit.get("vendor") and hit["vendor"] != vendor:
        return None, None
    return hit["gbps"], hit


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
        for mem in rec.get("memory_gib") or []:
            gbps, hit = _bandwidth_for(rec["name"], vendor, mem)
            prov = (hit or {}).get("provenance") or rec.get("provenance")
            # Some names already carry their size ("Jetson Orin Nano 8GB"); do not repeat it
            suffix = f"{mem:g}GB"
            name = rec["name"] if rec["name"].endswith(suffix) else f"{rec['name']} {suffix}"
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
    s = re.sub(r"\b(nvidia|geforce|amd|radeon|instinct|apple|intel|laptop gpu|laptop)\b", " ", s)
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
