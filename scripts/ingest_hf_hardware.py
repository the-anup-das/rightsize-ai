"""Ingest Hugging Face's hardware tables into data/hardware/gpus.yaml (F2).

    uv run python scripts/ingest_hf_hardware.py

huggingface.js keeps a public, MIT-licensed table of accelerators at
``packages/tasks/src/hardware*.ts``: memory options, approximate TFLOPS, CUDA compute
capability, AMD GFX version, MSRP, power and release year, for NVIDIA, AMD, Intel,
Qualcomm, Apple Silicon and CPUs. Its keys are the same three-part SKU the Hub returns for
a signed-in user's own machines (``["GPU", "NVIDIA", "RTX 3090"]``), so ingesting it gives
us both a device catalogue and the vocabulary ``hardware.from_hf`` needs.

What it does not carry is **memory bandwidth**, which is the number the speed model runs
on. That lives in data/hardware/bandwidth.yaml, computed from bus width and memory speed
rather than copied from anyone's database.

The fetch is pinned to a commit so a re-run reproduces the same file, and every record
carries the URL it came from.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

REPO = "huggingface/huggingface.js"
FILES = ("hardware.ts", "hardware-nvidia.ts", "hardware-amd.ts")
SRC_DIR = "packages/tasks/src"
BLOB = "https://github.com/{repo}/blob/{sha}/{path}"
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
OUT = Path(__file__).resolve().parents[1] / "data" / "hardware" / "gpus.yaml"

#: HF's vendor keys, mapped onto the Device.vendor literal.
VENDORS = {
    "NVIDIA": "nvidia",
    "AMD": "amd",
    "INTEL": "intel",
    "Intel": "intel",
    "QUALCOMM": "qualcomm",
    "-": "apple",
}


def resolve_sha(client: httpx.Client) -> str:
    """Current main, so every URL we record points at bytes that cannot change."""
    r = client.get(f"https://api.github.com/repos/{REPO}/commits/main")
    r.raise_for_status()
    return r.json()["sha"]


def ts_to_python(source: str) -> dict[str, Any]:
    """Parse one TypeScript object literal. These files are data, not logic: nested plain
    objects, numbers (sometimes written 8_600), string values and arrays of numbers."""
    s = re.sub(r"/\*.*?\*/", "", source, flags=re.S)  # block comments, including /** */
    s = re.sub(r"//[^\n]*", "", s)  # line comments
    s = re.sub(r"(?<=\d)_(?=\d)", "", s)  # 8_600 -> 8600
    s = re.sub(r"^(\s*)([A-Za-z_$][\w$]*)\s*:", r'\1"\2":', s, flags=re.M)  # bare keys
    # Imported tables are referenced by name (NVIDIA: NVIDIA_SKUS). Stand them up as empty
    # objects here; the caller splices in the parsed file they live in.
    s = re.sub(r":\s*(?!true|false|null)[A-Za-z_$][\w$]*\s*(?=[,}])", ": {}", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)  # trailing commas
    return json.loads(s)


def object_literal(text: str, const: str) -> str:
    """The ``{...}`` assigned to ``export const <name>``, up to its matching brace."""
    # The name may carry a type annotation: `export const X: Record<string, Y> = {`.
    m = re.search(rf"export const {const}\s*(?::[^=]*)?=\s*\{{", text)
    if not m:
        raise SystemExit(f"could not find 'export const {const}' - did the file move?")
    start = m.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise SystemExit(f"unbalanced braces after 'export const {const}'")


def records(skus: dict[str, Any], urls: dict[str, str], fetched: str) -> list[dict[str, Any]]:
    """Flatten kind -> vendor -> model into one row per device, keeping the SKU triple."""
    rows: list[dict[str, Any]] = []
    for kind, vendors in skus.items():
        for vendor, models in vendors.items():
            for model, spec in models.items():
                src = urls["hardware-nvidia.ts" if vendor == "NVIDIA" else "hardware.ts"]
                if vendor == "AMD" and kind == "GPU":
                    src = urls["hardware-amd.ts"]
                row: dict[str, Any] = {
                    "sku": [kind, vendor, model],
                    "kind": "gpu" if kind == "GPU" else ("cpu" if kind == "CPU" else "apple"),
                    "vendor": VENDORS.get(vendor, "other"),
                    "name": model,
                }
                # HF states memory the way the box does: a "24GB" card is 24 GiB.
                if spec.get("memory"):
                    row["memory_gib"] = spec["memory"]
                for key, out in (
                    ("tflops", "tflops"),
                    ("computeCapability", "compute_capability"),
                    ("gfxVersion", "gfx_version"),
                    ("msrp", "msrp_usd"),
                    ("power", "power_w"),
                    ("releaseYear", "release_year"),
                ):
                    if spec.get(key) is not None:
                        row[out] = spec[key]
                row["provenance"] = {"source_url": src, "fetched_at": fetched, "hand": False}
                rows.append(row)
    return rows


def to_yaml(rows: list[dict[str, Any]], sha: str, fetched: str) -> str:
    """Hand-rolled so the file stays diffable: one flow-style mapping per device."""
    out = [
        "# Accelerator catalogue, generated by scripts/ingest_hf_hardware.py - do not hand edit.",
        f"# Source: huggingface.js (MIT) at {sha[:12]}, fetched {fetched}.",
        "# memory_gib is the size as the vendor states it (a '24GB' card is 24 GiB).",
        "# Memory bandwidth is NOT in this table; see bandwidth.yaml.",
        "devices:",
    ]
    for r in rows:
        head = f"  - sku: [{', '.join(json.dumps(p) for p in r['sku'])}]"
        out.append(head)
        for k, v in r.items():
            if k in ("sku", "provenance"):
                continue
            out.append(f"    {k}: {json.dumps(v)}")
        p = r["provenance"]
        out.append(
            f"    provenance: {{ source_url: {p['source_url']}, "
            f'fetched_at: "{p["fetched_at"]}", hand: false }}'
        )
    return "\n".join(out) + "\n"


def main() -> int:
    fetched = dt.date.today().isoformat()
    with httpx.Client(follow_redirects=True, timeout=60) as c:
        sha = resolve_sha(c)
        print(f"huggingface.js @ {sha}")
        text, urls = {}, {}
        for f in FILES:
            path = f"{SRC_DIR}/{f}"
            r = c.get(RAW.format(repo=REPO, sha=sha, path=path))
            r.raise_for_status()
            text[f] = r.text
            urls[f] = BLOB.format(repo=REPO, sha=sha, path=path)
            print(f"  fetched {f} ({len(r.text)} bytes)")

    skus = ts_to_python(object_literal(text["hardware.ts"], "SKUS"))
    skus["GPU"]["NVIDIA"] = ts_to_python(object_literal(text["hardware-nvidia.ts"], "NVIDIA_SKUS"))
    skus["GPU"]["AMD"] = ts_to_python(object_literal(text["hardware-amd.ts"], "AMD_GPU_SKUS"))

    rows = records(skus, urls, fetched)
    OUT.write_text(to_yaml(rows, sha, fetched), encoding="utf-8")
    by_vendor: dict[str, int] = {}
    for r in rows:
        by_vendor[r["vendor"]] = by_vendor.get(r["vendor"], 0) + 1
    print(f"wrote {OUT} with {len(rows)} devices")
    for v, n in sorted(by_vendor.items(), key=lambda kv: -kv[1]):
        print(f"  {v:10s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
