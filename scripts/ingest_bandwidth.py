"""Ingest memory bandwidth for the accelerator catalogue from Wikipedia's GPU lists (F2).

    uv run python scripts/ingest_bandwidth.py

Hugging Face's SKU table (scripts/ingest_hf_hardware.py) has no memory bandwidth, and it is
the number the speed model runs on. TechPowerUp serves a captcha and marks its pages
noindex/nofollow, so it is off limits; Wikidata has no bandwidth property at all. What is
left is Wikipedia's "List of <vendor> graphics processing units", which tabulates bus width,
bus type, memory clock and bandwidth per SKU, is CC BY-SA, and cites vendor specs per row.

We take the *primary* facts from it - bus width and, where the page gives an effective data
rate, memory speed - and compute bandwidth ourselves:

    bandwidth_gbps = bus_width_bits * memory_speed_gbps / 8

Where a page states only bandwidth (AMD's tables give a memory clock in MHz whose multiplier
varies by generation), the stated figure is carried with ``stated: true``. Every record keeps
the page and revision id it came from, so a number can always be traced.

The run refuses to write if it disagrees with a hand-curated value in bandwidth.yaml by more
than 2%: those were typed from vendor pages, so a mismatch means the parse is wrong.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hardware" / "bandwidth_wikipedia.yaml"
CURATED = ROOT / "data" / "hardware" / "bandwidth.yaml"
API = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/w/index.php?title={title}&oldid={revid}"
UA = "rightsize-hardware-ingest/0.0.1 (+https://github.com/the-anup-das/rightsize-ai)"
PAGES = (
    ("List of Nvidia graphics processing units", "nvidia"),
    ("List of AMD graphics processing units", "amd"),
)
TOLERANCE = 0.02

# Header cells we need, matched case-insensitively against the sub-header row.
WANT = {
    "model": ("model", "graphics", "chip", "model (code name)"),
    "size": ("size (gb)", "size (mb)", "size (mib)", "size"),
    "bus_type": ("bus type", "memory type"),
    "bus_width": ("bus width (bit)", "bus width (bits)"),
    # AMD's RDNA tables put both in one cell, e.g. "GDDR6 384-bit"
    "bus_type_width": ("bus type & width", "bus width & type"),
    "bandwidth": ("bandwidth (gb/s)", "memory bandwidth (gb/s)"),
    "mem_clock": ("memory (mhz) (gt/s)", "memory clock (mhz)"),
    # ... and state the data rate in MT/s rather than parenthesised Gbps
    "mem_rate_mts": ("clock (mt/s)", "memory (mt/s)"),
}


class TableParser(HTMLParser):
    """Collect every <table> as a grid of strings, expanding rowspan and colspan.

    Wikipedia's spec tables lean on both heavily - a model's memory row often spans six
    variants - so a naive cell walk silently shifts columns and mixes up numbers.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._stack: list[dict[str, Any]] = []
        self._cell: list[str] | None = None
        self._skip = 0

    def _cur(self) -> dict[str, Any] | None:
        return self._stack[-1] if self._stack else None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self._stack.append({"grid": [], "pending": {}, "row": None, "col": 0, "span": (1, 1)})
            return
        t = self._cur()
        if t is None:
            return
        if tag == "tr":
            t["row"], t["col"] = [], 0
        elif tag in ("td", "th"):
            self._cell = []
            t["span"] = (int(a.get("colspan") or 1), int(a.get("rowspan") or 1))
        elif tag in ("sup", "style", "script"):  # footnote markers are not data
            self._skip += 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = self._cur()
        if tag == "table" and t is not None:
            self.tables.append(t["grid"])
            self._stack.pop()
            return
        if t is None:
            return
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            cols, rows = t["span"]
            r, c = len(t["grid"]), t["col"]
            while (r, c) in t["pending"]:  # a rowspan from an earlier row owns this slot
                c += 1
            for dc in range(cols):
                for dr in range(rows):
                    if dr:
                        t["pending"][(r + dr, c + dc)] = text
                    else:
                        while len(t["row"]) <= c + dc:
                            t["row"].append("")
                        t["row"][c + dc] = text
            t["col"] = c + cols
            self._cell = None
        elif tag == "tr" and t["row"] is not None:
            r, row = len(t["grid"]), t["row"]
            for (rr, cc), val in list(t["pending"].items()):
                if rr == r:
                    while len(row) <= cc:
                        row.append("")
                    row[cc] = row[cc] or val
                    t["pending"].pop((rr, cc))
            t["grid"].append(row)
            t["row"] = None
        elif tag in ("sup", "style", "script"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skip:
            self._cell.append(data)


def fetch(title: str, client: httpx.Client) -> tuple[str, int]:
    r = client.get(
        API,
        params={
            "action": "parse",
            "page": title,
            "prop": "text|revid",
            "format": "json",
            "formatversion": "2",
        },
    )
    r.raise_for_status()
    d = r.json()["parse"]
    return d["text"], d["revid"]


def _num(cell: str) -> float | None:
    """First plain number in a cell. '672.3' -> 672.3, '~91.4 (~102.0)' -> 91.4."""
    m = re.search(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
    return float(m.group()) if m else None


#: Plausible effective data rates in Gbps, from SDR-era DDR to GDDR7. Older tables put a
#: clock in MHz where newer ones put the rate in the parentheses, and multiplying a bus width
#: by 666 "Gbps" claims 10 TB/s for a GeForce GT 415. Outside this band we do not compute.
RATE_RANGE = (0.4, 40.0)


def _rate_gt_s(cell: str) -> float | None:
    """Effective data rate from 'clock (rate)' cells: '1313 (21.0)' -> 21.0 Gbps."""
    m = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*\)", cell)
    if not m:
        return None
    rate = float(m.group(1))
    return rate if RATE_RANGE[0] <= rate <= RATE_RANGE[1] else None


def columns(header: list[str], group: list[str] | None = None) -> dict[str, int]:
    """Map our field names onto this table's column indices.

    ``group`` is the header row above, which carries the spanning labels. AMD's RDNA tables
    have two "Bandwidth (GB/s)" columns - one under "Infinity Cache", one under "Memory" -
    and taking the first would quietly report cache bandwidth as memory bandwidth.
    """
    cands: dict[str, list[int]] = {}
    for i, cell in enumerate(header):
        low = cell.strip().lower()
        for field, options in WANT.items():
            if low in options:
                cands.setdefault(field, []).append(i)
    found: dict[str, int] = {}
    for field, idxs in cands.items():
        if len(idxs) == 1:
            found[field] = idxs[0]
            continue
        under_memory = [
            i for i in idxs if "memory" in (group[i].lower() if group and i < len(group) else "")
        ]
        found[field] = under_memory[0] if under_memory else idxs[-1]
    return found


def rows_from(grid: list[list[str]]) -> list[dict[str, Any]]:
    """Data rows of one spec table, if it is one: needs a model and a bandwidth column."""
    for h in range(min(4, len(grid))):
        cols = columns(grid[h], grid[h - 1] if h else None)
        if "model" in cols and "bandwidth" in cols and ("bus_width" in cols or "bus_type_width" in cols):
            break
    else:
        return []
    out = []
    for row in grid[h + 1 :]:
        if len(row) <= max(cols.values()):
            continue
        name = re.sub(r"\[[^\]]*\]", "", row[cols["model"]]).strip()
        name = re.sub(r"\s*\(.*", "", name).strip()  # "RX 7900 XTX (Navi 31)" -> "RX 7900 XTX"
        if not name or _num(name) == float(len(name)):  # header repeats, section labels
            continue
        rec: dict[str, Any] = {"name": name}
        rec["bandwidth"] = _num(row[cols["bandwidth"]])
        if "bus_width" in cols:
            rec["bus_width"] = _num(row[cols["bus_width"]])
            if "bus_type" in cols:
                rec["bus_type"] = row[cols["bus_type"]].strip() or None
        else:
            rec["bus_type"], rec["bus_width"] = _type_and_width(row[cols["bus_type_width"]])
        if "size" in cols:
            rec["size"] = _num(row[cols["size"]])
        if "mem_clock" in cols:
            rec["rate"] = _rate_gt_s(row[cols["mem_clock"]])
        elif "mem_rate_mts" in cols:
            rec["rate"] = _rate_from_mts(row[cols["mem_rate_mts"]])
        if rec["bandwidth"] and rec["bus_width"]:
            _fix_clock_vs_rate(rec)
            out.append(rec)
    return out


def _fix_clock_vs_rate(rec: dict[str, Any]) -> None:
    """Some tables give a memory *clock* where others give the effective data rate.

    HBM datacenter rows are the usual case: an A100's 5120-bit bus at the stated 1.215
    works out to 778 GB/s, half the 1555 the same row reports. When doubling the figure
    reconciles it with the page's own bandwidth column, the column was a clock (memory is
    double data rate) and we say so, rather than publishing half the real bandwidth.
    """
    rate, width, stated = rec.get("rate"), rec.get("bus_width"), rec.get("bandwidth")
    if not (rate and width and stated):
        return
    computed = width * rate / 8
    off = abs(computed - stated) / stated
    if off <= TOLERANCE:
        return
    # Vendors round the headline figure (NVIDIA calls 8192 GB/s "8 TB/s"), so allow a wider
    # margin when testing whether a data-rate multiple reconciles the two.
    for mult in (2, 4, 8):
        scaled = rate * mult
        if not RATE_RANGE[0] <= scaled <= RATE_RANGE[1]:
            continue
        if abs(width * scaled / 8 - stated) / stated < min(off, 0.05):
            rec["rate"] = scaled
            rec["rate_was_clock"] = True
            return


def _type_and_width(cell: str) -> tuple[str | None, float | None]:
    """Split AMD's combined cell: 'GDDR6 384-bit' -> ('GDDR6', 384)."""
    width = re.search(r"(\d+)\s*-?\s*bit", cell, re.I)
    kind = re.match(r"\s*([A-Za-z0-9]+)", cell)
    return (kind.group(1) if kind else None), (float(width.group(1)) if width else None)


def _rate_from_mts(cell: str) -> float | None:
    """A rate given in MT/s: '20000' -> 20 Gbps."""
    n = _num(cell)
    if n is None:
        return None
    rate = n / 1000
    return rate if RATE_RANGE[0] <= rate <= RATE_RANGE[1] else None


#: The loader's own normaliser, imported rather than copied: when the two drifted, an
#: RTX 3060 8GB row silently answered for the 12GB card, 224 GB/s instead of 360.
from rightsize.hardware.db import _norm  # noqa: E402


def collapse(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One record per model, or per model+size where variants differ.

    A model appears many times (die revisions, regional SKUs) with identical memory. But
    some names cover real differences - an RTX 3060 is 192-bit at 12 GB and 128-bit at 8 GB -
    so when the rows disagree we key on the size too rather than pick a winner.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_name.setdefault(_norm(r["name"]), []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, group in by_name.items():
        widths = {r["bus_width"] for r in group}
        rates = {r.get("rate") for r in group}
        if len(widths) == 1 and len(rates) == 1:
            out[key] = dict(group[0])
            continue
        for r in group:
            if r.get("size"):
                out[f"{key} {r['size']:g}gb"] = dict(r)
    return out


def value_of(rec: dict[str, Any]) -> float:
    """What we would store: computed from the primary facts, else the stated figure.

    The page's own bandwidth column is not always consistent with its bus width and rate -
    the RTX 3060 12 GB row says 336 GB/s where 192-bit at 15 Gbps is 360, which is what
    NVIDIA publishes. So the arithmetic wins and the stated column is kept for comparison.
    """
    if rec.get("rate"):
        return rec["bus_width"] * rec["rate"] / 8
    return float(rec["bandwidth"])


def check(records: dict[str, dict[str, Any]]) -> list[str]:
    """Compare against the hand-typed values; a big gap means the parse drifted."""
    curated = yaml.safe_load(CURATED.read_text(encoding="utf-8"))
    problems = []
    for rec in curated["devices"]:
        if rec.get("bandwidth_gbps") is not None:
            expected = float(rec["bandwidth_gbps"])
        else:
            expected = rec["bus_width_bits"] * rec["memory_speed_gbps"] / 8
        for name in (rec["name"], *rec.get("aliases", [])):
            got = records.get(_norm(name))
            if not got:
                continue
            mine = value_of(got)
            delta = abs(mine - expected) / expected
            status = "ok " if delta <= TOLERANCE else "BAD"
            print(
                f"  {status} {rec['name']:20s} curated {expected:>7.1f}  "
                f"ingested {mine:>7.1f}  {delta * 100:4.1f}%"
            )
            if delta > TOLERANCE:
                problems.append(f"{rec['name']}: curated {expected}, ingested {mine}")
            break
    return problems


def disagreements(records: dict[str, dict[str, Any]]) -> list[str]:
    """Rows where the page's bandwidth column and its own bus width x rate do not agree."""
    out = []
    for key, r in sorted(records.items()):
        if not r.get("rate"):
            continue
        computed = r["bus_width"] * r["rate"] / 8
        if abs(computed - r["bandwidth"]) / r["bandwidth"] > TOLERANCE:
            out.append(f"{key}: column {r['bandwidth']:g}, {r['bus_width']:g}-bit x {r['rate']:g} = {computed:.1f}")
    return out


def to_yaml(records: dict[str, dict[str, Any]], sources: dict[str, str], fetched: str) -> str:
    lines = [
        "# Memory bandwidth, generated by scripts/ingest_bandwidth.py - do not hand edit.",
        "# Extracted from Wikipedia's GPU list tables (CC BY-SA 4.0), which tabulate bus width,",
        "# bus type, memory clock and bandwidth per SKU and cite vendor specs per row.",
        "#",
        "# Where the page gives an effective data rate, bandwidth is computed from the two",
        "# primary facts (bus_width_bits * memory_speed_gbps / 8) and the page's own bandwidth",
        "# column is kept as stated_gbps so the two can be compared. Where it does not, the",
        "# stated figure is all we have and the record says so.",
        "#",
        "# Hand-curated values in bandwidth.yaml take precedence over anything here.",
        f"# Fetched {fetched}.",
        "formula_id: mem.bandwidth.bus_x_rate.v0",
        "sources:",
    ]
    for vendor, url in sources.items():
        lines.append(f"  {vendor}: {url}")
    lines.append("devices:")
    for key in sorted(records):
        r = records[key]
        # json.dumps gives a double-quoted scalar, which is valid YAML and needs no guessing
        # about when a name has to be quoted.
        lines.append(f"  - name: {json.dumps(r['name'])}")
        lines.append(f"    match: {json.dumps(key)}")
        if r.get("bus_type"):
            lines.append(f"    memory_type: {json.dumps(r['bus_type'])}")
        lines.append(f"    vendor: {json.dumps(r['vendor'])}")
        lines.append(f"    bus_width_bits: {r['bus_width']:g}")
        if r.get("rate"):
            lines.append(f"    memory_speed_gbps: {r['rate']:g}")
            lines.append(f"    stated_gbps: {r['bandwidth']:g}")
        else:
            lines.append(f"    bandwidth_gbps: {r['bandwidth']:g}")
            lines.append("    stated: true")
        # The revision URLs carry ? and &, which are special in YAML flow context.
        lines.append(
            f"    provenance: {{ source_url: {json.dumps(r['source'])}, "
            f"fetched_at: {json.dumps(fetched)} }}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    fetched = dt.date.today().isoformat()
    all_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    with httpx.Client(headers={"User-Agent": UA}, timeout=120, follow_redirects=True) as c:
        for title, vendor in PAGES:
            html, revid = fetch(title, c)
            url = PAGE_URL.format(title=title.replace(" ", "_"), revid=revid)
            sources[vendor] = url
            p = TableParser()
            p.feed(html)
            rows = [r for grid in p.tables for r in rows_from(grid)]
            for r in rows:
                r["source"] = url
                r["vendor"] = vendor
            print(f"{title}: rev {revid}, {len(p.tables)} tables, {len(rows)} rows with bandwidth")
            all_rows += rows

    records = collapse(all_rows)
    print(f"collapsed to {len(records)} devices")
    print("cross-check against hand-typed values:")
    problems = check(records)
    odd = disagreements(records)
    if odd:
        print(f"{len(odd)} rows whose bandwidth column disagrees with their own bus width x rate;")
        print("  the arithmetic is used and the column kept as stated_gbps. First few:")
        for line in odd[:5]:
            print("    " + line)
    if problems:
        print("\nrefusing to write, the parse disagrees with values typed from vendor pages:")
        for p_ in problems:
            print("  " + p_)
        return 1
    OUT.write_text(to_yaml(records, sources, fetched), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
