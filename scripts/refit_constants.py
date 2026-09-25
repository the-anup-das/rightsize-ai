"""Propose runtime overhead constants from calibration records (F9).

    uv run python scripts/refit_constants.py                   # the local store
    uv run python scripts/refit_constants.py a.jsonl b.jsonl   # records others exported
    uv run python scripts/refit_constants.py --write           # update the data file

For each record, what the prediction left unexplained is the overhead the runtime really
added: measured - weights - KV cache. Per runtime, overhead = fixed + fraction x weights is
fitted to those by least squares (fraction kept as it is when the weights barely vary),
and the error of the old and new constants is printed side by side.

--write changes data/runtimes/overheads.yaml, and only with at least MIN_RECORDS records
for a runtime: one machine's reading is a direction, not a constant. The residual also
absorbs any error in the weights estimate, so a refit is reviewed like any data change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "runtimes" / "overheads.yaml"
MIN_RECORDS = 5


def fit(points: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    """(fixed, fraction) for residual = fixed + fraction x weights, from (weights, residual)
    pairs. With too little spread in weights to see a slope, the fraction stays put."""
    ws = [w for w, _ in points]
    if len(points) >= 3 and max(ws) - min(ws) >= 2.0:
        mw, mr = mean(ws), mean(r for _, r in points)
        sxx = sum((w - mw) ** 2 for w in ws)
        slope = sum((w - mw) * (r - mr) for w, r in points) / sxx
        fraction = min(max(slope, 0.0), 0.2)
        fixed = mr - fraction * mw
    else:
        fixed = mean(r - fraction * w for w, r in points)
    return round(min(max(fixed, 0.0), 5.0), 3), round(fraction, 4)


def _mae(rows: list[tuple[float, float]]) -> float:
    """Mean absolute relative error of (predicted, measured) pairs."""
    return mean(abs(p / m - 1) for p, m in rows) if rows else float("nan")


def refit(records: list, current: dict[str, tuple[float, float]]) -> list[dict]:
    """One proposal per runtime seen in the records."""
    by_runtime: dict[str, list] = {}
    for r in records:
        p, m = r.predicted, r.measured
        if None in (p.weights_gb, p.kv_gb, p.vram_gb, m.vram_gb):
            continue
        by_runtime.setdefault(r.runtime.name, []).append(r)
    out = []
    for name, rows in sorted(by_runtime.items()):
        old_fixed, old_frac = current.get(name, current.get("llama.cpp", (0.75, 0.02)))
        points = [(r.predicted.weights_gb, r.measured.vram_gb - r.predicted.weights_gb
                   - r.predicted.kv_gb) for r in rows]
        fixed, frac = fit(points, old_frac)

        def predict(r, f=fixed, k=frac):
            return r.predicted.weights_gb + r.predicted.kv_gb + f + k * r.predicted.weights_gb

        out.append({
            "runtime": name,
            "records": len(rows),
            "old": (old_fixed, old_frac),
            "new": (fixed, frac),
            "error_before": _mae([(r.predicted.vram_gb, r.measured.vram_gb) for r in rows]),
            "error_after": _mae([(predict(r), r.measured.vram_gb) for r in rows]),
        })
    return out


def write(proposals: list[dict], path: Path = DATA) -> list[str]:
    """Apply the proposals with enough records to the data file; returns what changed."""
    text = path.read_text(encoding="utf-8")
    changed = []
    today = dt.date.today().isoformat()
    for p in proposals:
        if p["records"] < MIN_RECORDS:
            continue
        block = re.compile(
            r"(  - name: " + re.escape(p["runtime"]) + r"\n    fixed_gb: )[\d.]+"
            r"(\n    fraction: )[\d.]+(\n(?: {4,}.*\n)*? {6}hand: )\w+(\n {6}note: ).*"
        )
        note = (f"refit from {p['records']} records on {today}: error "
                f"{p['error_before']:.1%} -> {p['error_after']:.1%}")
        text, n = block.subn(
            lambda m, p=p, note=note: f"{m.group(1)}{p['new'][0]}{m.group(2)}{p['new'][1]}"
            f"{m.group(3)}false{m.group(4)}{note}", text)
        if n:
            changed.append(p["runtime"])
    path.write_text(text, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from rightsize.fit.llm import overhead_constants
    from rightsize.telemetry.record import read

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="exported record files; default the local store")
    ap.add_argument("--write", action="store_true", help="update data/runtimes/overheads.yaml")
    args = ap.parse_args(argv)

    records = [r for f in args.files for r in read(f)] if args.files else read()
    if not records:
        print("no records: run 'rightsize telemetry on' and 'rightsize calibrate' first")
        return 1
    runtimes = {r.runtime.name for r in records}
    proposals = refit(records, {n: overhead_constants(n) for n in runtimes})
    for p in proposals:
        caution = "" if p["records"] >= MIN_RECORDS else (
            f"   (only {p['records']}; {MIN_RECORDS} needed to change the constant)")
        print(f"{p['runtime']:10s} fixed {p['old'][0]} -> {p['new'][0]} GB, fraction "
              f"{p['old'][1]} -> {p['new'][1]}; error {p['error_before']:.1%} -> "
              f"{p['error_after']:.1%} over {p['records']} records{caution}")
    if args.write:
        changed = write(proposals)
        print("updated: " + (", ".join(changed) if changed else "nothing (too few records)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
