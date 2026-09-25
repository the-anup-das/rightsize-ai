"""Lightweight budgets (README "How it stays light"). These fail the build, on purpose.

Rule 1: importing rightsize must not import any ML framework.
Rule 4: importing rightsize must not import any optional extra.
Rule 5: `import rightsize` is stdlib-only and cold-imports fast; heavier types load lazily.
"""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY = {"torch", "transformers", "numpy", "diffusers", "unsloth", "mcp", "jinja2"}
# `import rightsize` is stdlib-only by design (PEP 562 lazy exports), so it must also
# not pull in the three core deps until a type is touched.
CORE_LAZY = {"pydantic", "httpx", "yaml", "rich"}

IMPORT_BUDGET_SECONDS = 0.15


def _probe(code: str) -> dict:
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def test_import_rightsize_is_stdlib_only_and_fast() -> None:
    result = _probe(
        "import json, sys, time\n"
        "t = time.perf_counter()\n"
        "import rightsize\n"
        "dt = time.perf_counter() - t\n"
        "tops = {m.split('.')[0] for m in sys.modules}\n"
        "print(json.dumps({'seconds': dt, 'tops': sorted(tops)}))\n"
    )
    tops = set(result["tops"])
    assert not (tops & HEAVY), f"heavy modules imported at startup: {sorted(tops & HEAVY)}"
    assert not (tops & CORE_LAZY), f"core deps should load lazily: {sorted(tops & CORE_LAZY)}"
    assert result["seconds"] < IMPORT_BUDGET_SECONDS, f"import took {result['seconds']:.3f}s"


def test_touching_types_never_imports_heavy_modules() -> None:
    result = _probe(
        "import json, sys\n"
        "import rightsize\n"
        "from rightsize import Plan, Device\n"
        "from rightsize.cli import build_parser\n"
        "build_parser()\n"
        "tops = {m.split('.')[0] for m in sys.modules}\n"
        "print(json.dumps({'tops': sorted(tops)}))\n"
    )
    tops = set(result["tops"])
    assert not (tops & HEAVY), f"heavy modules imported: {sorted(tops & HEAVY)}"
