"""Check the fit engine's KV cache against what llama.cpp actually allocates.

    uv run python scripts/validate_kv.py

``llama-fit-params`` ships with llama.cpp and dry-runs a model's allocation in well under a
second, printing a breakdown of self = model + context + compute per device. The context
column is the KV cache plus any recurrent state: the exact number fit/kv.py predicts. So for
each architecture we have a small GGUF of, this prints ours against llama.cpp's own.

It is how the sliding-window, MLA and hybrid rules in data/runtimes/llama.cpp/kv_cache.yaml
were checked, rather than only read from the source.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from rightsize.catalog import facts
from rightsize.execution.llamacpp import find_tools
from rightsize.fit import kv
from rightsize.types import GB

MIB = 1024**2

#: (gguf under models/validation, repo whose config describes it, what it exercises)
CASES = [
    ("gemma-3-270m-it-UD-IQ2_XXS.gguf", "unsloth/gemma-3-270m-it", "sliding window 5:1"),
    ("LFM2-350M-Q4_0.gguf", "LiquidAI/LFM2-350M", "hybrid: short conv + attention"),
    ("granite-4.0-h-350m-Q2_K.gguf", "ibm-granite/granite-4.0-h-350m", "hybrid: Mamba2 + attention"),
]
CTX = (4096, 32768)

#: One line per cache llama.cpp creates: the full and sliding KV caches of an iSWA model,
#: the KV and recurrent halves of a hybrid. The breakdown table rounds to whole MiB, which
#: hides a 15 MiB sliding cache's error, so these are what we add up.
_CACHE_SIZE = re.compile(r"(?:llama_kv_cache|llama_memory_recurrent)\s*:\s*size\s*=\s*([\d.]+)\s*MiB")


def llama_context_mib(fit_params: Path, gguf: Path, ctx: int) -> float:
    """KV cache plus recurrent state as llama.cpp sizes them, from a dry-run allocation."""
    proc = subprocess.run(
        [str(fit_params), "-m", str(gguf), "-c", str(ctx), "-v"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    sizes = [float(m.group(1)) for m in _CACHE_SIZE.finditer(proc.stderr + proc.stdout)]
    if not sizes:
        raise RuntimeError(f"no cache sizes from llama-fit-params for {gguf.name}")
    return sum(sizes)


def main() -> int:
    tools = find_tools()
    fit_params = tools.root / ("llama-fit-params.exe" if sys.platform == "win32" else "llama-fit-params")
    root = Path("models/validation")
    worst = 0.0
    print(f"{'model':36s} {'layout':34s} {'ctx':>6s} {'ours MiB':>9s} {'llama.cpp':>10s} {'diff':>6s}")
    for gguf_name, repo, what in CASES:
        gguf = root / gguf_name
        if not gguf.exists():
            print(f"{gguf_name:36s} missing, skipped")
            continue
        fx = facts(repo)
        for ctx in CTX:
            ours = kv.kv_cache_gb(fx, ctx) * GB / MIB
            theirs = llama_context_mib(fit_params, gguf, ctx)
            diff = (ours - theirs) / theirs * 100 if theirs else 0.0
            worst = max(worst, abs(diff))
            print(f"{repo.split('/')[-1]:36s} {what:34s} {ctx:>6d} {ours:>9.1f} {theirs:>10.1f} {diff:>+5.1f}%")
    print(f"worst disagreement {worst:.1f}%")
    return 0 if worst < 5 else 1


if __name__ == "__main__":
    sys.exit(main())
