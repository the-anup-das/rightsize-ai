"""Record ModelFacts fixtures from the live Hub for the offline tests (F1).

    uv run python scripts/record_fixtures.py

One model per layout the fit engine has to get right, fetched through the catalog itself
so the fixtures also pin what the catalog keeps from each config. Re-run after changing
what facts() extracts; the cache is bypassed so nothing stale is recorded.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rightsize.catalog import facts

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "facts"

REPOS = {
    "Qwen/Qwen3-4B": "plain GQA",
    "unsloth/gemma-3-27b-it": "sliding window, five layers in six",
    "unsloth/gemma-3-270m-it": "sliding window, the size validated against llama.cpp",
    "unsloth/gemma-2-9b-it": "sliding window, alternating",
    "openai/gpt-oss-20b": "sliding window from a per-layer list",
    "deepseek-ai/DeepSeek-V3": "MLA",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "hybrid: gated delta-net, one layer in four attends",
    "LiquidAI/LFM2-350M": "hybrid: short convolutions, attention by index",
    "ibm-granite/granite-4.0-h-350m": "hybrid: Mamba2, validated against llama.cpp",
    "ibm-granite/granite-4.0-h-small": "hybrid: Mamba2 MoE",
    "tiiuae/Falcon-H1-1.5B-Instruct": "hybrid: attention and Mamba2 in every layer",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    for repo, why in REPOS.items():
        try:
            fx = facts(repo, ttl_s=0)
        except Exception as exc:  # a gated or renamed repo should not stop the rest
            print(f"FAIL {repo:36s} {type(exc).__name__}: {str(exc)[:60]}")
            failed += 1
            continue
        path = OUT / (repo.replace("/", "__") + ".json")
        path.write_text(fx.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"ok   {repo:36s} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
