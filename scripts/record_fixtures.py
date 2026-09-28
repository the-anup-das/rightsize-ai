"""Record ModelFacts fixtures from the live Hub for the offline tests (F1).

    uv run python scripts/record_fixtures.py                   # every repo below
    uv run python scripts/record_fixtures.py openai/whisper-small  # just these

One model per layout the fit engine has to get right, fetched through the catalog itself
so the fixtures also pin what the catalog keeps from each config. Re-run after changing
what facts() extracts; the cache is bypassed so nothing stale is recorded.

The GGUF repos are the same models as the config.json ones above them; the tests check that
the two sources give the same KV cache.
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
    "black-forest-labs/FLUX.1-dev": "diffusion pipeline, gated: counted from file sizes",
    "stabilityai/stable-diffusion-xl-base-1.0": "diffusion pipeline, UNet, fp16 variants",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "video pipeline, fp32 files",
    "openai/whisper-large-v2": "Whisper, the size faster-whisper measured on GPU",
    "openai/whisper-large-v3": "Whisper large-v3",
    "openai/whisper-large-v3-turbo": "Whisper turbo: four decoder layers",
    "openai/whisper-small": "Whisper, the size faster-whisper measured on CPU",
    "BAAI/bge-m3": "embedding model shipped as pytorch_model.bin only",
    "google/vit-base-patch16-224": "vision transformer",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "MLA, small enough to have GGUFs everywhere",
    # the same models read from a GGUF header
    "unsloth/Qwen3-4B-GGUF": "GGUF: plain GQA, tied embeddings, 26 quantizations",
    "ggml-org/gpt-oss-20b-GGUF": "GGUF: sliding window; an EAGLE3 draft model beside it",
    "unsloth/gemma-3-270m-it-GGUF": "GGUF: sliding window, five layers in six",
    "LiquidAI/LFM2-350M-GGUF": "GGUF: hybrid, recurrent layers marked by zero KV heads",
    "ibm-granite/granite-4.0-h-350m-GGUF": "GGUF: hybrid, Mamba2 state from the ssm keys",
    "tiiuae/Falcon-H1-1.5B-Instruct-GGUF": "GGUF: attention and Mamba2 in every layer",
    "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF": "GGUF: gated delta-net; no MTP layer",
    "mradermacher/DeepSeek-V2-Lite-Chat-GGUF": "GGUF: MLA from an older conversion",
    "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF": "GGUF: split in three parts, MoE",
}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = 0
    wanted = {r: REPOS.get(r, "requested") for r in argv} or REPOS
    for repo, why in wanted.items():
        try:
            fx = facts(repo, use_cache=False)
        except Exception as exc:  # a gated or renamed repo should not stop the rest
            print(f"FAIL {repo:36s} {type(exc).__name__}: {str(exc)[:60]}")
            failed += 1
            continue
        path = OUT / (repo.replace("/", "__") + ".json")
        path.write_text(fx.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"ok   {repo:36s} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
