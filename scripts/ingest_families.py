"""Build the curated model lists for diffusion, audio, vision and embedding models (F1).

    uv run python scripts/ingest_families.py

LLMs have their own pool, with the facts recommend ranks by (data/models/candidates.yaml,
scripts/ingest_candidates.py). This is its counterpart for the other four families: for each
Hub task, the most downloaded models from the publishers that make them, without re-uploads
in other formats, adapters or community fine-tunes, and with each model's parameter count so
search() can answer "which image models are under 13B" offline. Names and sizes only; facts
are fetched when a model is estimated.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from rightsize.catalog import facts
from rightsize.data_models import FamiliesFile

OUT = Path(__file__).resolve().parents[1] / "data" / "models" / "families.yaml"
LISTING = "https://huggingface.co/api/models"
POOL = 300
KEEP = 8  # per task

#: Hub tasks per family, the vocabulary search() takes.
TASKS = {
    "diffusion": ["text-to-image", "image-to-image", "text-to-video", "image-to-video"],
    "audio": ["automatic-speech-recognition", "text-to-speech"],
    "vision": [
        "image-classification",
        "object-detection",
        "image-segmentation",
        "depth-estimation",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
        "mask-generation",
        "image-feature-extraction",
    ],
    "embedding": ["sentence-similarity", "feature-extraction", "text-ranking"],
}

#: Publishers whose own models we take; a curated list, not a quality judgement on others.
PUBLISHERS = {
    "diffusion": (
        "black-forest-labs",
        "stabilityai",
        "stable-diffusion-v1-5",
        "CompVis",
        "Tongyi-MAI",
        "Qwen",
        "Wan-AI",
        "Lightricks",
        "tencent",
        "hunyuanvideo-community",
        "HiDream-ai",
        "zai-org",
        "THUDM",
        "Efficient-Large-Model",
        "nvidia",
        "PixArt-alpha",
        "playgroundai",
        "Alpha-VLLM",
        "lodestones",
        "stepfun-ai",
        "genmo",
        "Kwai-Kolors",
        "kandinsky-community",
        "OmniGen2",
        "briaai",
    ),
    "audio": (
        "openai",
        "nvidia",
        "distil-whisper",
        "facebook",
        "microsoft",
        "hexgrad",
        "coqui",
        "ResembleAI",
        "SWivid",
        "Qwen",
        "mistralai",
        "kyutai",
        "sesame",
        "canopylabs",
        "fishaudio",
        "myshell-ai",
        "suno",
        "k2-fsa",
        "openbmb",
        "FunAudioLLM",
        "ibm-granite",
        "UsefulSensors",
        "google",
    ),
    "vision": (
        "google",
        "facebook",
        "microsoft",
        "nvidia",
        "apple",
        "depth-anything",
        "PekingU",
        "IDEA-Research",
        "openai",
        "laion",
        "timm",
        "hustvl",
        "Intel",
        "ZhengPeng7",
        "briaai",
        "CIDAS",
        "shi-labs",
        "LiheYoung",
        "allenai",
    ),
    "embedding": (
        "sentence-transformers",
        "BAAI",
        "intfloat",
        "nomic-ai",
        "Alibaba-NLP",
        "mixedbread-ai",
        "jinaai",
        "Qwen",
        "Snowflake",
        "google",
        "thenlper",
        "nvidia",
        "ibm-granite",
        "cross-encoder",
    ),
}

#: Re-uploads in another format, adapters and conversions: the original is listed instead.
SKIP = re.compile(
    r"(gguf|onnx|openvino|mlx|awq|gptq|fp8|int8|int4|nf4|bnb|4bit|8bit|ct2|faster-whisper|"
    r"ggml|coreml|quantized|exl2|tensorrt|nunchaku|svdq|lora|comfy)",
    re.I,
)


def listing(client: httpx.Client, task: str) -> list[dict[str, Any]]:
    r = client.get(
        LISTING,
        params={
            "pipeline_tag": task,
            "sort": "downloads",
            "direction": -1,
            "limit": POOL,
            "expand[]": ["downloads", "createdAt", "gated", "library_name"],
        },
    )
    r.raise_for_status()
    return r.json()


def _is_pipeline(repo: str) -> bool:
    r = httpx.get(f"https://huggingface.co/api/models/{repo}", timeout=60, follow_redirects=True)
    if r.status_code >= 400:
        return False
    return any(s["rfilename"] == "model_index.json" for s in r.json().get("siblings") or [])


def entry(row: dict[str, Any], family: str, task: str) -> dict[str, Any] | None:
    repo = row["id"]
    org, _, name = repo.partition("/")
    if org not in PUBLISHERS[family] or SKIP.search(name):
        return None
    if family == "diffusion" and not _is_pipeline(repo):
        # Single-file checkpoint repos, lone VAEs, original-format video repos: the diffusion
        # estimator sizes diffusers pipelines. Checked from the file list, before any header.
        print(f"  skip {repo}: not a diffusers pipeline")
        return None
    try:
        fx = facts(repo, use_cache=False)
    except (httpx.HTTPError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  skip {repo}: {type(exc).__name__}")
        return None
    if not fx.params_total:
        return None
    if family == "diffusion" and not (fx.extra.get("pipeline") or {}).get("components"):
        print(f"  skip {repo}: a model_index.json that names no weights")
        return None
    if "rerank" in name.lower():
        task = "text-ranking"  # some rerankers are tagged feature-extraction on the Hub
    return {
        "repo": repo,
        "family": family,
        "tasks": [task],
        "publisher": org,
        "params_total": fx.params_total,
        "library": row.get("library_name"),
        "license": fx.license,
        "gated": bool(row.get("gated")),
        "created_at": str(row.get("createdAt", ""))[:10],
        "downloads_30d": int(row.get("downloads") or 0),
    }


def main() -> int:
    fetched = dt.date.today().isoformat()
    kept: dict[str, dict[str, Any]] = {}
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        for family, tasks in TASKS.items():
            for task in tasks:
                n = 0
                for row in listing(c, task):
                    if n >= KEEP:
                        break
                    if row["id"] in kept:  # one model, several tasks
                        if task not in kept[row["id"]]["tasks"]:
                            kept[row["id"]]["tasks"].append(task)
                        n += 1
                        continue
                    got = entry(row, family, task)
                    if got:
                        kept[row["id"]] = got
                        n += 1
                        print(
                            f"  {family:9s} {task:30s} {got['repo']:50s} "
                            f"{got['params_total'] / 1e9:7.3f}B"
                        )
    models = sorted(kept.values(), key=lambda m: (m["family"], -m["downloads_30d"]))
    doc = {
        "provenance": {
            "source_url": "https://huggingface.co/models?sort=downloads",
            "fetched_at": fetched,
            "note": f"top {POOL} per Hub task by downloads, filtered as described in "
            "scripts/ingest_families.py",
        },
        "tasks": TASKS,
        "models": models,
    }
    FamiliesFile.model_validate(doc)
    header = (
        "# Curated diffusion, audio, vision and embedding models, generated by\n"
        "# scripts/ingest_families.py. Do not hand edit; re-run the script to refresh it.\n"
        "# LLMs are in candidates.yaml.\n"
    )
    OUT.write_text(
        header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"kept {len(models)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
