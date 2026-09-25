"""Build the model pool `recommend` chooses from, from what people actually download (F4).

    uv run python scripts/ingest_candidates.py

Hand-picking candidates bakes in whatever the author happened to know; the Hub already knows
what is in use. This takes the most-downloaded text-generation models, keeps those from
established model publishers, drops quantized re-uploads and base checkpoints, and drops a
release when the same publisher has a newer one of similar size for the same task (Qwen2.5-7B
gives way to Qwen3-8B). What remains is recorded with the facts the fit engine needs, so
`recommend` works offline and without a request per model.

Gated repos (Llama, Gemma) are read through Unsloth's public mirrors of the same weights.
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
from rightsize.data_models import CandidatesFile

OUT = Path(__file__).resolve().parents[1] / "data" / "models" / "candidates.yaml"
LISTING = "https://huggingface.co/api/models"
POOL = 400
KEEP = 45

#: Publishers whose own uploads we take; a curated list, not a quality judgement on others.
PUBLISHERS = (
    "Qwen", "meta-llama", "google", "mistralai", "microsoft", "openai", "deepseek-ai",
    "ibm-granite", "HuggingFaceTB", "allenai", "nvidia", "LiquidAI", "moonshotai", "zai-org",
    "baidu", "tencent", "swiss-ai",
)
#: Re-uploads in another format, and checkpoints that are not chat models.
SKIP = re.compile(
    r"(awq|gptq|fp8|nvfp4|mxfp4|gguf|bnb|int4|int8|mlx|exl2|4bit|8bit|onnx|"
    r"(^|[-_])base([-_]|$)|guard|reward|embed|rerank|vision-encoder|eagle|-mtp|math)",
    re.I,
)
#: Suffixes publishers give the chat version of a model they also release bare.
CHAT_SUFFIXES = ("-Instruct", "-it", "-Chat", "-chat", "-instruct")
CODING = re.compile(r"(coder|devstral|codestral|starcoder|codegemma|deepcoder)", re.I)
MIRRORS = {"meta-llama": "unsloth", "google": "unsloth"}
MIN_PARAMS = 3e8
MAX_PARAMS = 1.2e12

#: Model facts worth carrying; the rest of ModelFacts.extra is not used by the fit engine.
EXTRA_KEYS = (
    "model_type", "attention", "vocab_size", "hidden_size", "intermediate_size",
    "num_attention_heads", "tie_word_embeddings", "num_experts", "num_experts_per_tok",
    "sliding_window",
)


def listing(client: httpx.Client) -> list[dict[str, Any]]:
    r = client.get(
        LISTING,
        params={
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": -1,
            "limit": POOL,
            "expand[]": ["downloads", "createdAt", "gated", "tags"],
        },
    )
    r.raise_for_status()
    return r.json()


def is_base(row: dict[str, Any], listed: set[str]) -> bool:
    """A base checkpoint, as far as the Hub lets us tell.

    Two signals, because neither is enough alone. The Hub tags chat models
    ``conversational``, which rules out Llama, Mistral and OLMo base models; but Qwen2.5's
    base models carry a chat template and the tag with it. So a bare repo is also taken for
    a base model when the same publisher lists ``<name>-Instruct`` (or -it, -Chat) beside it.
    Qwen3-8B has no such sibling, correctly, because it is itself the chat model."""
    if "conversational" not in (row.get("tags") or []):
        return True
    return any(row["id"] + suffix in listed for suffix in CHAT_SUFFIXES)


def candidate(row: dict[str, Any], listed: set[str]) -> dict[str, Any] | None:
    repo = row["id"]
    org, name = repo.split("/", 1)
    if org not in PUBLISHERS or SKIP.search(name) or is_base(row, listed):
        return None
    source = repo
    try:
        fx = facts(source, ttl_s=0)
    except httpx.HTTPStatusError:
        mirror = MIRRORS.get(org)
        if not mirror:
            return None
        source = f"{mirror}/{name}"
        try:
            fx = facts(source, ttl_s=0)
        except httpx.HTTPStatusError:
            return None
    if not fx.params_total or not MIN_PARAMS <= fx.params_total <= MAX_PARAMS:
        return None
    if not (fx.num_layers and fx.num_kv_heads and fx.head_dim):
        return None
    extra = {k: fx.extra[k] for k in EXTRA_KEYS if fx.extra.get(k) is not None}
    return {
        "repo": repo,
        "facts_from": source,
        "publisher": org,
        "tasks": ["coding"] if CODING.search(name) else ["chat"],
        "created_at": str(row.get("createdAt", ""))[:10],
        "downloads_30d": int(row.get("downloads") or 0),
        "gated": bool(row.get("gated")),
        "license": fx.license,
        "facts": {
            "params_total": fx.params_total,
            "params_active": fx.params_active,
            "num_layers": fx.num_layers,
            "num_kv_heads": fx.num_kv_heads,
            "head_dim": fx.head_dim,
            "context_max": fx.context_max,
            "dtype": fx.dtype,
            "extra": extra,
        },
    }


def superseded(c: dict[str, Any], pool: list[dict[str, Any]]) -> bool:
    """A newer release from the same publisher, for the same task, within 0.7x-1.4x the size."""
    size = c["facts"]["params_total"]
    for other in pool:
        if other is c or other["publisher"] != c["publisher"] or other["tasks"] != c["tasks"]:
            continue
        ratio = other["facts"]["params_total"] / size
        if 0.7 <= ratio <= 1.4 and other["created_at"] > c["created_at"]:
            return True
    return False


def main() -> int:
    fetched = dt.date.today().isoformat()
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        rows = listing(c)
    listed = {r["id"] for r in rows}
    pool = []
    for row in rows:
        got = candidate(row, listed)
        if got:
            pool.append(got)
            print(f"  {got['repo']:52s} {got['facts']['params_total'] / 1e9:7.2f}B {got['created_at']}")
    kept = [c for c in pool if not superseded(c, pool)]
    dropped = sorted({c["repo"] for c in pool} - {c["repo"] for c in kept})
    kept = sorted(kept, key=lambda c: -c["downloads_30d"])[:KEEP]
    doc = {
        "provenance": {
            "source_url": "https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads",
            "fetched_at": fetched,
            "note": f"top {POOL} by downloads, filtered as described in scripts/ingest_candidates.py",
        },
        "superseded": dropped,
        "models": kept,
    }
    CandidatesFile.model_validate(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# The model pool `recommend` ranks, generated by scripts/ingest_candidates.py.\n"
        "# Do not hand edit; re-run the script to refresh it.\n"
    )
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"kept {len(kept)} of {len(pool)} candidates; {len(dropped)} superseded by newer releases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
