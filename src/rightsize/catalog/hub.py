"""Model facts from the Hugging Face Hub without downloading weights (F1, first slice).

Reads ``config.json``, the safetensors index and each shard's header (8-byte length prefix +
JSON) over HTTP Range requests. httpx only; ``HF_TOKEN`` is honoured for gated repos.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import httpx

from rightsize.types import Family, ModelFacts, ModelRef

HUB = "https://huggingface.co"
_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "I4": 0.5,
    "U4": 0.5,
}
_TEXT_MODEL_KEYS = ("text_config", "language_config", "llm_config")


def _headers(token: str | None) -> dict[str, str]:
    tok = token or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _cache_path(repo: str, revision: str) -> Path:
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    return base / "models" / repo.replace("/", "__") / f"{revision}.json"


def _read_cache(path: Path, ttl_s: float) -> dict[str, Any] | None:
    try:
        if time.time() - path.stat().st_mtime > ttl_s:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def safetensors_header(client: httpx.Client, url: str) -> dict[str, Any]:
    """Fetch only the JSON header of a safetensors file (two small Range requests)."""
    r = client.get(url, headers={"Range": "bytes=0-7"})
    r.raise_for_status()
    (n,) = struct.unpack("<Q", r.content[:8])
    r = client.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    r.raise_for_status()
    return json.loads(r.content[:n])


def _count_params(header: dict[str, Any]) -> tuple[int, dict[str, int]]:
    total = 0
    by_dtype: dict[str, int] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in info["shape"]:
            n *= d
        total += n
        by_dtype[info["dtype"]] = by_dtype.get(info["dtype"], 0) + n
    return total, by_dtype


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in _TEXT_MODEL_KEYS:
        if isinstance(cfg.get(key), dict):
            return cfg[key]
    return cfg


def _family(cfg: dict[str, Any], pipeline_tag: str | None) -> Family:
    mt = str(cfg.get("model_type", "")).lower()
    tag = (pipeline_tag or "").lower()
    if "diffusion" in mt or tag in {"text-to-image", "image-to-image", "text-to-video"}:
        return Family.diffusion
    if "whisper" in mt or tag in {"automatic-speech-recognition", "text-to-speech"}:
        return Family.audio
    if tag in {"feature-extraction", "sentence-similarity"}:
        return Family.embedding
    if tag in {"object-detection", "image-classification", "image-segmentation"}:
        return Family.vision
    return Family.llm


def facts(
    repo: str,
    revision: str = "main",
    *,
    token: str | None = None,
    offline: bool = False,
    ttl_s: float = 7 * 24 * 3600,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> ModelFacts:
    """Return ModelFacts for a Hub repo. Weights are never downloaded."""
    cache = _cache_path(repo, revision)
    cached = None if transport else _read_cache(cache, ttl_s if not offline else float("inf"))
    if cached is not None:
        return ModelFacts.model_validate(cached)
    if offline:
        raise FileNotFoundError(f"no cached facts for {repo}@{revision} and offline=True")

    with httpx.Client(
        headers=_headers(token), timeout=timeout, follow_redirects=True, transport=transport
    ) as c:
        info = c.get(f"{HUB}/api/models/{repo}", params={"revision": revision})
        info.raise_for_status()
        meta = info.json()
        siblings = [s["rfilename"] for s in meta.get("siblings", [])]
        base = f"{HUB}/{repo}/resolve/{revision}"

        cfg: dict[str, Any] = {}
        if "config.json" in siblings:
            r = c.get(f"{base}/config.json")
            r.raise_for_status()
            cfg = r.json()

        shards = sorted(s for s in siblings if s.endswith(".safetensors") and "/" not in s)
        total = 0
        by_dtype: dict[str, int] = {}
        for shard in shards:
            hdr = safetensors_header(c, f"{base}/{shard}")
            n, d = _count_params(hdr)
            total += n
            for k, v in d.items():
                by_dtype[k] = by_dtype.get(k, 0) + v

    tc = _text_config(cfg)
    hidden = tc.get("hidden_size")
    heads = tc.get("num_attention_heads")
    head_dim = tc.get("head_dim") or (hidden // heads if hidden and heads else None)
    kv_heads = tc.get("num_key_value_heads") or heads
    dominant = max(by_dtype, key=by_dtype.get) if by_dtype else cfg.get("torch_dtype")
    experts = tc.get("num_experts") or tc.get("num_local_experts")
    active_experts = tc.get("num_experts_per_tok")

    facts_ = ModelFacts(
        ref=ModelRef(repo=repo, revision=revision),
        family=_family(cfg, meta.get("pipeline_tag")),
        params_total=total or None,
        params_active=None,
        dtype=str(dominant).upper() if dominant else None,
        num_layers=tc.get("num_hidden_layers"),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        context_max=tc.get("max_position_embeddings"),
        license=(meta.get("cardData") or {}).get("license"),
        base_model=_base_model(meta),
        confidence=1.0 if total else 0.5,
        extra={
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "params_by_dtype": by_dtype,
            "shards": shards,
            "gated": meta.get("gated", False),
            "num_experts": experts,
            "num_experts_per_tok": active_experts,
            "sliding_window": tc.get("sliding_window"),
            "tie_word_embeddings": cfg.get("tie_word_embeddings"),
            "vocab_size": tc.get("vocab_size"),
            "hidden_size": hidden,
            "num_attention_heads": heads,
        },
    )
    if experts and active_experts and total:
        # Rough MoE active-parameter estimate: attention + shared parts count fully,
        # expert FFNs scale by active/total. Marked in extra so callers know it is approximate.
        facts_.params_active = int(total * (active_experts / experts))
        facts_.extra["params_active_note"] = "approximate: total x active/experts"
    if transport is None:
        _write_cache(cache, facts_.model_dump(mode="json"))
    return facts_


def _base_model(meta: dict[str, Any]) -> str | None:
    card = meta.get("cardData") or {}
    bm = card.get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    return bm
