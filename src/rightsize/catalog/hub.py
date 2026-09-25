"""Model facts from the Hugging Face Hub without downloading weights (F1).

Reads ``config.json``, the safetensors index and each shard's header (8-byte length prefix +
JSON) over HTTP Range requests. httpx only; ``HF_TOKEN`` is honoured for gated repos.
Diffusers pipelines are counted component by component (catalog/weights.py), and repos
that ship only PyTorch files are sized from the files.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from rightsize.catalog.weights import (
    count_files,
    pick_variant,
    pipeline_components,
    torch_weights,
    variant_groups,
)
from rightsize.types import Family, ModelFacts, ModelRef

HUB = "https://huggingface.co"
_TEXT_MODEL_KEYS = ("text_config", "language_config", "llm_config")

#: Config keys that decide how much KV cache a model really needs, beyond plain GQA:
#: which layers slide, whether attention is MLA, which layers are recurrent and how big
#: their state is. Kept raw so the fit engine can apply each runtime's own rules, which
#: differ (llama.cpp caches a sliding window only for architectures it implements it for).
_ATTENTION_KEYS = (
    "model_type",
    "layer_types",
    "sliding_window",
    "sliding_window_pattern",
    "use_sliding_window",
    "full_attention_interval",
    "attn_layer_period",
    "attn_layer_offset",
    "hybrid_override_pattern",
    "full_attn_idxs",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "hidden_size",
    "mamba_d_state",
    "mamba_d_conv",
    "mamba_expand",
    "mamba_n_groups",
    "mamba_n_heads",
    "mamba_d_head",
    "mamba_head_dim",
    "mamba_num_heads",
    "mamba_d_ssm",
    "ssm_state_size",
    "conv_kernel",
    "conv_L_cache",
    "n_groups",
    "expand",
    "linear_num_value_heads",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)


def _headers(token: str | None) -> dict[str, str]:
    tok = token or os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _cache_path(repo: str, revision: str) -> Path:
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    return base / "models" / repo.replace("/", "__") / f"{revision}.json"


#: Bump whenever facts() starts keeping something new. A cached entry from an older
#: version is refetched rather than trusted: without this, adding the attention fields left
#: every warm cache describing LFM2 as all-attention, 166% over what llama.cpp allocates.
CACHE_VERSION = 7


def _read_cache(path: Path, ttl_s: float) -> dict[str, Any] | None:
    try:
        if time.time() - path.stat().st_mtime > ttl_s:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.pop("_cache_version", None) != CACHE_VERSION:
        return None
    return payload


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**payload, "_cache_version": CACHE_VERSION}), encoding="utf-8")
    except OSError:
        pass


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in _TEXT_MODEL_KEYS:
        if isinstance(cfg.get(key), dict):
            return cfg[key]
    return cfg


_DIFFUSION_TAGS = {
    "text-to-image", "image-to-image", "text-to-video", "image-to-video", "video-to-video",
    "unconditional-image-generation",
}
_AUDIO_TAGS = {
    "automatic-speech-recognition", "text-to-speech", "text-to-audio", "audio-to-audio",
    "audio-classification", "voice-activity-detection",
}
_EMBEDDING_TAGS = {"feature-extraction", "sentence-similarity", "text-ranking"}
_VISION_TAGS = {
    "object-detection", "image-classification", "image-segmentation", "mask-generation",
    "depth-estimation", "zero-shot-image-classification", "zero-shot-object-detection",
    "image-feature-extraction", "keypoint-detection", "video-classification",
}


def _family(
    cfg: dict[str, Any], pipeline_tag: str | None, library: str | None = None,
    pipeline: bool = False,
) -> Family:
    mt = str(cfg.get("model_type", "")).lower()
    tag = (pipeline_tag or "").lower()
    lib = (library or "").lower()
    if pipeline or lib == "diffusers" or "diffusion" in mt or tag in _DIFFUSION_TAGS:
        return Family.diffusion
    if "whisper" in mt or tag in _AUDIO_TAGS:
        return Family.audio
    if lib == "sentence-transformers" or tag in _EMBEDDING_TAGS:
        return Family.embedding
    if tag in _VISION_TAGS:
        return Family.vision
    return Family.llm


def _moe_active(
    total: int, tc: dict[str, Any], experts: int, k: int, hidden: int | None
) -> tuple[int | None, str]:
    """Parameters that run for each token of a mixture-of-experts model.

    Only the routed experts are sparse. Attention, embeddings, the router and any shared
    experts run for every token, so the active count is the total minus the experts that sit
    idle: expert_params * (1 - k / experts), where each routed expert is a SwiGLU block of
    3 * hidden * moe_intermediate_size in every MoE layer.

    Scaling the whole model by k / experts, as this first did, ignored the always-on part:
    it put Qwen3-30B-A3B at 1.9B active where the name says 3B, and made MoE speed estimates
    up to 74% too fast."""
    inter = tc.get("moe_intermediate_size") or tc.get("intermediate_size")
    layers = tc.get("num_hidden_layers") or 0
    dense_lead = tc.get("first_k_dense_replace") or 0
    step = tc.get("decoder_sparse_step") or 1
    moe_layers = len([i for i in range(dense_lead, layers) if (i + 1) % step == 0])
    # Multi-token-prediction layers ship in the checkpoint but plain decoding never runs
    # them; DeepSeek-V3's is about 14B of its 685B, the gap between 51B and the stated 37B.
    mtp = tc.get("num_nextn_predict_layers") or tc.get("mtp_num_hidden_layers") or 0
    if inter and hidden and moe_layers:
        expert_params = moe_layers * experts * 3 * hidden * inter
        if 0 < expert_params < total:
            per_layer = expert_params / moe_layers + (total - expert_params) / (layers + mtp)
            idle = expert_params * (1 - k / experts) + mtp * per_layer
            how = "total minus idle routed experts"
            if mtp:
                how += f" and {mtp} prediction layer{'s' if mtp > 1 else ''}"
            return int(total - idle), how
    # The expert layout did not add up (latent-space experts, two-matrix MLPs, experts on
    # only some layers). Scaling the total by k / experts would under-count badly - it put
    # Nemotron-3-Super at 5.3B active against a stated 12B - and an under-count makes the
    # model look fast. Leave it unset: speed then assumes every weight is read, which errs
    # slow.
    return None, "unknown: the expert layout did not add up, so speed assumes a dense read"


_ACTIVE_IN_NAME = re.compile(r"[-_]A(\d+(?:\.\d+)?)B(?:[-_]|$)")


def _active_from_name(repo: str) -> int | None:
    """Publishers write the active size into mixture-of-experts names: 30B-A3B, 120B-A12B.
    It is rounded, but it is their number, and it survives architectures the formula below
    has not met."""
    m = _ACTIVE_IN_NAME.search(repo.split("/")[-1])
    return int(float(m.group(1)) * 1e9) if m else None


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
        info = c.get(
            f"{HUB}/api/models/{repo}", params={"revision": revision, "blobs": "true"}
        )
        info.raise_for_status()
        meta = info.json()
        sizes = {s["rfilename"]: s.get("size") for s in meta.get("siblings", [])}
        siblings = list(sizes)
        base = f"{HUB}/{repo}/resolve/{revision}"

        cfg: dict[str, Any] = {}
        if "config.json" in siblings:
            r = c.get(f"{base}/config.json")
            if r.status_code not in (401, 403):  # a gated repo still lists its files
                r.raise_for_status()
                cfg = r.json()

        pipeline = pipeline_components(c, base, sizes) if "model_index.json" in sizes else None
        shards: list[str] = []
        total = 0
        by_dtype: dict[str, int] = {}
        counted_from = None
        summary = (meta.get("safetensors") or {}).get("parameters")
        if pipeline and pipeline["components"]:
            # The Hub's summary covers one set of files, not the pipeline: for FLUX it is
            # the single-file transformer at the root, for SDXL the UNet alone.
            for comp in pipeline["components"].values():
                total += comp["params"]
                for k, v in comp["params_by_dtype"].items():
                    by_dtype[k] = by_dtype.get(k, 0) + v
            counted_from = "pipeline components"
            if any("file sizes" in comp["counted_from"]
                   for comp in pipeline["components"].values()):
                counted_from += ", from file sizes (the headers are gated)"
        elif isinstance(summary, dict) and summary:
            # The Hub already counts parameters per dtype in the response we just fetched.
            # Reading every shard header instead cost DeepSeek-V3 163 requests and minutes.
            by_dtype = {str(k): int(v) for k, v in summary.items()}
            total = sum(by_dtype.values())
            counted_from = "Hub parameter summary"
        else:
            root = [f for f in siblings if f.endswith(".safetensors") and "/" not in f]
            shards, chosen = pick_variant(root)
            if shards:
                total, by_dtype, counted_from = count_files(
                    c, base, shards, sizes, variant_groups(root), chosen
                )
            else:
                torch = torch_weights(sizes, cfg.get("torch_dtype"))
                if torch:
                    total = torch["params"]
                    by_dtype = {str(cfg.get("torch_dtype") or "float32").upper(): total}
                    counted_from = torch["counted_from"]
                    shards = torch["files"]

    tc = _text_config(cfg)
    hidden = tc.get("hidden_size")
    heads = tc.get("num_attention_heads")
    head_dim = tc.get("head_dim") or (hidden // heads if hidden and heads else None)
    kv_heads = tc.get("num_key_value_heads") or heads
    dominant = max(by_dtype, key=by_dtype.get) if by_dtype else cfg.get("torch_dtype")
    experts = tc.get("num_experts") or tc.get("num_local_experts") or tc.get("n_routed_experts")
    active_experts = tc.get("num_experts_per_tok") or tc.get("experts_per_token")

    facts_ = ModelFacts(
        ref=ModelRef(repo=repo, revision=revision),
        family=_family(cfg, meta.get("pipeline_tag"), meta.get("library_name"),
                       pipeline=bool(pipeline and pipeline["components"])),
        params_total=total or None,
        params_active=None,
        dtype=str(dominant).upper() if dominant else None,
        num_layers=tc.get("num_hidden_layers"),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        context_max=tc.get("max_position_embeddings"),
        license=(meta.get("cardData") or {}).get("license"),
        base_model=_base_model(meta),
        confidence=_confidence(total, counted_from),
        extra={
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "params_by_dtype": by_dtype,
            "params_counted_from": counted_from,
            "shards": shards,
            "library": meta.get("library_name"),
            "pipeline_tag": meta.get("pipeline_tag"),
            "pipeline": pipeline,
            "gated": meta.get("gated", False),
            "private": meta.get("private"),
            "num_experts": experts,
            "num_experts_per_tok": active_experts,
            "sliding_window": tc.get("sliding_window"),
            "attention": {k: tc[k] for k in _ATTENTION_KEYS if tc.get(k) is not None},
            "tie_word_embeddings": cfg.get("tie_word_embeddings"),
            "vocab_size": tc.get("vocab_size"),
            "hidden_size": hidden,
            "intermediate_size": tc.get("intermediate_size"),
            "num_attention_heads": heads,
            "image_size": tc.get("image_size"),
            "patch_size": tc.get("patch_size"),
        },
    )
    if experts and active_experts and total:
        stated = _active_from_name(repo)
        if stated:
            facts_.params_active = stated
            facts_.extra["params_active_note"] = "stated in the model name"
        else:
            active, how = _moe_active(total, tc, experts, active_experts, hidden)
            facts_.params_active = active
            facts_.extra["params_active_note"] = how
    if transport is None:
        _write_cache(cache, facts_.model_dump(mode="json"))
    return facts_


def _confidence(total: int, counted_from: str | None) -> float:
    """1.0 when the parameters were counted from the weights themselves; less when they
    were inferred from file sizes, which means assuming a dtype."""
    if not total:
        return 0.5
    if counted_from and "file sizes" in counted_from:
        return 0.7
    return 1.0


def _base_model(meta: dict[str, Any]) -> str | None:
    card = meta.get("cardData") or {}
    bm = card.get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    return bm
