"""KV cache and recurrent state by architecture, the way llama.cpp allocates them (F3).

The plain formula, 2 x layers x kv_heads x head_dim x ctx x bytes, holds for ordinary GQA
models and is wrong for three families, each in its own direction:

* **Sliding window** (Gemma 2 and 3, Llama 4, gpt-oss): most layers cache only a window, a
  few cache the full context. Capping every layer at the window under-states the cache;
  ignoring the window over-states it.
* **MLA** (DeepSeek V2/V3, Kimi K2): one compressed latent per token and layer, and no V
  cache at all. The GQA formula over-states DeepSeek-V3 at 128K about seventy-fold.
* **Hybrid** (Qwen3-Next, Jamba, Nemotron-H, Granite 4, LFM2, Falcon-H1): recurrent layers
  hold a small fixed state instead of a KV cache; only attention layers grow with context.

Which rule applies is runtime behaviour, not a model property - llama.cpp caches a window
only for architectures it implements sliding attention for - so the rules are data, in
``data/runtimes/llama.cpp/kv_cache.yaml``, each citing the source file it was read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rightsize._data import load_yaml
from rightsize.types import GB, ModelFacts


@dataclass(frozen=True)
class KvLayout:
    """Which layers cache what, per token."""

    kind: str  # "gqa", "sliding", "mla" or "hybrid"
    full_layers: int  # layers caching the whole context
    swa_layers: int = 0  # layers caching only the window
    window: int | None = None
    k_elems: int = 0  # K elements per token per caching layer
    v_elems: int = 0  # V elements per token per caching layer (0 for MLA)
    recurrent_layers: int = 0
    recurrent_elems: int | None = None  # state per recurrent layer and sequence
    rule: str = "plain GQA"
    source_url: str | None = None
    notes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _rules() -> dict[str, Any]:
    return load_yaml("runtimes/llama.cpp/kv_cache.yaml")


def _arch_rule(model_type: str | None) -> dict[str, Any]:
    if not model_type:
        return {}
    for rec in _rules()["architectures"]:
        if model_type in rec["model_types"]:
            return rec
    return {}


def _attention_config(facts: ModelFacts) -> dict[str, Any]:
    extra = facts.extra or {}
    cfg = dict(extra.get("attention") or {})
    # facts cached before the attention block existed still carry these two
    cfg.setdefault("model_type", extra.get("model_type"))
    if extra.get("sliding_window") is not None:
        cfg.setdefault("sliding_window", extra["sliding_window"])
    return cfg


def layout(facts: ModelFacts) -> KvLayout:
    """Decide how this model's cache is laid out under llama.cpp."""
    if not (facts.num_layers and facts.num_kv_heads and facts.head_dim):
        raise ValueError("num_layers, num_kv_heads and head_dim are required for the KV cache")
    n = facts.num_layers
    gqa = facts.num_kv_heads * facts.head_dim
    cfg = _attention_config(facts)
    rule = _arch_rule(cfg.get("model_type"))
    src = rule.get("source_url")

    if rule.get("mla") or (cfg.get("kv_lora_rank") and cfg.get("qk_rope_head_dim")):
        k = int(cfg["kv_lora_rank"]) + int(cfg["qk_rope_head_dim"])
        return KvLayout(
            kind="mla",
            full_layers=n,
            k_elems=k,
            v_elems=0,
            rule=f"MLA: one {k}-wide latent per token and layer, no V cache",
            source_url=src,
        )

    if hybrid := rule.get("hybrid"):
        return _hybrid(facts, cfg, hybrid, gqa, src)

    if (sliding := rule.get("sliding")) and (
        window := cfg.get(sliding.get("window_key") or "") or sliding.get("window_default")
    ):
        mask = _sliding_mask(cfg, n, int(sliding["pattern"]))
        swa = sum(mask)
        return KvLayout(
            kind="sliding",
            full_layers=n - swa,
            swa_layers=swa,
            window=int(window),
            k_elems=gqa,
            v_elems=gqa,
            rule=f"sliding window: {swa} of {n} layers cache {int(window)} tokens",
            source_url=src,
        )

    notes = []
    if cfg.get("sliding_window") and not rule:
        # llama.cpp only applies a window for architectures it implements one for; a config
        # value alone (Mistral 7B v0.1, Qwen2 with use_sliding_window false) changes nothing.
        notes.append("config declares a sliding window; llama.cpp caches the full context")
    return KvLayout(kind="gqa", full_layers=n, k_elems=gqa, v_elems=gqa, notes=notes)


def _sliding_mask(cfg: dict[str, Any], n: int, pattern: int) -> list[bool]:
    """True for each layer that slides. A per-layer list in the config wins; otherwise
    llama.cpp's set_swa_pattern: layer i slides when i % pattern < pattern - 1."""
    types = cfg.get("layer_types")
    if isinstance(types, list) and len(types) >= n:
        return ["sliding" in str(t) for t in types[:n]]
    pattern = int(cfg.get("sliding_window_pattern") or pattern)
    return [i % pattern < pattern - 1 for i in range(n)]


def _hybrid(
    facts: ModelFacts, cfg: dict[str, Any], hybrid: dict[str, Any], gqa: int, src: str | None
) -> KvLayout:
    n = facts.num_layers or 0
    how = hybrid["rule"]
    if how == "parallel":
        attn = [True] * n
        recurrent = n
    else:
        attn = _attention_mask(cfg, n, how, hybrid)
        recurrent = n - sum(attn)
    elems = _recurrent_elems(cfg)
    notes = [] if elems is not None else ["recurrent state size not in the config; not counted"]
    return KvLayout(
        kind="hybrid",
        full_layers=sum(attn),
        k_elems=gqa,
        v_elems=gqa,
        recurrent_layers=recurrent,
        recurrent_elems=elems,
        rule=f"hybrid: {sum(attn)} of {n} layers keep a KV cache, {recurrent} keep a fixed state",
        source_url=src,
        notes=notes,
    )


def _attention_mask(cfg: dict[str, Any], n: int, how: str, hybrid: dict[str, Any]) -> list[bool]:
    # An explicit index list is unambiguous whatever the family (LFM2: full_attn_idxs).
    # Guessing here once cost 166%: every LFM2 layer was counted as attention, where
    # llama.cpp caches six of sixteen.
    idxs = cfg.get("full_attn_idxs")
    if isinstance(idxs, list) and idxs:
        wanted = {int(i) for i in idxs}
        return [i in wanted for i in range(n)]
    types = cfg.get("layer_types")
    if isinstance(types, list) and len(types) >= n:
        # "full_attention", "sliding_attention", "attention" keep a cache; "linear_attention",
        # "mamba", "conv" keep a state
        return [("attention" in str(t) and "linear" not in str(t)) for t in types[:n]]
    if how == "full_attention_interval":
        k = int(cfg.get("full_attention_interval") or hybrid.get("interval_default") or 4)
        return [(i + 1) % k == 0 for i in range(n)]
    if how == "attn_layer_period":
        period = int(cfg.get("attn_layer_period") or 8)
        offset = int(cfg.get("attn_layer_offset") or 0)
        return [i % period == offset for i in range(n)]
    if how == "hybrid_override_pattern" and isinstance(cfg.get("hybrid_override_pattern"), str):
        pat = cfg["hybrid_override_pattern"]
        return [i < len(pat) and pat[i] == "*" for i in range(n)]
    # no way to tell which layers attend: assume all do, which over-states rather than under
    return [True] * n


def _recurrent_elems(cfg: dict[str, Any]) -> int | None:
    """Floats of state per recurrent layer and sequence, llama.cpp's n_embd_s + n_embd_r:

        d_state * d_inner + (d_conv - 1) * (d_inner + 2 * n_groups * d_state)

    Field names differ by model; each branch maps one family's config onto that formula.
    """
    g = cfg.get
    hidden = g("hidden_size")
    if g("linear_num_value_heads") and g("linear_value_head_dim") and g("linear_key_head_dim"):
        # gated delta-net (Qwen3-Next): the converter maps these onto the ssm_* fields
        d_inner = g("linear_num_value_heads") * g("linear_value_head_dim")
        d_state, groups = g("linear_key_head_dim"), g("linear_num_key_heads") or 1
        d_conv = g("linear_conv_kernel_dim") or 4
    elif g("conv_L_cache") and hidden:
        # LFM2 short convolutions: n_embd_r = n_embd * (L_cache - 1), no state matrix
        return int(hidden * (g("conv_L_cache") - 1))
    else:
        heads = g("mamba_n_heads") or g("mamba_num_heads")
        head_dim = g("mamba_d_head") or g("mamba_head_dim")
        d_inner = g("mamba_d_ssm") or (heads * head_dim if heads and head_dim else None)
        if d_inner is None and hidden and (g("mamba_expand") or g("expand")):
            d_inner = (g("mamba_expand") or g("expand")) * hidden
        d_state = g("mamba_d_state") or g("ssm_state_size")
        groups = g("mamba_n_groups") or g("n_groups") or 0
        d_conv = g("mamba_d_conv") or g("conv_kernel") or 4
        if not (d_inner and d_state):
            return None
    return int(d_state * d_inner + (d_conv - 1) * (d_inner + 2 * groups * d_state))


def _pad(n: int, to: int) -> int:
    return (n + to - 1) // to * to


def kv_breakdown(
    facts: ModelFacts,
    ctx: int,
    batch: int = 1,
    kv_bytes: float = 2.0,
    *,
    swa_full: bool = False,
) -> dict[str, float]:
    """Cache memory in decimal GB, split so the estimate can say where it went."""
    lay = layout(facts)
    d = _rules()["defaults"]
    per_token = lay.k_elems + lay.v_elems
    full = lay.full_layers * per_token * ctx * batch * kv_bytes
    swa = 0.0
    if lay.swa_layers:
        # llama_kv_cache_iswa: pad(min(ctx, n_swa * n_seq + n_ubatch), 256) cells - the pad
        # comes after the min, so a short context still rounds up to a whole 256 cells
        wanted = min(ctx, lay.window * batch + d["n_ubatch"])
        cells = ctx if swa_full else _pad(wanted, d["swa_pad"])
        swa = lay.swa_layers * per_token * cells * kv_bytes
    state = 0.0
    if lay.recurrent_layers and lay.recurrent_elems:
        state = lay.recurrent_layers * lay.recurrent_elems * batch * d["recurrent_state_bytes"]
    return {
        "kv_full": round(full / GB, 4),
        "kv_window": round(swa / GB, 4),
        "recurrent_state": round(state / GB, 4),
    }


def kv_cache_gb(
    facts: ModelFacts, ctx: int, batch: int = 1, kv_bytes: float = 2.0, *, swa_full: bool = False
) -> float:
    """Total context-dependent memory: KV for attention layers plus recurrent state."""
    return round(sum(kv_breakdown(facts, ctx, batch, kv_bytes, swa_full=swa_full).values()), 4)


def describe(facts: ModelFacts) -> str:
    """One line for FitResult.notes, so a surprising number explains itself."""
    lay = layout(facts)
    return lay.rule if lay.kind != "gqa" else "plain GQA: every layer caches the full context"
