"""KV cache by architecture, against real model configs (tests/fixtures/facts).

Every fixture is a ModelFacts fetched from the Hub by the catalog itself, so these tests
exercise the config fields the catalog keeps as well as the rules applied to them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rightsize.fit import estimate, kv
from rightsize.types import Device, ModelFacts

FIXTURES = Path(__file__).parent / "fixtures" / "facts"


def _facts(name: str) -> ModelFacts:
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"fixture {name} not recorded")
    return ModelFacts.model_validate_json(path.read_text(encoding="utf-8"))


def _gqa(fx: ModelFacts, ctx: int) -> float:
    return 2 * fx.num_layers * fx.num_kv_heads * fx.head_dim * ctx * 2 / 1e9


def test_plain_gqa_is_unchanged() -> None:
    """Qwen3 is ordinary GQA, and llama-fit-params confirmed the formula exactly: 896 MiB
    for Qwen3-1.7B at 8192 tokens."""
    fx = _facts("Qwen__Qwen3-4B")
    assert kv.layout(fx).kind == "gqa"
    assert kv.kv_cache_gb(fx, 131072) == pytest.approx(_gqa(fx, 131072), rel=1e-3)


def test_gemma3_keeps_the_window_on_five_layers_in_six() -> None:
    """Capping every layer at the window, as the old code did, put Gemma 3 27B at 128K at
    about 0.5 GB. Ignoring the window gives 66.6 GB. llama.cpp's layout is 11.4 GB: ten
    full layers and 52 that hold 1024 tokens plus a ubatch, padded to 256."""
    fx = _facts("unsloth__gemma-3-27b-it")
    lay = kv.layout(fx)
    assert (lay.kind, lay.full_layers, lay.swa_layers, lay.window) == ("sliding", 10, 52, 1024)
    got = kv.kv_cache_gb(fx, 131072)
    assert got == pytest.approx(11.39, abs=0.02)
    assert got < _gqa(fx, 131072) / 5
    # the window part does not grow with context
    parts_8k, parts_128k = kv.kv_breakdown(fx, 8192), kv.kv_breakdown(fx, 131072)
    assert parts_8k["kv_window"] == parts_128k["kv_window"]


def test_swa_full_turns_the_window_off() -> None:
    fx = _facts("unsloth__gemma-3-27b-it")
    assert kv.kv_cache_gb(fx, 131072, swa_full=True) == pytest.approx(_gqa(fx, 131072), rel=1e-3)


def test_a_per_layer_list_beats_the_default_pattern() -> None:
    """gpt-oss publishes layer_types; half its layers slide over 128 tokens."""
    fx = _facts("openai__gpt-oss-20b")
    lay = kv.layout(fx)
    assert (lay.full_layers, lay.swa_layers, lay.window) == (12, 12, 128)


def test_a_config_window_alone_changes_nothing() -> None:
    """llama.cpp applies a window only for architectures it implements one for, so a plain
    llama-architecture model declaring sliding_window 4096 still caches everything."""
    fx = _facts("Qwen__Qwen3-4B").model_copy(
        update={"extra": {"model_type": "mistral", "attention": {"sliding_window": 4096}}}
    )
    lay = kv.layout(fx)
    assert lay.kind == "gqa" and lay.notes, "the ignored window is noted, not silently dropped"
    assert kv.kv_cache_gb(fx, 32768) == pytest.approx(_gqa(fx, 32768), rel=1e-3)


def test_mla_caches_one_latent_and_no_values() -> None:
    """DeepSeek-V3: 61 layers x (512 + 64) x 2 bytes = 70 KB a token, no V cache. The GQA
    formula over its 128 KV heads would ask for hundreds of gigabytes."""
    fx = _facts("deepseek-ai__DeepSeek-V3")
    lay = kv.layout(fx)
    assert (lay.kind, lay.k_elems, lay.v_elems) == ("mla", 576, 0)
    assert kv.kv_cache_gb(fx, 131072) == pytest.approx(61 * 576 * 2 * 131072 / 1e9, rel=1e-3)
    # What caching the uncompressed heads would cost: 128 heads, K of qk_nope + qk_rope wide
    # and V of v_head_dim. hidden / heads (56 here) is not an attention width for MLA models,
    # so the plain formula over the catalog's head_dim is not the right comparison either.
    a = fx.extra["attention"]
    k_w, v_w = a["qk_nope_head_dim"] + a["qk_rope_head_dim"], a["v_head_dim"]
    uncompressed = fx.num_layers * fx.num_kv_heads * (k_w + v_w) * 131072 * 2 / 1e9
    assert uncompressed == pytest.approx(655, abs=1)
    assert kv.kv_cache_gb(fx, 131072) < uncompressed / 70


def test_qwen3_next_caches_one_layer_in_four() -> None:
    fx = _facts("Qwen__Qwen3-Next-80B-A3B-Instruct")
    lay = kv.layout(fx)
    assert lay.kind == "hybrid"
    assert lay.full_layers == fx.num_layers // 4
    assert lay.recurrent_layers == fx.num_layers - lay.full_layers
    assert lay.recurrent_elems, "gated delta-net state size comes from the linear_* fields"
    # the recurrent state is fixed: it does not grow with context
    short, long = kv.kv_breakdown(fx, 4096), kv.kv_breakdown(fx, 131072)
    assert short["recurrent_state"] == long["recurrent_state"] > 0


def test_every_hybrid_fixture_gets_a_layout() -> None:
    for name in (
        "LiquidAI__LFM2-350M",
        "ibm-granite__granite-4.0-h-small",
        "tiiuae__Falcon-H1-1.5B-Instruct",
    ):
        fx = _facts(name)
        lay = kv.layout(fx)
        assert lay.kind == "hybrid", name
        assert 0 < lay.full_layers <= fx.num_layers, name
        assert lay.recurrent_elems, f"{name}: recurrent state not derived from its config"


def test_falcon_h1_has_attention_and_state_in_every_layer() -> None:
    fx = _facts("tiiuae__Falcon-H1-1.5B-Instruct")
    lay = kv.layout(fx)
    assert lay.full_layers == lay.recurrent_layers == fx.num_layers


def test_estimate_explains_an_unusual_layout() -> None:
    fx = _facts("unsloth__gemma-3-27b-it")
    dev = Device(name="big", memory_gib=80, bandwidth_gbps=2000)
    r = estimate(fx, "Q4_K_M", dev, ctx=131072)
    assert any("sliding window" in n for n in r.notes)
    assert r.breakdown["kv_window"] > 0 and r.breakdown["kv_full"] > 0
