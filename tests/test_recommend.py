"""Recommendations as invariants (F4).

The pool is regenerated from live download data, so these check properties that must hold
whatever the pool holds, not particular model names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rightsize
from rightsize.rules.recommend import (
    QUALITY_FLOORS,
    RANK_ID,
    effective_params_b,
    recommend_for_model,
    recommend_result,
)
from rightsize.types import ModelFacts, Plan, Verdict

FIXTURES = Path(__file__).parent / "fixtures" / "facts"


def _serve(p: Plan):
    return p.steps[-1]


@pytest.mark.parametrize("device", ["RTX 3060 12GB", "RTX 4070 Ti SUPER", "M4 Max 64GB", "H100"])
def test_every_plan_fits_and_explains_itself(device: str) -> None:
    plans = rightsize.recommend("chat", device)
    assert plans, f"nothing recommended for {device}"
    for i, p in enumerate(plans, start=1):
        assert p.rank == i
        serve = _serve(p)
        assert serve.stage == "serve" and serve.fit.verdict in (Verdict.fits, Verdict.tight)
        assert any(line.startswith(RANK_ID) for line in p.trace), "the score shows its working"
        assert p.quality_penalty is not None and p.quality_penalty <= QUALITY_FLOORS["noticeable"]
    assert [p.score for p in plans] == sorted((p.score for p in plans), reverse=True)
    assert len({p.model.ref.repo for p in plans}) == len(plans), "one plan per model"


def test_interactive_plans_are_fast_enough_unless_asked() -> None:
    slow_device = "CPU only 32GB"
    for p in rightsize.recommend("chat", slow_device):
        assert _serve(p).fit.speed is None or _serve(p).fit.speed >= 5
    more = recommend_result("chat", slow_device, allow_slow=True, top_k=50).plans
    fewer = recommend_result("chat", slow_device, top_k=50).plans
    assert len(more) >= len(fewer)


def test_quality_floor_is_respected() -> None:
    for p in rightsize.recommend("chat", "RTX 4090", quality="near-lossless"):
        assert p.quality_penalty <= QUALITY_FLOORS["near-lossless"]


def test_coding_prefers_code_models_on_equal_footing() -> None:
    plans = rightsize.recommend("coding", "RTX 3060 12GB")
    assert any("code model" in line for line in plans[0].trace)


def test_agentic_never_goes_below_the_tool_calling_floor() -> None:
    for p in recommend_result("agentic", "M4 Max 64GB", quality="any", top_k=30).plans:
        assert _serve(p).quant.bits_per_weight >= 2.8


def test_unified_memory_devices_say_the_share_is_assumed() -> None:
    for device in ("M4 Max 64GB", "Jetson Orin Nano 8GB"):
        plans = rightsize.recommend("chat", device)
        assert plans
        assert any("unified_memory_share_is_an_assumption" in t for t in plans[0].trace)


def test_two_stage_plans_fine_tune_where_they_fit() -> None:
    plans = rightsize.recommend(
        "chat", "RTX 4070 Ti SUPER", finetune_device="T4", mode="qlora", top_k=10
    )
    assert plans
    for p in plans:
        ft = p.steps[0]
        assert ft.stage == "finetune" and ft.framework == "unsloth"
        assert ft.fit.verdict in (Verdict.fits, Verdict.tight)
        assert ft.device.name.startswith("T4")


def test_model_first_lists_quantizations_and_renders_commands() -> None:
    fx = ModelFacts.model_validate_json((FIXTURES / "Qwen__Qwen3-4B.json").read_text("utf-8"))
    result = recommend_for_model(fx, "RTX 3060 12GB")
    quants = [_serve(p).quant.variant for p in result.plans]
    assert "Q8_0" in quants and len(set(quants)) == len(quants)
    iq = next(p for p in result.plans if _serve(p).quant.variant == "IQ4_XS")
    steps = [s.text for s in iq.render()]
    assert steps[0].startswith("python convert_hf_to_gguf.py")
    assert any("llama-imatrix" in s for s in steps), "i-quants get an importance matrix"
    assert "--imatrix" in steps[-1]


def test_when_nothing_fits_the_reasons_come_back() -> None:
    result = recommend_result("chat", "Arc A380", quality="near-lossless", top_k=5)
    if not result.plans:
        assert result.rejected
        assert any("does not fit" in r.reason for r in result.rejected)


def test_moe_counts_as_the_geometric_mean() -> None:
    fx = ModelFacts.model_validate_json(
        (FIXTURES / "Qwen__Qwen3-Next-80B-A3B-Instruct.json").read_text("utf-8")
    )
    eff, note = effective_params_b(fx)
    total, active = fx.params_total / 1e9, fx.params_active / 1e9
    assert eff == pytest.approx((total * active) ** 0.5)
    assert "rule of thumb" in note


def test_plans_round_trip_as_json() -> None:
    p = rightsize.recommend("chat", "RTX 4090")[0]
    assert Plan.model_validate_json(p.to_json()) == p
