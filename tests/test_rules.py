"""The condition language and every rule's own test cases (F4)."""

from __future__ import annotations

import pytest

from rightsize.rules.engine import evaluate, load_rules
from rightsize.rules.expr import UNKNOWN, Condition, ExprError

RULES = load_rules()


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_rule_fires_and_stays_silent_as_it_says(rule) -> None:
    """Each rule ships a case that must fire and one that must not; a condition that drifts
    from its intent fails here rather than quietly never firing."""
    assert rule.fires(rule.test["fires"]), f"{rule.id} did not fire on its own fires case"
    assert not rule.fires(rule.test["silent"]), f"{rule.id} fired on its own silent case"


def test_rules_are_sourced_and_unique() -> None:
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))
    assert all(r.source_url.startswith("https://") for r in RULES)
    assert len(RULES) >= 30


def test_conditions_compare_and_combine() -> None:
    c = Condition("device.vendor == 'nvidia' and device.compute_capability < 8.9")
    assert c({"device": {"vendor": "nvidia", "compute_capability": 8.6}})
    assert not c({"device": {"vendor": "nvidia", "compute_capability": 8.9}})
    assert Condition("quant.type in ['IQ2_XXS', 'IQ1_S']")({"quant": {"type": "IQ1_S"}})


def test_missing_data_is_unknown_and_never_fires() -> None:
    """Three-valued logic. With plain booleans `not device.os == 'macos'` was True for a
    device whose OS we did not know, so a negated block rule fired on missing data."""
    negated = Condition("not device.os == 'macos'")
    assert negated.truth({"device": {}}) is UNKNOWN
    assert negated({"device": {}}) is False
    # but a definite answer on one side of `or` still decides it
    either = Condition("device.os == 'linux' or device.vendor == 'nvidia'")
    assert either({"device": {"vendor": "nvidia"}}) is True
    both = Condition("device.os == 'linux' and device.vendor == 'nvidia'")
    assert both.truth({"device": {"vendor": "amd"}}) is False


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('x')",
        "device.__class__",
        "device['vendor']",
        "(lambda: 1)()",
        "devcie.vendor == 'nvidia'",
        "1 if device else 2",
        "[x for x in device]",
    ],
)
def test_anything_but_comparisons_is_refused_at_load(source: str) -> None:
    with pytest.raises(ExprError):
        Condition(source)


def test_evaluate_collects_effects() -> None:
    ctx = {
        "quant": {
            "method": "gguf",
            "type": "IQ2_XXS",
            "requires_imatrix": True,
            "is_iquant": True,
            "band": "unknown",
            "bits": 2.38,
        },
        "stage": "serve",
        "task": "agentic",
        "device": {"vendor": "apple", "unified_memory": True},
        "plan": {"tok_s": 30, "verdict": "tight", "ctx": 8192},
        "model": {"context_max": 131072},
    }
    out = evaluate(ctx)
    fired = {r.id for r in out.fired}
    assert {
        "imatrix_required_by_llama_quantize",
        "low_bits_break_tool_calling",
        "iquants_slow_on_apple",
        "unified_memory_share_is_an_assumption",
        "tight_fit",
    } <= fired
    assert out.blocked and "imatrix" in out.requires
    assert out.multiplier == pytest.approx(0.9 * 0.95)
    assert all("(https://" in line for line in out.trace())
