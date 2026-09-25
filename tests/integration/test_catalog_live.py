"""Live Hub reads. Marked `network`; excluded from the default CI run."""

from __future__ import annotations

import pytest

from rightsize.catalog import facts
from rightsize.fit import estimate, predicted_file_gb
from rightsize.hardware import get

pytestmark = pytest.mark.network


def test_qwen3_0_6b_facts_from_hub() -> None:
    fx = facts("Qwen/Qwen3-0.6B")
    assert 0.5e9 < fx.params_total < 0.9e9
    assert (fx.num_layers, fx.num_kv_heads, fx.head_dim) == (28, 8, 128)
    assert fx.dtype == "BF16" and fx.family.value == "llm"


def test_qwen3_4b_q4_k_m_prediction_matches_known_upload() -> None:
    # bartowski/Qwen_Qwen3-4B-GGUF Q4_K_M is 2.50 GB on the Hub.
    fx = facts("Qwen/Qwen3-4B")
    assert predicted_file_gb(fx, "Q4_K_M") == pytest.approx(2.50, rel=0.05)
    r = estimate(fx, "Q4_K_M", get("RTX 3060"), ctx=8192)
    assert r.verdict.value == "fits"
