"""The formats outside GGUF on llama.cpp's quality scale (F4)."""

from __future__ import annotations

import pytest

from rightsize.fit.quality import band, format_delta, ppl_delta


def test_a_format_is_worth_the_gguf_type_with_its_kl_divergence() -> None:
    """On Qwen3-0.6B, FP8's KLD (0.020) falls between Q6_K's (0.0135) and Q5_K_M's (0.037),
    so its cost lands between theirs on Llama-3-8B: 0.0217 and 0.0569."""
    delta, how = format_delta("fp8")
    assert 0.0217 < delta < 0.0569
    assert "between GGUF Q6_K and Q5_K_M" in how and "Qwen3-0.6B" in how
    assert band(delta) == "near-lossless"


def test_the_gate_orders_the_formats_as_it_measured_them() -> None:
    costs = {f: format_delta(f)[0] for f in ("openvino-int8", "fp8", "nf4", "openvino-int4")}
    assert costs["openvino-int8"] < costs["fp8"] < costs["nf4"] < costs["openvino-int4"]
    # the 4-bit formats sit where llama.cpp's Q4_K_S and Q4_0 do on the same model
    assert ppl_delta("Q4_K_S")[0] < costs["nf4"] < ppl_delta("Q4_0")[0]
    assert band(costs["nf4"]) == "noticeable"


def test_ppl_delta_reaches_the_formats_by_name() -> None:
    assert ppl_delta("awq") == format_delta("awq")
    assert ppl_delta("AWQ")[0] == format_delta("awq")[0]
    # one row per model it was measured on, averaged
    delta, how = format_delta("awq")
    assert how.count("by KL divergence on") == 2 and "Qwen3-1.7B" in how
    assert ppl_delta("Q4_K_M")[0] < delta < ppl_delta("Q3_K_M")[0]


def test_an_unmeasured_format_says_so() -> None:
    assert format_delta("exl3") == (None, "unknown format")
    assert ppl_delta("exl3") == (None, "unknown file type")


def test_the_scale_holds_its_ends() -> None:
    from rightsize.fit.quality import _bridge, _ladders

    ladder = _ladders()["Qwen/Qwen3-0.6B"]
    assert _bridge(0.0, ladder) == (pytest.approx(ppl_delta("Q8_0")[0]), "at most GGUF Q8_0")
    assert _bridge(99.0, ladder)[1] == "at least GGUF Q2_K"
