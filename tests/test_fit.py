"""Golden tests for the GGUF fit estimator against published numbers."""

from __future__ import annotations

import pytest

from rightsize.fit import estimate, gguf_bpw, kv_cache_gb, predicted_file_gb
from rightsize.types import Device, Family, ModelFacts, ModelRef, Verdict


def _llama_31_70b() -> ModelFacts:
    return ModelFacts(
        ref=ModelRef(repo="meta-llama/Llama-3.1-70B"),
        family=Family.llm,
        params_total=70_553_706_496,
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        context_max=131072,
    )


def _qwen3_4b() -> ModelFacts:
    return ModelFacts(
        ref=ModelRef(repo="Qwen/Qwen3-4B"),
        family=Family.llm,
        params_total=4_022_468_096,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
        context_max=40960,
    )


def test_kv_cache_llama31_70b_at_128k_is_about_43gb() -> None:
    # 2 x 80 layers x 8 kv heads x 128 dim x 131072 ctx x 2 bytes = 42.95 GB
    assert kv_cache_gb(_llama_31_70b(), 131072) == pytest.approx(42.95, abs=0.05)


def test_real_bpw_not_nominal() -> None:
    # llama.cpp's measured effective bits on Llama-3.1-8B (tools/quantize/README.md)
    assert gguf_bpw("Q4_K_M") == (4.8944, "measured by llama.cpp")
    assert gguf_bpw("Q8_0")[0] == pytest.approx(8.5008, abs=1e-4)
    # an i-quant's nominal bits undercount its mix: IQ4_XS is 4.46 effective, not 4.25
    assert gguf_bpw("IQ4_XS")[0] == pytest.approx(4.4597, abs=1e-4)
    assert gguf_bpw("Q4_K")[0] == 4.5  # a tensor type, not a file type: from the block formula
    assert gguf_bpw("q6_k")[0] > gguf_bpw("q5_k_m")[0] > gguf_bpw("q4_k_m")[0]


def test_predicted_file_size_qwen3_4b_q4_k_m() -> None:
    # 4.02B x 4.899 / 8 = 2.46 GB; bartowski's Qwen3-4B-Q4_K_M.gguf is 2.50 GB.
    assert predicted_file_gb(_qwen3_4b(), "Q4_K_M") == pytest.approx(2.46, abs=0.03)


def test_estimate_fits_on_16gb_and_reports_breakdown() -> None:
    dev = Device(
        name="RTX 4070 Ti SUPER 16GB",
        vendor="nvidia",
        memory_gib=16,
        bandwidth_gbps=672,
        usable_fraction=0.92,
    )
    r = estimate(_qwen3_4b(), "Q4_K_M", dev, ctx=8192)
    assert r.verdict is Verdict.fits
    assert set(r.breakdown) >= {"weights", "kv_cache", "overhead", "usable_memory"}
    assert r.vram_gb == pytest.approx(
        r.breakdown["weights"] + r.breakdown["kv_cache"] + r.breakdown["overhead"], abs=0.02
    )
    assert r.speed and 50 < r.speed < 250  # bandwidth-bound decode on a 672 GB/s card
    assert r.formula_id == "llm.gguf.analytic.v1"
    assert 0 < r.confidence < 1


def test_estimate_no_fit_and_offload_verdicts() -> None:
    small = Device(
        name="RTX 3060 12GB", vendor="nvidia", memory_gib=12, bandwidth_gbps=360, system_ram_gib=32
    )
    r = estimate(_llama_31_70b(), "Q4_K_M", small, ctx=4096)
    assert r.verdict in (Verdict.no_fit, Verdict.offload)
    big = Device(name="H100 80GB", vendor="nvidia", memory_gib=80, bandwidth_gbps=3350)
    assert estimate(_llama_31_70b(), "Q4_K_M", big, ctx=4096).verdict is Verdict.fits


def test_unknown_quant_raises() -> None:
    with pytest.raises(KeyError):
        gguf_bpw("Q9_ULTRA")


def test_a_model_between_the_two_units_still_fits() -> None:
    """16 GiB is 17.18 GB, so a 16.7 GB model fits with room to spare. Under the old
    mix-up the same model was judged against 16.0 and came back no_fit."""
    dev = Device(name="16 GiB card", vendor="nvidia", memory_gib=16, usable_fraction=1.0)
    assert dev.memory_gb == 17.18
    facts = _qwen3_4b().model_copy(update={"params_total": 26_500_000_000})
    r = estimate(facts, "Q4_K_M", dev, ctx=512)
    assert r.breakdown["usable_memory"] == pytest.approx(17.18, abs=0.01), "budget is decimal GB"
    assert 16.0 < r.vram_gb <= dev.memory_gb, "sits between the decimal and binary readings"
    assert r.verdict is not Verdict.no_fit


def test_an_llm_is_sized_at_a_format_outside_gguf() -> None:
    """estimate --quant fp8 on an LLM: the formats table sizes it, and a QuantSpec that says
    what the toolkit leaves 16-bit is sized the way the --to predictor sizes the file."""
    import json
    from pathlib import Path

    from rightsize.hardware import resolve
    from rightsize.types import QuantSpec

    path = Path(__file__).parent / "fixtures" / "facts" / "Qwen__Qwen3-4B.json"
    facts = ModelFacts.model_validate(json.loads(path.read_text(encoding="utf-8")))
    card = resolve("RTX 4090")
    plain = estimate(facts, "fp8", card, ctx=512)
    assert plain.breakdown["weights"] == pytest.approx(facts.params_total * 8 / 8 / 1e9, abs=0.01)
    spec = QuantSpec(method="fp8", variant="fp8", bits_per_weight=8.0, embedding_bits=16)
    kept = estimate(facts, spec, card, ctx=512)
    table = 151936 * 2560 * 8 / 8 / 1e9  # the embedding at 16 bits costs another 8 bits each
    assert kept.breakdown["weights"] - plain.breakdown["weights"] == pytest.approx(table, abs=0.01)
    assert "embedding at 16 bits" in kept.notes[0]


def test_bits_that_fit_inverts_the_estimate() -> None:
    """The bits a card has room for are the bits at which the estimate says "fits" with
    nothing to spare: sizing the model at them lands on the usable memory less the headroom."""
    import json
    from pathlib import Path

    from rightsize.fit.llm import _HEADROOM, bits_that_fit
    from rightsize.hardware import resolve
    from rightsize.types import QuantSpec

    path = Path(__file__).parent / "fixtures" / "facts" / "Qwen__Qwen3-4B.json"
    facts = ModelFacts.model_validate(json.loads(path.read_text(encoding="utf-8")))
    card = resolve("RTX 3060 12GB")
    bits, parts = bits_that_fit(facts, card, ctx=32768, runtime="vllm")
    assert 4 < bits < 16
    spec = QuantSpec(method="mixed", variant="auto", bits_per_weight=bits, embedding_bits=16)
    fit = estimate(facts, spec, card, ctx=32768, runtime="vllm")
    assert fit.vram_gb == pytest.approx(parts["usable_memory"] * (1 - _HEADROOM), abs=0.05)
    assert fit.verdict.value in ("fits", "tight")
