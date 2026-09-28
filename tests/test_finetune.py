"""Fine-tuning memory and quantization quality (F3), against published figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from rightsize.fit import estimate, estimate_finetune, ppl_delta
from rightsize.fit.finetune import lora_params, unsloth_floor
from rightsize.fit.quality import band
from rightsize.types import Device, Family, Mode, ModelFacts, ModelRef, Verdict

FIXTURES = Path(__file__).parent / "fixtures" / "facts"
CARD = Device(name="16 GiB card", memory_gib=16, usable_fraction=0.92)
BIG = Device(name="big", memory_gib=1024)


def _qwen3_4b() -> ModelFacts:
    return ModelFacts.model_validate_json((FIXTURES / "Qwen__Qwen3-4B.json").read_text("utf-8"))


def _sized(params_b: float) -> ModelFacts:
    """A dense model of a given size, shaped like the Llama family."""
    return ModelFacts(
        ref=ModelRef(repo=f"test/{params_b}b"),
        family=Family.llm,
        params_total=int(params_b * 1e9),
        num_layers=80 if params_b > 30 else 32,
        num_kv_heads=8,
        head_dim=128,
        extra={
            "hidden_size": 8192 if params_b > 30 else 4096,
            "vocab_size": 128256,
            "num_attention_heads": 64 if params_b > 30 else 32,
            "intermediate_size": 28672 if params_b > 30 else 14336,
        },
    )


@pytest.mark.parametrize(
    "params_b,qlora,lora",
    [(7, 5, 19), (14, 8.5, 33), (70, 41, 164)],
)
def test_never_below_unsloths_published_minimum(params_b: float, qlora: float, lora: float) -> None:
    """Unsloth's table: 7B needs 5 / 19 GB, 70B 41 / 164 GB for QLoRA / 16-bit LoRA."""
    fx = _sized(params_b)
    assert estimate_finetune(fx, BIG, Mode.qlora).vram_gb >= qlora
    assert estimate_finetune(fx, BIG, Mode.lora).vram_gb >= lora


def test_floor_interpolates_between_rows() -> None:
    """A 4B model sits between the 3B and 7B rows. Rounding up to the 7B row made 16-bit
    LoRA on a 4B model a 19 GB no_fit on a 16 GB card; between the rows it is 10.8 GB."""
    gb, where = unsloth_floor(4_022_468_096, Mode.lora)
    assert gb == pytest.approx(8 + (19 - 8) * (4.022 - 3) / 4, abs=0.05)
    assert "3B and 7B" in where
    r = estimate_finetune(_qwen3_4b(), CARD, Mode.lora)
    assert r.verdict is Verdict.fits and r.vram_gb == pytest.approx(10.81, abs=0.1)


def test_no_floor_outside_the_table_or_for_full_finetuning() -> None:
    assert unsloth_floor(int(1e12), Mode.qlora) == (None, None)
    assert unsloth_floor(int(8e9), Mode.full) == (None, None)


def test_components_track_the_published_figure() -> None:
    """The breakdown should explain Unsloth's number, not merely sit under it: for Qwen3-4B
    QLoRA it lands within a gigabyte of the interpolated minimum."""
    r = estimate_finetune(_qwen3_4b(), CARD, Mode.qlora)
    parts = sum(r.breakdown[k] for k in ("weights", "trainable_state", "activations", "overhead"))
    assert abs(parts - r.breakdown["published_minimum"]) < 1.0


def test_full_finetuning_is_sixteen_bytes_a_parameter() -> None:
    fx = _qwen3_4b()
    r = estimate_finetune(fx, BIG, Mode.full)
    assert r.breakdown["weights"] + r.breakdown["trainable_state"] == pytest.approx(
        fx.params_total * 16 / 1e9, rel=1e-3
    )


def test_lora_adapter_count_matches_the_standard_formula() -> None:
    """Qwen3-4B, rank 16 on q, k, v, o, gate, up, down: r * (d_in + d_out) per layer."""
    fx = _qwen3_4b()
    h, q, kv, inter = 2560, 32 * 128, 8 * 128, 9728
    per_layer = (h + q) + 2 * (h + kv) + (q + h) + 2 * (h + inter) + (inter + h)
    assert lora_params(fx, 16) == 16 * per_layer * 36


def test_estimate_dispatches_training_modes() -> None:
    r = estimate(_qwen3_4b(), "Q4_K_M", CARD, mode=Mode.qlora)
    assert r.formula_id == "llm.finetune.components.v1"


def test_quantization_quality_from_llama_cpp() -> None:
    assert ppl_delta("Q4_K_M") == (0.1754, "measured by llama.cpp on Llama-3-8B")
    assert band(ppl_delta("Q4_K_S")[0]) == "good"
    assert band(ppl_delta("Q2_K")[0]) == "damaged"


def test_i_quants_are_an_upper_bound_between_their_neighbours() -> None:
    """IQ4_XS is 4.46 effective bits, between Q3_K_L and Q4_K_S. Comparing its nominal 4.25
    with the K-quants' effective bits once put it below Q3_K_L."""
    iq4, how = ppl_delta("IQ4_XS")
    assert ppl_delta("Q4_K_S")[0] < iq4 < ppl_delta("Q3_K_L")[0]
    assert how.startswith("at most this")


def test_below_every_measured_type_is_not_guessed() -> None:
    delta, how = ppl_delta("IQ1_S")
    assert delta is None and "too far out" in how
    assert band(delta) == "unknown"


@pytest.mark.parametrize(
    "name, allocated_gb",
    [
        ("Qwen__Qwen3-0.6B", 1.798),
        ("Qwen__Qwen3-1.7B", 2.758),
    ],
)
def test_qlora_tensors_match_what_unsloth_allocated(name: str, allocated_gb: float) -> None:
    """Unsloth 2026.9.11 QLoRA, batch 2, 2048 tokens, 30 steps on an RTX 4070 Ti SUPER
    (2026-09-28): torch.cuda.max_memory_allocated() for the training step alone. Without the
    logits the parts came to 0.86 and 1.99 GB."""
    import json

    path = Path(__file__).parent / "fixtures" / "facts" / f"{name}.json"
    facts = ModelFacts.model_validate(json.loads(path.read_text(encoding="utf-8")))
    b = estimate_finetune(facts, CARD, Mode.qlora, batch=2).breakdown
    tensors = b["weights"] + b["trainable_state"] + b["activations"] + b["logits"]
    assert tensors == pytest.approx(allocated_gb, rel=0.05)
