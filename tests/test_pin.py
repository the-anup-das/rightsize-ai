"""One framework at a time: pinning a trainer or a server in a plan (F4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightsize.errors import NotImplementedYet
from rightsize.rules.recommend import recommend_for_model
from rightsize.types import ModelFacts

FACTS = ModelFacts.model_validate(json.loads(
    (Path(__file__).parent / "fixtures/facts/Qwen__Qwen3-4B.json").read_text(encoding="utf-8")))


def _plan(device: str = "RTX 4090", **kw):
    return recommend_for_model(FACTS, device, top_k=1, **kw).plans[0]


def _rendered(plan) -> dict[str, str]:
    return {r.recipe_id: r.text for r in plan.render()}


def test_pinning_ollama_serves_with_it_and_sizes_it_with_its_overheads() -> None:
    default, ollama = _plan(), _plan(framework="ollama")
    serve = [s for s in ollama.steps if s.stage == "serve"]
    assert [s.recipe_id for s in serve] == ["ollama/modelfile", "ollama/create"]
    assert serve[0].runtime.name == "ollama"
    assert ollama.steps[0].fit.vram_gb != default.steps[0].fit.vram_gb, "Ollama's own overheads"
    text = _rendered(ollama)
    assert "PARAMETER num_ctx 8192" in text["ollama/modelfile"]
    assert text["ollama/create"] == "ollama create qwen3-4b -f Modelfile"


@pytest.mark.parametrize(("framework", "chain", "converted"), [
    ("trl", ["trl/sft", "trl/merge"], "Qwen__Qwen3-4B-merged"),
    ("axolotl", ["axolotl/qlora", "axolotl/merge"], "Qwen__Qwen3-4B-finetune/merged"),
    ("unsloth", ["unsloth/sft"], "Qwen__Qwen3-4B-merged"),
])
def test_a_pinned_trainer_trains_and_hands_the_converter_a_whole_model(
    framework, chain, converted
) -> None:
    """Trainers that write an adapter get their merge step; the conversion reads what the
    last of them wrote."""
    plan = _plan(finetune_device="RTX 4090", mode="qlora", framework=framework)
    ids = [s.recipe_id for s in plan.steps]
    assert ids[: len(chain)] == chain and ids[len(chain)] == "llama.cpp/convert"
    assert f"convert_hf_to_gguf.py {converted} " in _rendered(plan)["llama.cpp/convert"]


def test_trl_merges_the_adapter_where_its_training_saved_it() -> None:
    text = _rendered(_plan(finetune_device="RTX 4090", mode="lora", framework="trl"))
    assert "--output_dir Qwen__Qwen3-4B-finetune" in text["trl/sft"]
    assert 'from_pretrained("Qwen__Qwen3-4B-finetune"' in text["trl/merge"]
    assert 'save_pretrained("Qwen__Qwen3-4B-merged")' in text["trl/merge"]


def test_on_a_mac_mlx_quantizes_trains_and_fuses_on_its_own_4bit_copy() -> None:
    plan = _plan("M4 Max 64GB", finetune_device="M4 Max 64GB", mode="qlora")
    assert [s.recipe_id for s in plan.steps][:3] == ["mlx-lm/convert", "mlx-lm/lora",
                                                     "mlx-lm/fuse"]
    fuse = _rendered(plan)["mlx-lm/fuse"]
    assert "--model Qwen__Qwen3-4B-mlx-4bit" in fuse and fuse.endswith("--dequantize")


def test_a_framework_that_does_not_run_on_the_device_is_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="MLX LM runs on apple hardware"):
        _plan(finetune_device="RTX 4090", mode="qlora", framework="mlx-lm")


def test_a_trainer_pinned_without_a_fine_tune_says_what_is_missing() -> None:
    with pytest.raises(ValueError, match="--mode lora or qlora"):
        _plan(framework="unsloth")


def test_frameworks_plans_do_not_cover_yet_say_so() -> None:
    with pytest.raises(NotImplementedYet, match="Planning with vLLM"):
        _plan(framework="vllm")


def test_an_unknown_framework_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known:"):
        _plan(framework="nosuch")
