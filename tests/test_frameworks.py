"""Framework descriptors: a new SDK is a folder of data, not a change to rightsize (F5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rightsize.registry import all_recipes, framework_infos, loader, trainer_for
from rightsize.types import ModelFacts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh():
    loader.clear_cache()
    yield
    loader.clear_cache()


def test_every_framework_with_recipes_has_a_descriptor_that_matches_them() -> None:
    infos = framework_infos()
    for folder in sorted((ROOT / "data" / "recipes").iterdir()):
        if folder.is_dir():
            assert (folder / "framework.yaml").exists(), f"{folder.name} has no framework.yaml"
            assert infos[folder.name].name == folder.name
    for recipe in all_recipes().values():
        assert recipe.stage in infos[recipe.framework].stages, recipe.id


def test_every_default_is_an_input_one_of_the_frameworks_recipes_reads() -> None:
    """A misspelt default would silently fill nothing."""
    for info in framework_infos().values():
        inputs = {n for r in all_recipes().values() if r.framework == info.name for n in r.inputs}
        assert set(info.defaults) <= inputs, (info.name, set(info.defaults) - inputs)


def test_trainer_roles_name_real_recipes_of_the_right_stage() -> None:
    recipes = all_recipes()
    for info in framework_infos().values():
        role = info.finetune
        if role is None:
            continue
        trains = recipes[role.recipe]
        assert trains.stage == "finetune" and trains.framework == info.name
        for before in role.before.values():
            assert recipes[before.recipe].stage == before.stage


@pytest.mark.parametrize(("vendor", "trainer"), [
    ("nvidia", "unsloth"), ("intel", "unsloth"), ("apple", "mlx-lm"), ("amd", "axolotl"),
    ("cpu", "unsloth"), ("qualcomm", "unsloth"),
])
def test_the_default_trainer_per_vendor_comes_from_the_descriptors(vendor, trainer) -> None:
    assert trainer_for(vendor).name == trainer


# ---------------------------------------------------------------- adding an SDK


ZOOMTUNE = {
    "framework.yaml": """
name: zoomtune
title: ZoomTune
summary: a trainer someone published as a rightsize plugin
stages: [finetune]
families: [llm]
hardware: {vendors: [nvidia]}
install: {kind: pip, packages: [zoomtune], check: zoomtune, line: pip install zoomtune}
finetune:
  modes: [lora, qlora]
  recipe: zoomtune/train
  qlora_quant: {method: bnb, variant: nf4}
  writes: merged
  default_for: [nvidia]
  priority: 20
defaults: {epochs: "3"}
homepage: https://example.org/zoomtune
source_doc_url: https://example.org/zoomtune/docs
""",
    "train.yaml": """
id: zoomtune/train
framework: zoomtune
stage: finetune
families: [llm]
inputs:
  model: {type: str, required: true}
  load_in_4bit: {type: bool, default: true}
  merged_dir: {type: path, default: merged}
  epochs: {type: int, default: 1}
template: >-
  zoomtune train {{ model }} --4bit {{ load_in_4bit }} --epochs {{ epochs }}
  --out {{ merged_dir }}
source_doc_url: https://example.org/zoomtune/docs
""",
}


def _plugin(tmp_path: Path, files: dict[str, str], name: str = "zoomtune") -> Path:
    root = tmp_path / "plugin"
    (root / name).mkdir(parents=True)
    for fname, text in files.items():
        (root / name / fname).write_text(text.lstrip(), encoding="utf-8")
    return root


def _facts() -> ModelFacts:
    raw = (ROOT / "tests/fixtures/facts/Qwen__Qwen3-4B.json").read_text(encoding="utf-8")
    return ModelFacts.model_validate(json.loads(raw))


def test_a_plugin_trainer_takes_over_without_touching_rightsize(tmp_path, monkeypatch, fresh):
    """A package that ships a framework.yaml and a recipe becomes the default trainer on the
    hardware it claims, and plans render its command, with no change to rightsize itself."""
    from rightsize.rules.recommend import recommend_for_model

    root = _plugin(tmp_path, ZOOMTUNE)
    monkeypatch.setattr(loader, "_plugin_targets", lambda: [("zoomtune", root)])
    assert trainer_for("nvidia").name == "zoomtune"
    assert trainer_for("apple").name == "mlx-lm", "other vendors keep their defaults"

    result = recommend_for_model(_facts(), "RTX 4090", finetune_device="RTX 4090",
                                 mode="qlora", top_k=1)
    plan = result.plans[0]
    ft = next(s for s in plan.steps if s.stage == "finetune")
    assert (ft.framework, ft.recipe_id, ft.quant.variant) == ("zoomtune", "zoomtune/train", "nf4")
    rendered = {r.recipe_id: r for r in plan.render()}
    text = rendered["zoomtune/train"].text
    assert "zoomtune train Qwen/Qwen3-4B --4bit true --epochs 3" in text, "its own default"
    assert "Qwen__Qwen3-4B-merged" in text, "writes where the llama.cpp steps read next"


def test_a_plugin_cannot_replace_a_bundled_framework(tmp_path, monkeypatch, fresh) -> None:
    clash = {"framework.yaml": ZOOMTUNE["framework.yaml"].replace("name: zoomtune",
                                                                  "name: unsloth")}
    root = _plugin(tmp_path, clash, name="unsloth2")
    monkeypatch.setattr(loader, "_plugin_targets", lambda: [("evil", root)])
    with pytest.raises(ValueError, match="duplicate framework unsloth"):
        framework_infos()


def test_recipes_without_a_descriptor_still_list(tmp_path, monkeypatch, fresh) -> None:
    """An older plugin with recipes only gets a descriptor made from them."""
    root = _plugin(tmp_path, {"train.yaml": ZOOMTUNE["train.yaml"]})
    monkeypatch.setattr(loader, "_plugin_targets", lambda: [("zoomtune", root)])
    info = framework_infos()["zoomtune"]
    assert info.stages == ["finetune"] and info.finetune is None
    assert trainer_for("nvidia").name == "unsloth"


def test_the_descriptors_are_data_files_like_any_other() -> None:
    from rightsize.data_models import model_for

    name, model = model_for("recipes/unsloth/framework.yaml")
    assert name == "framework"
    doc = yaml.safe_load((ROOT / "data/recipes/unsloth/framework.yaml").read_text(encoding="utf-8"))
    assert model.model_validate(doc).finetune.recipe == "unsloth/sft"


def test_mlx_qlora_records_mlx_4bit_not_bitsandbytes() -> None:
    """MLX trains QLoRA on a model it quantized itself; the plan says so on both steps."""
    from rightsize.rules.recommend import recommend_for_model

    plan = recommend_for_model(_facts(), "M4 Max 64GB", finetune_device="M4 Max 64GB",
                               mode="qlora", top_k=1).plans[0]
    first, ft = plan.steps[0], plan.steps[1]
    assert (first.recipe_id, first.quant.method) == ("mlx-lm/convert", "mlx")
    assert (ft.recipe_id, ft.quant.method, ft.quant.variant) == ("mlx-lm/lora", "mlx", "4bit")
