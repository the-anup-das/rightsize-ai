"""rightsize quantize --to: any format a toolkit's recipe declares (F8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rightsize.execution import quantize_to as qt
from rightsize.hardware import resolve
from rightsize.registry import all_recipes, loader
from rightsize.registry.schema import Target
from rightsize.types import ModelFacts

ROOT = Path(__file__).resolve().parents[1]
FACTS = ModelFacts.model_validate(
    json.loads((ROOT / "tests/fixtures/facts/Qwen__Qwen3-4B.json").read_text(encoding="utf-8"))
)


def test_every_target_names_real_inputs_and_a_known_size() -> None:
    formats = {
        f["id"]
        for f in yaml.safe_load((ROOT / "data/quants/formats.yaml").read_text(encoding="utf-8"))[
            "formats"
        ]
    }
    recipes = all_recipes()
    for recipe in recipes.values():
        for target in recipe.targets:
            assert set(target.inputs) <= set(recipe.inputs), (recipe.id, target.name)
            assert target.size_from in formats, (recipe.id, target.size_from)
            assert recipe.writes, f"{recipe.id} must say what it writes to be a target"
            if target.gate is None:
                continue
            check = recipes[target.gate.recipe]  # a gate names a recipe that exists
            assert check.stage == "evaluate", (target.name, check.id)
            assert set(target.gate.inputs) <= set(check.inputs), (target.name, check.id)
            # what quantize_to hands every gate, and where record_gate reads the numbers
            assert {"model", "candidate", "eval_file"} <= set(check.inputs), check.id
            assert check.writes == ["gate_file"], check.id


def test_the_gated_formats() -> None:
    gated = {t.name for r in all_recipes().values() for t in r.targets if t.gate}
    assert gated == {"fp8", "w4a16", "awq", "nf4", "openvino-int4", "openvino-int8", "onnx-int8"}


def test_a_target_without_a_gate_says_so(monkeypatch) -> None:
    import rightsize.catalog as catalog

    monkeypatch.setattr(catalog, "facts", lambda *a, **k: FACTS)
    with pytest.raises(ValueError, match="no quality gate yet"):
        qt.plan_for("Qwen/Qwen3-4B", "ct2-int8", resolve("RTX 4090"), evaluate=True)


def test_the_gate_judges_embeddings_by_cosine() -> None:
    from rightsize.execution.llamacpp import gate

    assert gate({"cosine_mean": 0.995})["verdict"] == "pass"
    assert gate({"cosine_mean": 0.98})["verdict"] == "warn"
    assert gate({"cosine_mean": 0.9, "cosine_min": 0.5})["verdict"] == "fail"


def test_the_formats_on_offer() -> None:
    assert {
        "fp8",
        "w4a16",
        "awq",
        "nf4",
        "openvino-int4",
        "openvino-int8",
        "mlx-4bit",
        "mlx-8bit",
        "onnx-int8",
        "ct2-int8",
    } <= set(qt.targets())


@pytest.mark.parametrize("target", sorted(qt.targets()))
def test_every_target_plans_and_renders(target, monkeypatch, tmp_path) -> None:
    import rightsize.catalog as catalog

    monkeypatch.setattr(catalog, "facts", lambda *a, **k: FACTS)
    recipe, spec = qt.choose(target)
    manifest = qt.quantize_to(
        "Qwen/Qwen3-4B",
        target,
        resolve("RTX 4090"),
        dry_run=True,
        evaluate=spec.gate is not None,
        workdir=str(tmp_path),
        log=lambda _line: None,
    )
    assert len(manifest.steps) == (2 if spec.gate else 1)
    for step in manifest.steps:
        # a config recipe has no interpreter to name where its toolkit is not installed (CI),
        # so the dry run promises only a skipped step that says so
        assert step.skipped and step.measurements[0].note.startswith("dry run")
    if spec.gate:
        # the gate runs where the candidate loads: the quantizer's own environment
        plan = qt.plan_for("Qwen/Qwen3-4B", target, resolve("RTX 4090"), evaluate=True)
        assert plan.steps[1].stage == "evaluate"
        assert plan.steps[1].framework == recipe.framework
        assert plan.steps[1].recipe_id == spec.gate.recipe


def test_a_format_nobody_produces_lists_the_ones_that_exist() -> None:
    with pytest.raises(KeyError, match="formats: gguf, "):
        qt.choose("exl3")
    with pytest.raises(KeyError, match="llm-compressor does"):
        qt.choose("fp8", framework="unsloth")


def test_the_plan_predicts_the_size_from_the_formats_table(monkeypatch) -> None:
    import rightsize.catalog as catalog

    monkeypatch.setattr(catalog, "facts", lambda *a, **k: FACTS)
    plan = qt.plan_for("Qwen/Qwen3-4B", "w4a16", resolve("RTX 4090"))
    (step,) = plan.steps
    assert (step.recipe_id, step.framework) == ("llm-compressor/gptq-w4a16", "llm-compressor")
    assert step.quant.bits_per_weight == pytest.approx(4.156)
    table = 151936 * 2560  # the embedding stays 16-bit; the tied head is not saved again
    body = FACTS.params_total - table
    assert step.fit.breakdown["file_gb"] == pytest.approx(
        (body * 4.156 + table * 16) / 8 / 1e9, abs=0.001
    )


def _facts(**extra) -> ModelFacts:
    return FACTS.model_copy(
        update={
            "params_total": extra.pop("params", FACTS.params_total),
            "extra": {**FACTS.extra, **extra},
        }
    )


def test_the_size_counts_the_embedding_and_head_apart() -> None:
    """Measured on Qwen3-0.6B (2026-09-28): llm-compressor's FP8 wrote 752.4 MB of weights
    and its W4A16 538.5 MB. Its checkpoint stores the tied head a second time, and the
    embedding stays 16-bit; params x bits said 0.752 and 0.390 GB."""
    linear_only = Target(name="t", embedding_bits=16, head_bits=16)
    qwen06 = _facts(
        params=751_632_384,
        vocab_size=151936,
        hidden_size=1024,
        tie_word_embeddings=True,
        tied_head_stored=155_582_464,
    )
    assert qt.output_gb(qwen06, 8.0, linear_only) == pytest.approx(0.752, abs=0.001)
    assert qt.output_gb(qwen06, 4.156, linear_only) == pytest.approx(0.540, abs=0.001)
    # the same bits throughout: only the stored second copy of the head drops out
    assert qt.output_gb(qwen06, 4.0, Target(name="t")) == pytest.approx(
        (751_632_384 - 155_582_464) * 4 / 8 / 1e9
    )
    # a head of its own is counted once, at its own bits
    untied = _facts(
        params=1_000_000_000,
        vocab_size=100_000,
        hidden_size=1000,
        tie_word_embeddings=False,
        tied_head_stored=None,
    )
    assert qt.output_gb(untied, 4.0, linear_only) == pytest.approx(
        (800_000_000 * 4 + 200_000_000 * 16) / 8 / 1e9
    )
    # no vocabulary facts: every parameter at the format's bits
    bare = _facts(params=1_000_000_000, vocab_size=None, hidden_size=None, tied_head_stored=None)
    assert qt.output_gb(bare, 4.0, linear_only) == pytest.approx(0.5)


PLUGIN = {
    "framework.yaml": """
name: squash
title: Squash
summary: makes a file of the size the format promises, for testing
stages: [quantize, evaluate]
install: {kind: pip, packages: [squash], check: json, line: pip install squash}
homepage: https://example.org/squash
source_doc_url: https://example.org/squash
""",
    "int8.yaml": """
id: squash/int8
framework: squash
stage: quantize
kind: config
language: python
inputs:
  model: {type: str, required: true}
  out: {type: path, default: out.bin}
  nbytes: {type: int, default: 1000}
writes: [out]
targets:
  - {name: squash-int8, size_from: int8, gate: {recipe: squash/check, inputs: {kld: 0.02}}}
template: |
  open("{{ out }}", "wb").write(b"0" * {{ nbytes }})
source_doc_url: https://example.org/squash
""",
    "check.yaml": """
id: squash/check
framework: squash
stage: evaluate
kind: config
language: python
inputs:
  model: {type: str, required: true}
  candidate: {type: path, required: true}
  eval_file: {type: path, required: true}
  kld: {type: float, default: 0.5}
  gate_file: {type: path, default: gate.json}
writes: [gate_file]
template: |
  import json, os
  assert os.path.exists(r"{{ candidate }}") and os.path.exists(r"{{ eval_file }}")
  json.dump({"kld_mean": {{ kld }}, "top1_agreement": 0.95, "ppl": 9.0, "tokens": 10},
            open(r"{{ gate_file }}", "w"))
source_doc_url: https://example.org/squash
""",
}


def _plugin(tmp_path, monkeypatch) -> None:
    import rightsize.catalog as catalog

    root = tmp_path / "plugin" / "squash"
    root.mkdir(parents=True)
    for name, text in PLUGIN.items():
        (root / name).write_text(text.lstrip(), encoding="utf-8")
    monkeypatch.setattr(loader, "_plugin_targets", lambda: [("squash", root.parent)])
    monkeypatch.setattr(catalog, "facts", lambda *a, **k: FACTS)
    # the evaluation text is downloaded once into the tools folder; here it is already there
    texts = tmp_path / "texts" / "wikitext-2-raw"
    texts.mkdir(parents=True)
    (texts / "wiki.test.raw").write_text("a line of text\n", encoding="utf-8")
    monkeypatch.setenv("RIGHTSIZE_TEXTS_DIR", str(texts.parent))
    loader.clear_cache()


def test_a_plugin_gate_runs_after_its_format_and_judges_it(tmp_path, monkeypatch) -> None:
    _plugin(tmp_path, monkeypatch)
    try:
        manifest = qt.quantize_to(
            "Qwen/Qwen3-4B",
            "squash-int8",
            resolve("RTX 4090"),
            workdir=str(tmp_path / "run"),
            evaluate=True,
            log=lambda _line: None,
        )
    finally:
        loader.clear_cache()
    assert [s.recipe_id for s in manifest.steps] == ["squash/int8", "squash/check"]
    verdict = manifest.gate["squash-int8"]
    assert verdict["verdict"] == "pass" and verdict["kld_mean"] == 0.02  # the target's input
    kinds = {m.kind for m in manifest.steps[1].measurements}
    assert {"kld_mean", "top1_agreement", "ppl"} <= kinds
    written = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert written["gate"]["squash-int8"]["verdict"] == "pass", "the manifest on disk has it too"


def test_a_plugin_format_runs_and_is_measured_against_the_prediction(tmp_path, monkeypatch):
    _plugin(tmp_path, monkeypatch)
    try:
        manifest = qt.quantize_to(
            "Qwen/Qwen3-4B",
            "squash-int8",
            resolve("RTX 4090"),
            workdir=str(tmp_path / "run"),
            inputs={"nbytes": 4000},
            log=lambda _line: None,
        )
    finally:
        loader.clear_cache()
    assert (tmp_path / "run" / "Qwen__Qwen3-4B-squash-int8").stat().st_size == 4000
    size = next(m for m in manifest.steps[0].measurements if m.kind == "file_size_gb")
    assert size.predicted == pytest.approx(FACTS.params_total * 8 / 8 / 1e9, abs=0.001)
