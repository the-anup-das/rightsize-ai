"""The orchestrator's dry run, with the catalog and toolchain faked: no network, no binaries."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rightsize.execution import quantize as q
from rightsize.execution.llamacpp import LlamaCppTools
from rightsize.types import Device, Family, ModelFacts, ModelRef


@pytest.fixture
def fake_env(monkeypatch, tmp_path: Path):
    fx = ModelFacts(
        ref=ModelRef(repo="Qwen/Qwen3-0.6B"),
        family=Family.llm,
        params_total=751_632_384,
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        context_max=40960,
        dtype="BF16",
    )
    monkeypatch.setattr(q, "hub_facts", lambda repo, revision, token=None: fx)
    root = tmp_path / "llama.cpp"
    root.mkdir()
    exe = ".exe" if sys.platform == "win32" else ""
    tools = LlamaCppTools(
        root=root,
        quantize=root / f"llama-quantize{exe}",
        imatrix=root / f"llama-imatrix{exe}",
        perplexity=root / f"llama-perplexity{exe}",
        convert_script=root / "convert_hf_to_gguf.py",
        version="test",
    )
    monkeypatch.setattr(q, "find_tools", lambda explicit=None: tools)
    dev = Device(name="RTX 4070 Ti SUPER 16GB", vendor="nvidia", memory_gib=16, bandwidth_gbps=672)
    return fx, tools, dev, tmp_path


def test_dry_run_renders_every_step_and_writes_manifest(fake_env) -> None:
    fx, tools, dev, tmp = fake_env
    m = q.quantize_model(
        "Qwen/Qwen3-0.6B",
        ["Q4_K_M", "IQ4_XS"],
        device=dev,
        imatrix=False,
        evaluate=True,
        out_dir=tmp / "runs",
        models_dir=tmp / "models",
        dry_run=True,
        log=lambda s: None,
    )
    assert m.status == "succeeded"
    ids = [s.recipe_id for s in m.steps]
    # IQ4_XS forces an imatrix step even though imatrix=False
    assert ids == [
        "llama.cpp/convert",
        "llama.cpp/imatrix",
        "llama.cpp/quantize",
        "llama.cpp/quantize",
    ]
    assert all(s.skipped for s in m.steps)
    assert set(m.predicted) == {"Q4_K_M", "IQ4_XS"}
    assert m.predicted["Q4_K_M"].verdict.value == "fits"
    argv = m.steps[2].argv
    assert argv[0] == str(tools.quantize) and argv[-1] == "Q4_K_M" and "--imatrix" in argv
    assert (tmp / "runs" / m.id / "manifest.json").exists()
    assert not (tmp / "runs" / "measurements.jsonl").exists(), "dry runs record nothing"


def test_dry_run_predictions_are_in_manifest_json(fake_env) -> None:
    fx, tools, dev, tmp = fake_env
    m = q.quantize_model(
        "Qwen/Qwen3-0.6B",
        ["Q8_0"],
        device=dev,
        out_dir=tmp / "runs",
        models_dir=tmp / "models",
        dry_run=True,
        log=lambda s: None,
    )
    data = (tmp / "runs" / m.id / "manifest.json").read_text()
    assert '"formula_id": "llm.gguf.analytic.v0"' in data and '"Q8_0"' in data


def test_logits_file_size_matches_a_real_run() -> None:
    """Golden: a Qwen3-1.7B reference pass (vocab 151936, ctx 512) had written
    3,719,692,288 bytes after 48 chunks on this machine. The prediction must land on it,
    because 100 chunks means ~8 GB of disk and people deserve the warning."""
    facts = SimpleNamespace(extra={"vocab_size": 151936})
    assert q._logits_gb(facts, 48) == pytest.approx(3.7195, rel=0.001)
    assert q._logits_gb(facts, 100) == pytest.approx(7.749, rel=0.001)
    # no vocab in facts: fall back to a small vocab rather than crashing
    assert q._logits_gb(SimpleNamespace(extra={}), 100) > 0
