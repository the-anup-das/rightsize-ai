"""End-to-end: convert, imatrix, quantize and gate Qwen3-0.6B with real llama.cpp binaries.

Marked `slow` (downloads ~1.5 GB the first time, several minutes). Skips when the toolchain or
the llamacpp extra is missing. Run:  uv run pytest -m slow tests/integration -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightsize.execution.llamacpp import ToolchainError, find_tools

pytestmark = pytest.mark.slow

REPO = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tools():
    try:
        return find_tools()
    except ToolchainError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def llamacpp_extra():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("torch")


def test_quantize_qwen3_0_6b_end_to_end(tools, llamacpp_extra, tmp_path_factory) -> None:
    from rightsize.execution import quantize_model

    out = tmp_path_factory.mktemp("runs")
    models = Path("models")  # reuse converted files across runs
    m = quantize_model(
        REPO,
        ["Q4_K_M", "Q8_0"],
        imatrix=True,
        evaluate=True,
        eval_chunks=10,
        out_dir=out,
        models_dir=models,
        log=lambda s: None,
    )
    assert m.status == "succeeded"
    assert Path(m.artifacts["Q4_K_M"]).exists() and Path(m.artifacts["Q8_0"]).exists()

    sizes = {
        meas.note: (meas.predicted, meas.value)
        for step in m.steps
        for meas in step.measurements
        if meas.kind == "file_size_gb"
    }
    for q in ("Q4_K_M", "Q8_0"):
        predicted, measured = sizes[q]
        assert measured == pytest.approx(predicted, rel=0.15), f"{q}: {measured} vs {predicted}"

    assert set(m.gate) == {"Q4_K_M", "Q8_0"}
    assert m.gate["Q8_0"]["verdict"] in ("pass", "warn")
    assert m.gate["Q8_0"]["kld_mean"] < m.gate["Q4_K_M"]["kld_mean"]

    manifest = json.loads((out / m.id / "manifest.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["toolchain"]["llama.cpp"]
    assert (out / "measurements.jsonl").read_text().count("\n") >= 4
