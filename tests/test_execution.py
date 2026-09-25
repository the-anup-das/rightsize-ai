"""Execution adapter pieces that run without llama.cpp: parsing, gate, tool discovery, run_step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rightsize.execution.llamacpp import (
    ToolchainError,
    find_tools,
    gate,
    parse_perplexity_output,
    run_step,
)
from rightsize.registry.schema import RenderedStep

SAMPLE = """
[1]4.9797,[2]5.8801,[3]6.2345
Final estimate: PPL = 6.2345 +/- 0.03456

====== KL divergence statistics ======
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   1.234567
99.9%   KLD:   0.456789
Mean    ln(PPL(Q)/PPL(base)):   0.004567 ±   0.001234
====== Token probability statistics ======
Same top p: 96.123 ±  0.05 %
"""


def test_parse_perplexity_output() -> None:
    m = parse_perplexity_output(SAMPLE)
    assert m["ppl"] == pytest.approx(6.2345)
    assert m["kld_mean"] == pytest.approx(0.012345)
    assert m["top1_agreement"] == pytest.approx(0.96123)
    assert m["ln_ppl_ratio"] == pytest.approx(0.004567)


@pytest.mark.parametrize(
    "kld,top1,expected",
    [
        (0.01, 0.96, "pass"),
        (0.08, 0.96, "warn"),
        (0.30, 0.96, "fail"),
        (0.01, 0.75, "fail"),
        (0.01, 0.85, "warn"),
    ],
)
def test_gate_thresholds(kld: float, top1: float, expected: str) -> None:
    assert gate({"kld_mean": kld, "top1_agreement": top1})["verdict"] == expected


def test_gate_without_metrics_is_unknown() -> None:
    assert gate({})["verdict"] == "unknown"


def test_find_tools_explicit_dir(tmp_path: Path) -> None:
    exe = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
    (tmp_path / exe).write_bytes(b"")
    (tmp_path / "VERSION").write_text("b11177")
    t = find_tools(tmp_path)
    assert t.version == "b11177" and t.quantize.name == exe and t.convert_script is None


def test_find_tools_missing_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RIGHTSIZE_LLAMA_CPP", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rightsize.execution.llamacpp.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "rightsize.execution.llamacpp.Path.resolve", lambda self: tmp_path / "a" / "b" / "c" / "d"
    )
    with pytest.raises(ToolchainError):
        find_tools()


def test_run_step_streams_and_measures(tmp_path: Path) -> None:
    step = RenderedStep(
        recipe_id="test/echo",
        framework="test",
        stage="evaluate",
        kind="command",
        text="",
        argv=[sys.executable, "-c", "print('hello'); print('Final estimate: PPL = 3.5 +/- 0.1')"],
        source_doc_url="https://example.invalid",
    )
    lines: list[str] = []
    rs, text, peak = run_step(step, log_path=tmp_path / "log.txt", log=lines.append)
    assert rs.returncode == 0 and rs.finished
    assert "hello" in lines and "PPL = 3.5" in text
    assert rs.measurements[0].kind == "wall_s"
    assert (tmp_path / "log.txt").read_text().startswith("$ ")
