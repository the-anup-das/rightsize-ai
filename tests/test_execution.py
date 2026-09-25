"""Execution adapter pieces that run without llama.cpp: parsing, gate, tool discovery, run_step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rightsize.execution import llamacpp
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


# ---------------------------------------------------------------- GPU preflight

_SMI_MEM = ["3887, 16376"]
_SMI_APPS = [
    r"C:\Windows\explorer.exe, [N/A]",
    r"C:\Users\me\.lmstudio\backends\llama-server.exe, [N/A]",
    "/usr/bin/ollama, 8192",
]


def test_gpu_memory_parses_free_and_total(monkeypatch) -> None:
    monkeypatch.setattr(llamacpp, "_smi", lambda q: _SMI_MEM)
    assert llamacpp.gpu_memory() == (4.08, 17.17)
    monkeypatch.setattr(llamacpp, "_smi", lambda q: [])
    assert llamacpp.gpu_memory() is None


def test_gpu_holders_keeps_runtimes_and_handles_missing_sizes(monkeypatch) -> None:
    monkeypatch.setattr(llamacpp, "_smi", lambda q: _SMI_APPS)
    holders = llamacpp.gpu_holders()
    assert holders == ["llama-server", "ollama (8.6 GB)"], "explorer.exe is not worth reporting"


def test_preflight_warns_and_names_the_holder(monkeypatch) -> None:
    monkeypatch.setattr(llamacpp, "_smi", lambda q: _SMI_MEM if "gpu=" in q else _SMI_APPS)
    said: list[str] = []
    assert llamacpp.preflight_vram(5.0, what="the reference pass", log=said.append) is False
    assert "only 4.1 GB free" in said[0] and "needs about 5.0 GB" in said[0]
    assert "llama-server" in said[1]
    said.clear()
    assert llamacpp.preflight_vram(2.0, what="imatrix", log=said.append) is True
    assert said and "4.1 GB free" in said[0]


def test_preflight_is_quiet_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(llamacpp, "_smi", lambda q: [])
    said: list[str] = []
    assert llamacpp.preflight_vram(99.0, what="anything", log=said.append) is True
    assert said == []


def test_log_tail_returns_the_last_words_of_a_crash(tmp_path: Path) -> None:
    p = tmp_path / "step.log"
    chunks = "".join(f"[{i}]5.0,\n" for i in range(20))
    p.write_text("start\n" + chunks + "CUDA error: out of memory")
    tail = llamacpp.log_tail(p, lines=2)
    assert tail.endswith("CUDA error: out of memory") and "[19]" in tail
    assert llamacpp.log_tail(tmp_path / "missing.log") == ""


#: Exactly what llama-perplexity b11177 prints, from the format strings in
#: tools/perplexity/perplexity.cpp: "Mean    KLD: %10.6lf ± %10.6lf" (line 1949),
#: "Mean ln(PPL(Q)/PPL(base))     : %10.6lf ± %10.6lf" (1934) with its padding before the
#: colon, "Same top p: %6.3lf ± %5.3lf %%" (2005) as a percentage, and the per-chunk table
#: header (1858) that must not be mistaken for the summary.
_CHUNK_HEADER = (
    "chunk             PPL               ln(PPL(Q)/PPL(base))"
    "          KL Divergence              Δp RMS            Same top p"
)
_CHUNK_ROW = (
    "    1      15.7006 ±    0.1234     0.004567 ±  0.001234"
    "      0.012345 ±  0.000123    1.234 ±  0.045 %    96.123 ±  0.050 %"
)
REAL_B11177 = f"""
{_CHUNK_HEADER}
{_CHUNK_ROW}
Final estimate: PPL = 15.8871 +/- 0.24680

====== Perplexity statistics ======
Mean  PPL(Q)                   :  15.887100 ±   0.246800
Mean ln(PPL(Q)/PPL(base))     :   0.011872 ±   0.000456
Cor(ln(PPL(Q)), ln(PPL(base))):  99.87%

====== KL divergence statistics ======
Mean    KLD:   0.009876 ±   0.000234
Maximum KLD:   2.345678
99.9%   KLD:   0.456789
Median  KLD:   0.003210
Minimum KLD:   0.000001

====== Token probability statistics ======
Same top p: 97.410 ± 0.048 %
"""


def test_parses_the_real_b11177_output_format() -> None:
    m = parse_perplexity_output(REAL_B11177)
    assert m["ppl"] == 15.8871
    assert m["kld_mean"] == 0.009876, "Mean and KLD are separated by padding, not one space"
    assert m["top1_agreement"] == 0.9741, "printed as a percentage, the gate wants a fraction"
    assert m["ln_ppl_ratio"] == 0.011872, "the label is padded before its colon"
    assert gate(m)["verdict"] == "pass"
