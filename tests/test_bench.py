"""Deriving a device's bandwidth from measured throughput. No GPU, no llama.cpp."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightsize.execution import bench
from rightsize.fit import estimate
from rightsize.types import Device, Family, ModelFacts, ModelRef


def _facts(params: int | None = 2_031_000_000) -> ModelFacts:
    return ModelFacts(
        ref=ModelRef(repo="Qwen/Qwen3-1.7B"),
        family=Family.llm,
        params_total=params,
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        dtype="BF16",
    )


def test_throughput_inverts_the_speed_model() -> None:
    """The measurement is the fit engine's own formula solved the other way round.

    Derive a bandwidth from a throughput, hand it back to estimate(), and the predicted
    tok/s must be the throughput we started from. If these ever drift apart, a measured
    device silently stops predicting its own measurement.
    """
    facts = _facts()
    gbps = bench.bandwidth_from_throughput(facts, "Q4_K_M", 362.7, ctx=512)
    dev = Device(name="test card", vendor="nvidia", memory_gib=16, bandwidth_gbps=gbps)
    assert estimate(facts, "Q4_K_M", dev, ctx=512).speed == pytest.approx(362.7, rel=0.01)


def test_derived_bandwidth_matches_this_machines_real_run() -> None:
    """A real RTX 4070 Ti SUPER run: 362.7 tok/s on Qwen3-1.7B Q4_K_M at ctx 512.

    The card is 672 GB/s, and the derivation lands on it, so the decode reached the 70%
    of peak the speed model assumes. That is the constant F9 exists to refit, now with one
    real point behind it.
    """
    gbps = bench.bandwidth_from_throughput(_facts(), "Q4_K_M", 362.7, ctx=512)
    assert gbps == pytest.approx(675, rel=0.02)


def test_unknown_parameter_count_refuses_rather_than_guesses() -> None:
    with pytest.raises(bench.BenchError, match="parameter count"):
        bench.bandwidth_from_throughput(_facts(params=None), "Q4_K_M", 100.0)


def test_llama_bench_json_survives_the_backend_banner(monkeypatch, tmp_path: Path) -> None:
    """llama-bench prints CUDA backend lines before its JSON, so a plain parse fails."""
    rows = [{"n_gen": 128, "avg_ts": 362.7}, {"n_prompt": 512, "avg_ts": 9000.0}]
    out = "ggml_cuda_init: found 1 CUDA devices\nload_backend: loaded CUDA\n" + json.dumps(rows)

    class _Done:
        returncode, stdout, stderr = 0, out, ""

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: _Done())
    got = bench.run_llama_bench(tmp_path / "llama-bench", tmp_path / "m.gguf")
    assert [r["avg_ts"] for r in got] == [362.7], "prompt rows are compute bound, not memory"


def test_failure_carries_what_the_tool_said(monkeypatch, tmp_path: Path) -> None:
    class _Failed:
        returncode, stdout, stderr = 1, "", "error: failed to load model\n"

    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **k: _Failed())
    with pytest.raises(bench.BenchError, match="failed to load model"):
        bench.run_llama_bench(tmp_path / "llama-bench", tmp_path / "m.gguf")


def test_measurement_is_remembered_and_preferred_over_the_table(monkeypatch, tmp_path) -> None:
    """A measurement of your machine beats a published figure about your model of card."""
    monkeypatch.setattr(bench, "CACHE_FILE", tmp_path / "bandwidth.json")
    assert bench.measured_bandwidth("RTX 4090 24GB") is None
    bench._remember("RTX 4090 24GB", {"bandwidth_gbps": 940.0})
    assert bench.measured_bandwidth("RTX 4090 24GB") == 940.0

    from rightsize import hardware

    assert hardware.get("RTX 4090").bandwidth_gbps == 1008.0, "the table, untouched"
    assert hardware.resolve("RTX 4090").bandwidth_gbps == 940.0, "resolve prefers the run"


def test_a_corrupt_cache_is_ignored_not_fatal(monkeypatch, tmp_path: Path) -> None:
    bad = tmp_path / "bandwidth.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(bench, "CACHE_FILE", bad)
    assert bench.measured_bandwidth("anything") is None
