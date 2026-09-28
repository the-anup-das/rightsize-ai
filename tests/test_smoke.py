"""Placeholder-release smoke tests: import, version, CLI, Plan JSON round-trip."""

from __future__ import annotations

import json
import subprocess
import sys

import rightsize
from rightsize import (
    Device,
    Family,
    FitResult,
    Mode,
    ModelFacts,
    ModelRef,
    Plan,
    PlanStep,
    Verdict,
)
from rightsize.cli import main


def test_version() -> None:
    assert rightsize.__version__ == "0.0.1"


def test_cli_version_subprocess() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "rightsize.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == f"rightsize {rightsize.__version__}"


def test_cli_recommend_ranks_plans(capsys) -> None:
    assert main(["--no-color", "recommend", "--device", "RTX 4090", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "RTX 4090" in out and "vram GB" in out


def test_cli_json_flag(capsys) -> None:
    assert main(["--json", "recommend", "--device", "RTX 4090", "--top", "2"]) == 0
    plans = json.loads(capsys.readouterr().out)
    assert len(plans) == 2 and plans[0]["rank"] == 1
    assert plans[0]["steps"][-1]["stage"] == "serve"


def test_a_saved_plan_renders_from_the_cli(tmp_path, capsys) -> None:
    """recommend --json writes plans; plan render turns one back into commands, with any
    recipe input overridden."""
    assert main(["--json", "recommend", "--device", "RTX 4090", "--top", "2"]) == 0
    saved = tmp_path / "plans.json"
    saved.write_text(capsys.readouterr().out, encoding="utf-8")
    assert (
        main(
            [
                "--json",
                "plan",
                "render",
                str(saved),
                "--rank",
                "2",
                "--set",
                "quantize_bin=/opt/llama/llama-quantize",
            ]
        )
        == 0
    )
    steps = json.loads(capsys.readouterr().out)
    assert any(s["argv"] and s["argv"][0] == "/opt/llama/llama-quantize" for s in steps)
    assert main(["plan", "render", str(saved), "--rank", "9"]) == 1


def test_public_recommend_returns_plans() -> None:
    plans = rightsize.recommend("chat", "RTX 4090", top_k=2)
    assert [p.rank for p in plans] == [1, 2]
    assert isinstance(plans[0], rightsize.Plan)


def _sample_plan() -> Plan:
    device = Device(
        name="RTX 4090", vendor="nvidia", memory_gib=24, bandwidth_gbps=1008, os="linux"
    )
    facts = ModelFacts(
        ref=ModelRef(repo="Qwen/Qwen3-14B"),
        family=Family.llm,
        params_total=14_800_000_000,
        num_layers=40,
        num_kv_heads=8,
        head_dim=128,
    )
    fit = FitResult(
        verdict=Verdict.fits,
        vram_gb=10.2,
        breakdown={"weights": 8.9, "kv_cache": 0.8, "overhead": 0.5},
        speed=42.0,
        speed_unit="tok/s",
        confidence=0.8,
        formula_id="llm.analytic.v0",
    )
    step = PlanStep(stage="serve", framework="llama.cpp", device=device, fit=fit)
    return Plan(rank=1, model=facts, mode=Mode.infer, steps=[step], score=0.91)


def test_plan_json_round_trip() -> None:
    plan = _sample_plan()
    restored = Plan.model_validate_json(plan.to_json())
    assert restored == plan


def test_plan_schema_is_exportable() -> None:
    schema = json.loads(Plan.schema_json())
    assert schema["title"] == "Plan"
    assert "steps" in schema["properties"]


def test_top_level_estimate_and_detect_are_real() -> None:
    """These raised NotImplementedYet long after the CLI had working versions, so anyone
    using the Python API got stubs. Offline: facts and device passed in directly."""
    from rightsize.types import Family, ModelFacts, ModelRef

    fx = ModelFacts(
        ref=ModelRef(repo="Qwen/Qwen3-4B"),
        family=Family.llm,
        params_total=4_022_468_096,
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
    )
    r = rightsize.estimate(fx, "q4_k_m", "RTX 4090")
    assert r.verdict.value == "fits" and r.speed and r.formula_id
    slow = rightsize.estimate(fx, "Q4_K_M", "Jetson Orin Nano 8GB")
    assert slow.speed is None, "no bandwidth for it in any table"
    given = rightsize.estimate(fx, "Q4_K_M", "Jetson Orin Nano 8GB", bandwidth_gbps=102)
    assert given.speed and given.speed < r.speed
    assert isinstance(rightsize.detect(), rightsize.Device)
