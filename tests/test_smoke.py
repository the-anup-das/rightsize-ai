"""Placeholder-release smoke tests: import, version, CLI, Plan JSON round-trip."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

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


def test_cli_recommend_points_at_plan(capsys) -> None:
    assert main(["recommend"]) == 0
    captured = capsys.readouterr().out
    assert "F04-rules-engine.md" in captured


def test_cli_json_flag(capsys) -> None:
    assert main(["--json", "recommend"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_implemented"
    assert payload["plan"].endswith("F04-rules-engine.md")


def test_public_functions_raise_not_implemented_yet() -> None:
    with pytest.raises(rightsize.NotImplementedYet) as info:
        rightsize.recommend()
    assert "F04" in str(info.value)


def _sample_plan() -> Plan:
    device = Device(name="RTX 4090", vendor="nvidia", memory_gb=24, bandwidth_gbps=1008, os="linux")
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
