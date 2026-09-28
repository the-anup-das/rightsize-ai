"""Running a plan: each step with its framework's toolkit, measured and recorded (F8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from rightsize.execution import envs
from rightsize.execution.runner import (
    GenericRunner,
    LlamaCppRunner,
    RunFailed,
    run_plan,
    runner_for,
)
from rightsize.registry import loader
from rightsize.types import (
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

# A toolkit whose "package" is the standard library, so its steps really run here.
FRAMEWORK = """
name: echo-tool
title: Echo Tool
summary: writes what it is told, for testing the runner
stages: [quantize, serve]
install: {kind: pip, packages: [echo-tool], check: json, line: pip install echo-tool}
homepage: https://example.org/echo
source_doc_url: https://example.org/echo/docs
"""
WRITE = """
id: echo-tool/write
framework: echo-tool
stage: quantize
kind: config
language: python
inputs:
  message: {type: str, default: hello}
  out: {type: path, default: out.txt}
  code: {type: int, default: 0}
  skip: {type: int, default: 0}
writes: [out]
template: |
  import sys
  print("writing {{ message }}")
  if {{ code }} == 0 and not {{ skip }}:
      open("{{ out }}", "w").write("{{ message }}")
  sys.exit({{ code }})
source_doc_url: https://example.org/echo/docs
"""
SERVE = """
id: echo-tool/serve
framework: echo-tool
stage: serve
inputs:
  out: {type: path, default: out.txt}
template: python -c "print(open('{{ out }}').read())"
source_doc_url: https://example.org/echo/docs
"""


@pytest.fixture
def echo(tmp_path, monkeypatch):
    root = tmp_path / "plugin" / "echo-tool"
    root.mkdir(parents=True)
    for name, text in {
        "framework.yaml": FRAMEWORK,
        "write.yaml": WRITE,
        "serve.yaml": SERVE,
    }.items():
        (root / name).write_text(text.lstrip(), encoding="utf-8")
    monkeypatch.setattr(loader, "_plugin_targets", lambda: [("echo", root.parent)])
    monkeypatch.setenv("RIGHTSIZE_TOOLS", str(tmp_path / "tools"))
    loader.clear_cache()
    yield tmp_path
    loader.clear_cache()


def _plan(*recipes: str) -> Plan:
    dev = Device(name="Test GPU", vendor="nvidia", memory_gib=16)
    fit = FitResult(verdict=Verdict.fits, vram_gb=1.0, confidence=0.5, formula_id="test")
    steps = [
        PlanStep(
            stage="serve" if r.endswith("serve") else "quantize",
            framework="echo-tool",
            device=dev,
            fit=fit,
            recipe_id=r,
        )
        for r in recipes
    ]
    facts = ModelFacts(ref=ModelRef(repo="org/tiny"), family=Family.llm)
    return Plan(rank=1, model=facts, mode=Mode.infer, steps=steps, score=1.0)


def test_a_config_step_is_written_run_and_measured(echo) -> None:
    work = echo / "run"
    manifest = run_plan(
        _plan("echo-tool/write"), workdir=work, inputs={"message": "hi"}, log=lambda _line: None
    )
    assert manifest.status == "succeeded"
    assert (work / "out.txt").read_text() == "hi"
    assert (work / "echo_tool_write.py").exists(), "the config is kept beside its output"
    step = manifest.steps[0]
    assert step.returncode == 0 and Path(step.log_path).read_text(encoding="utf-8").count("hi")
    assert any(m.kind == "file_size_gb" and m.note == "out: out.txt" for m in step.measurements)
    saved = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "succeeded" and saved["model"]["repo"] == "org/tiny"


def test_a_failing_step_stops_the_run_and_says_where_to_look(echo) -> None:
    work = echo / "run"
    with pytest.raises(RunFailed, match="exit code 3; log:"):
        run_plan(_plan("echo-tool/write"), workdir=work, inputs={"code": 3}, log=lambda _line: None)
    assert json.loads((work / "manifest.json").read_text())["status"] == "failed"


def test_a_step_that_does_not_write_its_output_has_failed(echo) -> None:
    """Exit 0 is not success when the file the recipe promises is not there."""
    with pytest.raises(RunFailed, match="did not write out"):
        run_plan(
            _plan("echo-tool/write"),
            workdir=echo / "run",
            inputs={"skip": 1},
            log=lambda _line: None,
        )


def test_serve_steps_are_printed_not_started_and_dry_runs_run_nothing(echo) -> None:
    work = echo / "run"
    lines: list[str] = []
    manifest = run_plan(
        _plan("echo-tool/write", "echo-tool/serve"), workdir=work, dry_run=True, log=lines.append
    )
    assert all(s.skipped for s in manifest.steps)
    assert not (work / "out.txt").exists()
    assert any("serve step, not started" in line for line in lines)


def test_a_missing_toolkit_says_how_to_install_it(echo, monkeypatch) -> None:
    monkeypatch.setattr(envs, "_importable", lambda module: False)
    with pytest.raises(envs.ToolkitMissing, match="rightsize tools install echo-tool"):
        run_plan(_plan("echo-tool/write"), workdir=echo / "run", log=lambda _line: None)


def test_installing_a_toolkit_makes_its_own_environment(echo, monkeypatch) -> None:
    """uv creates .tools/<framework>, installs the listed packages, and the versions are
    recorded beside it; no network here, the uv calls are recorded instead."""
    calls: list[list[str]] = []

    def fake_run(argv, log, **_kw):
        calls.append(argv)
        if argv[1:3] == ["venv", str(envs.tools_root() / "echo-tool")]:
            py = envs._python_in(Path(argv[2]))
            py.parent.mkdir(parents=True)
            py.write_text("")
        if "freeze" in argv:
            return "echo-tool==1.2.3\nsix==1.0\n"
        return ""

    monkeypatch.setattr(envs, "_run", fake_run)
    monkeypatch.setattr(envs, "_uv", lambda: ["uv"])
    from rightsize.registry import framework

    env = envs.install(framework("echo-tool"), log=lambda _line: None)
    assert env.where == "managed" and env.versions["echo-tool"] == "1.2.3"
    assert any(c[:3] == ["uv", "pip", "install"] and "echo-tool" in c for c in calls)
    assert not any("--torch-backend" in c for c in calls), "no PyTorch unless it needs one"
    assert envs.managed("echo-tool") is not None
    assert envs.status(framework("echo-tool"))["version"] == "1.2.3"
    assert envs.remove("echo-tool") and envs.managed("echo-tool") is None


def test_python_and_installed_programs_resolve_to_the_toolkits_environment(tmp_path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")).write_text("")
    env = envs.Env("optimum-intel", tmp_path / "py", bin_dir, "managed")
    assert GenericRunner.resolve_program(["python", "x.py"], env)[0] == str(tmp_path / "py")
    assert GenericRunner.resolve_program(["optimum-cli", "export"], env)[0].startswith(str(bin_dir))
    assert GenericRunner.resolve_program(["git", "status"], env) == ["git", "status"]


def test_llama_cpp_has_its_own_runner_and_others_the_generic_one() -> None:
    assert isinstance(runner_for("llama.cpp"), LlamaCppRunner)
    assert type(runner_for("unsloth")) is GenericRunner


def test_the_cli_runs_a_saved_plan(echo, capsys) -> None:
    from rightsize.cli import main

    path = echo / "plan.json"
    path.write_text(
        json.dumps([_plan("echo-tool/write").model_dump(mode="json")]), encoding="utf-8"
    )
    work = echo / "cli-run"
    assert main(["run", str(path), "--workdir", str(work), "--set", "message=from-cli"]) == 0
    assert (work / "out.txt").read_text() == "from-cli"
    assert "echo-tool/write" in capsys.readouterr().out


def test_vram_is_what_the_step_added_not_what_the_desktop_holds(monkeypatch) -> None:
    """nvidia-smi's memory.used is the whole card. The step's figure is the rise over what
    was in use before it started: 1.5 GB of desktop is not the model's."""
    from rightsize.execution import llamacpp

    readings = iter([[1500], [1500], [3000], [2500]])
    monkeypatch.setattr(
        llamacpp._VramSampler, "_read", staticmethod(lambda: next(readings, [2500]))
    )
    sampler = llamacpp._VramSampler()
    sampler.baseline()
    for _ in range(3):
        sampler.peak_mib = [
            max(p, n) for p, n in zip(sampler.peak_mib or [0], sampler._read(), strict=True)
        ]
    assert sampler.peak_gb == round(1500 * 1024**2 / 1e9, 2)


def test_the_version_shown_is_the_frameworks_own() -> None:
    """tools list showed Transformers as 1.15.0: accelerate's version, the first of the
    descriptor's packages in an alphabetical freeze."""
    from rightsize.execution.envs import own_version
    from rightsize.registry import framework

    freeze = {"accelerate": "1.15.0", "bitsandbytes": "0.50.2", "transformers": "5.17.0"}
    assert own_version(framework("transformers"), freeze) == "5.17.0"
    assert own_version(framework("llm-compressor"), {"llmcompressor": "0.14.0"}) == "0.14.0"
    assert own_version(framework("transformers"), {"torch": "2.11.0"}) is None
