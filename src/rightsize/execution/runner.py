"""Carry out a plan on this machine, each step with its framework's toolkit (F8).

    rightsize run plan.json                 # every step; serve steps are printed, not started
    run_plan(plan, workdir="runs/qwen3")    # the same from Python

A step is a rendered recipe. A command runs as rendered, with its toolkit's environment first
on PATH; a config (a Python script, an Axolotl YAML) is written into the run directory and run
with the recipe's run line. Each step's log, exit code, wall time, peak VRAM and the size of
what it wrote go into the run's manifest beside what the plan predicted, so a run is also
calibration data (F9).

Most toolkits need nothing more. One whose steps do - llama.cpp's binaries come from its own
install, and its converter needs the model downloaded first - has a Runner, here or from a
plugin through the ``rightsize.runners`` entry point.
"""

from __future__ import annotations

import datetime as _dt
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rightsize import __version__
from rightsize.errors import RightsizeError
from rightsize.execution.envs import Env, ToolkitMissing, install, resolve, tools_root
from rightsize.registry import framework, render
from rightsize.registry.schema import FrameworkInfo, Recipe, RenderedStep
from rightsize.types import GB, Measurement, Plan, RunManifest, RunStep

Log = Callable[[str], None]
ENTRY_POINT_GROUP = "rightsize.runners"
#: Toolkits print progress bars and emoji; a Windows console codepage would turn that into a
#: UnicodeEncodeError in the middle of a training run.
_CHILD_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


class RunFailed(RightsizeError):
    """A step exited with an error, or did not write what its recipe says it writes."""


@dataclass
class Prepared:
    """What to execute for one step."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    path: Path | None = None  # put first on PATH
    note: str | None = None
    version: str | None = None  # of the toolkit that runs it, for the manifest


@dataclass
class Context:
    plan: Plan
    workdir: Path
    models_dir: Path
    install: bool
    log: Log


class Runner(Protocol):
    """How one framework's steps are executed."""

    def inputs(self, plan: Plan, ctx: Context) -> dict[str, Any]:
        """Values the plan's recipes should render with on this machine (tool paths)."""

    def prepare(
        self,
        step: RenderedStep,
        recipe: Recipe,
        info: FrameworkInfo,
        values: dict[str, Any],
        ctx: Context,
    ) -> Prepared:
        """The command for one rendered step, after anything it needs is in place."""


class GenericRunner:
    """Any recipe of a toolkit installed with pip: its environment is found (or installed,
    when asked), a config is written into the run directory, and ``python`` or a program
    the toolkit installed resolves to that environment's copy."""

    def inputs(self, plan: Plan, ctx: Context) -> dict[str, Any]:
        return {}

    def environment(self, info: FrameworkInfo, ctx: Context) -> Env:
        try:
            return resolve(info)
        except ToolkitMissing:
            if not ctx.install:
                raise
            ctx.log(f"installing {info.title} into {tools_root() / info.name}")
            return install(info, log=ctx.log)

    def prepare(
        self,
        step: RenderedStep,
        recipe: Recipe,
        info: FrameworkInfo,
        values: dict[str, Any],
        ctx: Context,
    ) -> Prepared:
        env = self.environment(info, ctx)
        if step.kind == "config":
            path = ctx.workdir / recipe.file_name()
            path.write_text(step.text, encoding="utf-8")
            line = recipe.run_line()
            if line is None:
                return Prepared(argv=[], note=f"wrote {path.name}")
            argv = [token.replace("{file}", str(path)) for token in shlex.split(line)]
        else:
            argv = list(step.argv or [])
        return Prepared(
            argv=self.resolve_program(argv, env),
            env=dict(_CHILD_ENV),
            path=env.bin_dir,
            version=_version(info, env),
        )

    @staticmethod
    def resolve_program(argv: list[str], env: Env) -> list[str]:
        if not argv:
            return argv
        if argv[0] in ("python", "python3"):
            return [str(env.python), *argv[1:]]
        found = env.executable(argv[0])
        return [found, *argv[1:]] if found else argv


class LlamaCppRunner(GenericRunner):
    """llama.cpp's binaries come from its own install (``rightsize tools install llama.cpp``);
    its converter runs with the Python that has rightsize's llamacpp extra; the model is
    downloaded before the converter reads it, and the calibration text before the importance
    matrix is computed."""

    def tools(self, ctx: Context):
        from rightsize.execution.llamacpp import ToolchainError, find_tools

        try:
            return find_tools()
        except ToolchainError as exc:
            if not ctx.install:
                raise ToolkitMissing(
                    "llama.cpp is not installed: rightsize tools install llama.cpp"
                ) from exc
            from rightsize.execution.install import install_llama_cpp

            install_llama_cpp(tools_root() / "llama.cpp", log=ctx.log)
            return find_tools()

    def inputs(self, plan: Plan, ctx: Context) -> dict[str, Any]:
        tools = self.tools(ctx)
        values: dict[str, Any] = {
            "quantize_bin": str(tools.quantize),
            "imatrix_bin": str(tools.imatrix),
            "perplexity_bin": str(tools.perplexity),
            "server_bin": str(tools.root / tools.quantize.name.replace("quantize", "server")),
            "python": sys.executable,
        }
        if tools.convert_script:
            values["convert_script"] = str(tools.convert_script)
        ids = {s.recipe_id for s in plan.steps}
        if "llama.cpp/convert" in ids and not any(s.stage == "finetune" for s in plan.steps):
            # the converter reads the original weights: a snapshot, downloaded before it runs
            values["model_dir"] = str(ctx.models_dir / plan.model.ref.repo.replace("/", "__"))
        if "llama.cpp/imatrix" in ids:
            from rightsize.execution.quantize import default_texts

            values["calibration_file"] = str(default_texts(tools, ctx.log)[0])
        return values

    def prepare(
        self,
        step: RenderedStep,
        recipe: Recipe,
        info: FrameworkInfo,
        values: dict[str, Any],
        ctx: Context,
    ) -> Prepared:
        if recipe.id == "llama.cpp/convert":
            model_dir = Path(values["model_dir"])
            if not model_dir.is_absolute():
                model_dir = ctx.workdir / model_dir
            if not model_dir.exists():
                from rightsize.execution.llamacpp import ensure_snapshot

                ctx.log(f"downloading {ctx.plan.model.ref.repo} -> {model_dir}")
                ensure_snapshot(ctx.plan.model.ref.repo, model_dir.parent)
        tools = self.tools(ctx)
        return Prepared(
            argv=list(step.argv or []), env=dict(_CHILD_ENV), path=tools.root, version=tools.version
        )


_BUILTIN: dict[str, type] = {"llama.cpp": LlamaCppRunner}


def runner_for(name: str) -> Runner:
    """The framework's own Runner (built in, or a plugin's), else the generic one."""
    if name in _BUILTIN:
        return _BUILTIN[name]()
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.load()()
    return GenericRunner()


def _version(info: FrameworkInfo, env: Env) -> str | None:
    """The installed version of the framework's own package, when the environment knows it."""
    wanted = {
        p.split("[")[0].split("=")[0].split("<")[0].split(">")[0].lower()
        for p in info.install.packages
    }
    return next((v for k, v in env.versions.items() if k.lower() in wanted), None)


def _progress(parser, stage: str, log: Log):
    """Every tenth of the way, a line, for tools whose output shows how far they are."""
    if parser is None:
        return None
    shown = {"at": -1}

    def on_text(captured: str) -> None:
        got = parser(captured)
        if got and got[1]:
            pct = min(100, int(100 * got[0] / got[1]))
            if pct // 10 > shown["at"]:
                shown["at"] = pct // 10
                log(f"  {stage} {pct}% ({got[0]}/{got[1]})")

    return on_text


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


_WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".onnx", ".onnx_data", ".bin", ".pt", ".pth", ".npz"}


def _weight_size(path: Path, pattern: str | None) -> int:
    """What a size prediction covers: the files ``pattern`` matches, else every weight file,
    not the tokenizer and config written beside them."""
    if path.is_file():
        return path.stat().st_size
    files = (
        path.glob(pattern)
        if pattern
        else (
            p for p in path.rglob("*") if p.suffix in _WEIGHT_SUFFIXES and "tokenizer" not in p.name
        )
    )
    return sum(p.stat().st_size for p in files if p.is_file())


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _skipped(recipe_id: str, argv: list[str], note: str) -> RunStep:
    return RunStep(
        recipe_id=recipe_id,
        argv=argv,
        started=_now(),
        finished=_now(),
        skipped=True,
        measurements=[Measurement(kind="wall_s", value=0.0, note=note)],
    )


def run_plan(
    plan: Plan,
    *,
    workdir: str | Path | None = None,
    inputs: dict[str, Any] | None = None,
    models_dir: str | Path = "models",
    install_missing: bool = False,
    include_serve: bool = False,
    dry_run: bool = False,
    log: Log = print,
    echo: Log | None = None,
) -> RunManifest:
    """Run every step of ``plan`` in ``workdir`` and return its manifest (also written to
    ``workdir/manifest.json``). A serve step is printed rather than started unless
    ``include_serve``; ``install_missing`` installs a toolkit that is not there yet;
    ``dry_run`` prepares every step and runs none. ``log`` gets one line per step and progress
    where the tool's output shows it; ``echo`` gets every line the tools print (all of it is
    in each step's log file either way). Raises RunFailed at the first step that fails, with
    the manifest written so far."""
    from rightsize.execution.llamacpp import run_step, write_manifest
    from rightsize.execution.progress import parser_for
    from rightsize.rules.render_plan import step_values

    slug = plan.model.ref.repo.replace("/", "__")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = Path(workdir or Path("runs") / f"{slug}-{stamp}")
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = Context(plan, workdir.resolve(), Path(models_dir).resolve(), install_missing, log)

    values: dict[str, Any] = {}
    names = list(dict.fromkeys(s.framework for s in plan.steps if s.recipe_id))
    runners = {name: runner_for(name) for name in names}
    for name in names:
        for key, value in runners[name].inputs(plan, ctx).items():
            values.setdefault(key, value)
    values.update(inputs or {})

    device = next((s.device for s in plan.steps if s.stage == "finetune"), plan.steps[-1].device)
    manifest = RunManifest(
        id=workdir.name,
        created=_now(),
        rightsize_version=__version__,
        model=plan.model.ref,
        facts=plan.model,
        device=device,
        predicted={
            f"{i:02d}-{s.recipe_id}": s.fit
            for i, s in enumerate((s for s in plan.steps if s.recipe_id), start=1)
        },
    )
    toolchain: dict[str, str] = {}
    for i, (pstep, recipe, wanted) in enumerate(step_values(plan, **values), start=1):
        label = f"{i:02d}-{recipe.id.replace('/', '-')}"
        rendered = render(recipe, **wanted)
        if pstep.stage == "serve" and not include_serve:
            manifest.steps.append(
                _skipped(recipe.id, rendered.argv or [], "serve: not started; run it yourself")
            )
            log(f"{label}: serve step, not started: {rendered.text}")
            continue
        info = framework(recipe.framework)
        try:
            prepared = runners[info.name].prepare(rendered, recipe, info, wanted, ctx)
        except ToolkitMissing as exc:
            if not dry_run:
                raise
            # a dry run shows the step and what it still needs, rather than stopping
            prepared = Prepared(argv=list(rendered.argv or []), note=f"dry run; {exc}")
        if not prepared.argv or dry_run:
            note = prepared.note or ("dry run" if dry_run else "nothing to run")
            manifest.steps.append(_skipped(recipe.id, prepared.argv, note))
            log(f"{label}: {note}" + (f": {' '.join(prepared.argv)}" if prepared.argv else ""))
            continue
        log(f"{label}: {' '.join(prepared.argv)}")
        rs, _text, peak = run_step(
            rendered.model_copy(update={"argv": prepared.argv}),
            log_path=workdir / "logs" / f"{label}.log",
            log=echo,
            cwd=workdir,
            on_text=_progress(parser_for(recipe.id), recipe.stage, log),
            extra_path=prepared.path,
            sample_vram=True,
            env_extra=prepared.env,
        )
        if peak is not None:
            rs.measurements.append(
                Measurement(
                    kind="peak_vram_gb",
                    value=round(peak, 3),
                    predicted=pstep.fit.vram_gb if pstep.stage == "finetune" else None,
                    note=f"{pstep.stage} on {pstep.device.name}",
                )
            )
        missing = []
        for name in recipe.writes:
            value = wanted.get(name)
            if value in (None, "") and name in recipe.inputs:
                value = recipe.inputs[name].default  # what the render used
            out = Path(str(value)) if value not in (None, "") else None
            if out is not None and not out.is_absolute():
                out = workdir / out
            if out is None or not out.exists():
                missing.append(name)
                continue
            predicted, size, note = None, _size(out), f"{name}: {out.name}"
            target = next(
                (t for t in recipe.targets if pstep.quant and t.name == pstep.quant.method), None
            )
            if recipe.id == "llama.cpp/quantize":
                from rightsize.fit import predicted_file_gb
                from rightsize.fit.finetune import merged

                # after a fine-tune, the converter read the model the trainer saved
                tuned = any(s.stage == "finetune" for s in plan.steps)
                model = merged(plan.model) if tuned else plan.model
                predicted = round(predicted_file_gb(model, str(wanted.get("quant"))), 3)
            elif target is not None and "file_gb" in pstep.fit.breakdown:
                predicted = pstep.fit.breakdown["file_gb"]
                weights = _weight_size(out, target.weights)
                if weights != size:
                    note += f", weights only ({size / GB:.3f} GB in all)"
                size = weights
            rs.measurements.append(
                Measurement(
                    kind="file_size_gb", value=round(size / GB, 3), predicted=predicted, note=note
                )
            )
        manifest.steps.append(rs)
        toolchain.setdefault(info.name, prepared.version or info.version_tested or "")
        manifest.toolchain = toolchain
        if rs.returncode != 0 or missing:
            manifest.status = "failed"
            write_manifest(manifest, workdir)
            why = (
                f"exit code {rs.returncode}"
                if rs.returncode
                else f"it did not write {', '.join(missing)}"
            )
            raise RunFailed(f"{recipe.id} failed: {why}; log: {rs.log_path}")
    manifest.status = "succeeded"
    write_manifest(manifest, workdir)
    return manifest
