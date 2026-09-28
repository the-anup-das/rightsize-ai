"""One Python environment per toolkit, installed when someone picks that toolkit (F8).

People use one toolkit at a time, and toolkits pin conflicting versions of the same libraries
(Unsloth caps transformers and trl; llm-compressor wants its own), so each gets its own
virtual environment under ``.tools/<framework>``, created with uv from the packages its
framework.yaml lists. Nothing is installed until ``rightsize tools install <framework>`` (or
``rightsize run --install``) asks for it. uv links files from its cache, so a second
environment with the same PyTorch costs little extra disk.

A toolkit already importable in the Python running rightsize is used as it is, and a binary
or app (llama.cpp, Ollama) is found on PATH; a managed environment wins over both.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rightsize.errors import RightsizeError

Log = Callable[[str], None]

TOOLS_ENV = "RIGHTSIZE_TOOLS"
MARKER = "rightsize-env.json"
DEFAULT_PYTHON = "3.12"


class ToolkitMissing(RightsizeError):
    """A step needs a toolkit that is not installed here."""


def tools_root() -> Path:
    """Where toolchains and environments live: RIGHTSIZE_TOOLS, else ./.tools."""
    return Path(os.environ.get(TOOLS_ENV, ".tools"))


def _python_in(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _bin_in(root: Path) -> Path:
    return root / ("Scripts" if os.name == "nt" else "bin")


@dataclass(frozen=True)
class Env:
    """Where a toolkit's steps run."""

    framework: str
    python: Path
    bin_dir: Path | None
    where: str  # "managed", "current" or "path"
    versions: dict[str, str] = field(default_factory=dict)

    def executable(self, name: str) -> str | None:
        """A program the environment installed (optimum-cli, axolotl, trl), as a full path."""
        if self.bin_dir is None:
            return None
        for candidate in (name, name + ".exe", name + ".cmd"):
            path = self.bin_dir / candidate
            if path.is_file():
                return str(path)
        return None


def managed(framework: str) -> Env | None:
    """The environment ``rightsize tools install`` made for this framework, if any."""
    root = tools_root() / framework
    marker = root / MARKER
    if not (marker.is_file() and _python_in(root).exists()):
        return None
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
    except ValueError:
        info = {}
    return Env(framework, _python_in(root), _bin_in(root), "managed", info.get("versions", {}))


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def resolve(info) -> Env:
    """Where this framework's steps run: its managed environment; else this Python, when the
    toolkit imports here; else PATH, for binaries and apps. Raises ToolkitMissing, saying how
    to install it, when none of those has it."""
    env = managed(info.name)
    if env is not None:
        return env
    check = info.install.check
    if info.install.kind == "pip":
        if check and _importable(check):
            return Env(info.name, Path(sys.executable), Path(sys.executable).parent, "current")
        raise ToolkitMissing(
            f"{info.title} is not installed: rightsize tools install {info.name} "
            f"(or install it yourself: {info.install.line})"
        )
    if check and shutil.which(check):
        return Env(info.name, Path(sys.executable), None, "path")
    raise ToolkitMissing(f"{info.title} is not installed: {info.install.line}")


def _uv() -> list[str]:
    found = shutil.which("uv")
    if found:
        return [found]
    if _importable("uv"):
        return [sys.executable, "-m", "uv"]
    raise ToolkitMissing("installing a toolkit needs uv: pip install uv")


def _run(argv: list[str], log: Log) -> str:
    log("$ " + " ".join(argv))
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    for line in out.splitlines()[-15:]:
        log("  " + line)
    if proc.returncode != 0:
        raise ToolkitMissing(f"{argv[0]} failed with exit code {proc.returncode}: "
                             f"{out.strip().splitlines()[-1] if out.strip() else ''}")
    return proc.stdout or ""


_TORCH_CHECK = ("import json, torch; print(json.dumps([torch.__version__, "
                "torch.cuda.is_available(), torch.version.cuda]))")


def install(info, *, python: str = DEFAULT_PYTHON, log: Log = print) -> Env:
    """Create (or bring up to date) ``.tools/<framework>`` with the packages its
    framework.yaml lists, PyTorch built for this machine's GPU when it needs one, then check
    the toolkit imports and record the versions installed."""
    spec = info.install
    if spec.kind != "pip":
        raise ToolkitMissing(f"{info.title} is not a Python package; install it with: {spec.line}")
    if not spec.packages:
        raise ToolkitMissing(f"{info.name}'s framework.yaml lists no packages to install; "
                             f"install it yourself: {spec.line}")
    uv = _uv()
    root = tools_root() / info.name
    py = _python_in(root)
    if not py.exists():
        _run([*uv, "venv", str(root), "--python", python], log)
    cmd = [*uv, "pip", "install", "--python", str(py), *spec.packages]
    if spec.needs_torch:
        cmd += ["--torch-backend", "auto"]
    _run(cmd, log)
    if spec.check:
        _run([str(py), "-c", f"import {spec.check}"], log)
    freeze = _run([*uv, "pip", "freeze", "--python", str(py)], lambda _line: None)
    versions = dict(line.split("==", 1) for line in freeze.splitlines() if "==" in line)
    torch = None
    if spec.needs_torch:
        torch = json.loads(_run([str(py), "-c", _TORCH_CHECK], lambda _line: None).strip())
        if not torch[1]:
            log("warning: PyTorch installed without a usable GPU; steps will run on the CPU")
    marker = {
        "framework": info.name,
        "packages": spec.packages,
        "versions": versions,
        "torch": torch,
        "python": python,
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }
    (root / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return Env(info.name, py, _bin_in(root), "managed", versions)


def remove(framework: str) -> bool:
    """Delete a managed environment; False when there is none. Only a folder rightsize made
    (it has the marker file) is ever deleted."""
    root = tools_root() / framework
    if not (root / MARKER).is_file():
        return False
    shutil.rmtree(root)
    return True


def status(info) -> dict[str, str | None]:
    """Whether and where a framework is installed, for ``rightsize tools list``."""
    try:
        env = resolve(info)
    except ToolkitMissing:
        if info.name == "llama.cpp":
            from rightsize.execution.llamacpp import ToolchainError, find_tools

            try:
                return {"where": "tools", "path": str(find_tools().root), "version": None}
            except ToolchainError:
                pass
        return {"where": None, "path": None, "version": None}
    names = {p.split("[")[0].split("=")[0].split("<")[0].split(">")[0].lower()
             for p in info.install.packages}
    version = next((v for k, v in env.versions.items() if k.lower() in names), None)
    path = str(env.python.parent.parent) if env.where == "managed" else (
        str(env.python) if env.where == "current" else shutil.which(info.install.check or ""))
    return {"where": env.where, "path": path, "version": version}
