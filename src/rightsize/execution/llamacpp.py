"""llama.cpp execution adapter (F8, first slice): convert, imatrix, quantize, KL-divergence gate.

Renders the llama.cpp recipes, runs them with streamed logs, measures what happened, and
writes a RunManifest. Heavy imports (huggingface_hub for downloads) happen inside functions.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rightsize import __version__
from rightsize._data import load_yaml
from rightsize.errors import MissingExtraError, RightsizeError
from rightsize.registry import get as get_recipe
from rightsize.registry import render
from rightsize.registry.schema import RenderedStep
from rightsize.types import Measurement, RunStep

Log = Callable[[str], None]


class ToolchainError(RightsizeError):
    pass


@dataclass
class LlamaCppTools:
    root: Path
    quantize: Path
    imatrix: Path
    perplexity: Path
    convert_script: Path | None
    version: str

    def as_dict(self) -> dict[str, str]:
        return {"llama.cpp": self.version, "llama.cpp_dir": str(self.root)}


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def find_tools(explicit: str | os.PathLike | None = None) -> LlamaCppTools:
    """Locate llama.cpp: explicit path, RIGHTSIZE_LLAMA_CPP, <cwd>/.tools/llama.cpp, then PATH."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("RIGHTSIZE_LLAMA_CPP"):
        candidates.append(Path(os.environ["RIGHTSIZE_LLAMA_CPP"]))
    candidates.append(Path.cwd() / ".tools" / "llama.cpp")
    candidates.append(Path(__file__).resolve().parents[3] / ".tools" / "llama.cpp")
    on_path = shutil.which("llama-quantize")
    if on_path:
        candidates.append(Path(on_path).parent)
    for root in candidates:
        q = root / _exe("llama-quantize")
        if q.exists():
            ver_file = root / "VERSION"
            version = ver_file.read_text().strip() if ver_file.exists() else "unknown"
            conv = root / "convert_hf_to_gguf.py"
            return LlamaCppTools(
                root=root,
                quantize=q,
                imatrix=root / _exe("llama-imatrix"),
                perplexity=root / _exe("llama-perplexity"),
                convert_script=conv if conv.exists() else None,
                version=version,
            )
    raise ToolchainError(
        "llama.cpp not found. Download a release from "
        "https://github.com/ggml-org/llama.cpp/releases into .tools/llama.cpp, "
        "or set RIGHTSIZE_LLAMA_CPP to its folder."
    )


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class _VramSampler:
    """Samples nvidia-smi memory.used every second; reports the peak in GB."""

    def __init__(self) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> _VramSampler:
        if shutil.which("nvidia-smi"):
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                for line in out.stdout.splitlines():
                    if line.strip().isdigit():
                        self.peak_mib = max(self.peak_mib, int(line.strip()))
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(1.0)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)

    @property
    def peak_gb(self) -> float | None:
        return round(self.peak_mib * 1024**2 / 1e9, 2) if self.peak_mib else None


def run_step(
    step: RenderedStep,
    *,
    log_path: Path,
    log: Log | None = None,
    cwd: Path | None = None,
    extra_path: Path | None = None,
    sample_vram: bool = False,
) -> tuple[RunStep, str, float | None]:
    """Run a rendered command, stream its output, return (RunStep, captured text, peak VRAM GB)."""
    assert step.argv, "run_step needs a command recipe"
    env = dict(os.environ)
    if extra_path:
        env["PATH"] = str(extra_path) + os.pathsep + env.get("PATH", "")
    rs = RunStep(recipe_id=step.recipe_id, argv=step.argv, started=_now(), log_path=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    t0 = time.perf_counter()
    sampler = _VramSampler() if sample_vram else None
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(step.argv) + "\n")
        proc = subprocess.Popen(
            step.argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sampler:
            sampler.__enter__()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                fh.write(line)
                captured.append(line)
                if log:
                    log(line.rstrip("\n"))
            rs.returncode = proc.wait()
        finally:
            if sampler:
                sampler.__exit__(None, None, None)
    rs.finished = _now()
    rs.measurements.append(Measurement(kind="wall_s", value=round(time.perf_counter() - t0, 1)))
    return rs, "".join(captured), (sampler.peak_gb if sampler else None)


# ---------------------------------------------------------------- model download and convert


def ensure_snapshot(
    repo: str, models_dir: Path, revision: str = "main", token: str | None = None
) -> Path:
    """Download config, tokenizer and safetensors for a Hub repo (needs the llamacpp extra)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingExtraError("llamacpp", "Downloading Hub snapshots for conversion") from exc
    dest = models_dir / repo.replace("/", "__")
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=dest,
        token=token or os.environ.get("HF_TOKEN"),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "*.jinja",
            "*.py",
        ],
        ignore_patterns=["*.bin", "*.pt", "*.h5", "*.msgpack", "*.onnx", "*.gguf"],
    )
    return dest


def convert_step(
    tools: LlamaCppTools, model_dir: Path, outfile: Path, outtype: str = "auto"
) -> RenderedStep:
    if not tools.convert_script:
        raise ToolchainError("convert_hf_to_gguf.py not found next to the llama.cpp binaries")
    return render(
        get_recipe("llama.cpp/convert"),
        python=sys.executable,
        convert_script=str(tools.convert_script),
        model_dir=str(model_dir),
        outfile=str(outfile),
        outtype=outtype,
    )


def imatrix_step(
    tools: LlamaCppTools,
    model_gguf: Path,
    calibration_file: Path,
    output_file: Path,
    *,
    gpu_layers: str = "all",
    ctx: int = 512,
    chunks: int | None = None,
) -> RenderedStep:
    return render(
        get_recipe("llama.cpp/imatrix"),
        imatrix_bin=str(tools.imatrix),
        model_gguf=str(model_gguf),
        calibration_file=str(calibration_file),
        output_file=str(output_file),
        gpu_layers=gpu_layers,
        ctx=ctx,
        chunks=chunks,
    )


def quantize_step(
    tools: LlamaCppTools,
    input_gguf: Path,
    output_gguf: Path,
    quant: str,
    *,
    imatrix: Path | None = None,
    pure: bool = False,
    threads: int | None = None,
) -> RenderedStep:
    return render(
        get_recipe("llama.cpp/quantize"),
        quantize_bin=str(tools.quantize),
        input_gguf=str(input_gguf),
        output_gguf=str(output_gguf),
        quant=quant,
        imatrix=str(imatrix) if imatrix else None,
        pure=pure,
        threads=threads,
    )


def kld_base_step(
    tools: LlamaCppTools,
    model_gguf: Path,
    eval_file: Path,
    logits_file: Path,
    *,
    gpu_layers: str = "all",
    ctx: int = 512,
    chunks: int | None = None,
) -> RenderedStep:
    return render(
        get_recipe("llama.cpp/kld-base"),
        perplexity_bin=str(tools.perplexity),
        model_gguf=str(model_gguf),
        eval_file=str(eval_file),
        logits_file=str(logits_file),
        gpu_layers=gpu_layers,
        ctx=ctx,
        chunks=chunks,
    )


def kld_eval_step(
    tools: LlamaCppTools,
    model_gguf: Path,
    eval_file: Path,
    logits_file: Path,
    *,
    gpu_layers: str = "all",
    ctx: int = 512,
    chunks: int | None = None,
) -> RenderedStep:
    return render(
        get_recipe("llama.cpp/kld-eval"),
        perplexity_bin=str(tools.perplexity),
        model_gguf=str(model_gguf),
        eval_file=str(eval_file),
        logits_file=str(logits_file),
        gpu_layers=gpu_layers,
        ctx=ctx,
        chunks=chunks,
    )


# ---------------------------------------------------------------- output parsing and the gate

_RE_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)")
_RE_KLD_MEAN = re.compile(r"Mean\s+KLD\s*:\s*([\d.]+)")
_RE_TOP1 = re.compile(r"Same top p\s*:\s*([\d.]+)")
_RE_PPL_RATIO = re.compile(r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-\d.]+)")


def parse_perplexity_output(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if m := _RE_PPL.search(text):
        out["ppl"] = float(m.group(1))
    if m := _RE_KLD_MEAN.search(text):
        out["kld_mean"] = float(m.group(1))
    if m := _RE_TOP1.search(text):
        out["top1_agreement"] = float(m.group(1)) / 100.0
    if m := _RE_PPL_RATIO.search(text):
        out["ln_ppl_ratio"] = float(m.group(1))
    return out


def gate(metrics: dict[str, float]) -> dict[str, object]:
    th = load_yaml("quality/gate_thresholds.yaml")
    verdicts: list[str] = []
    kld = metrics.get("kld_mean")
    if kld is not None:
        verdicts.append(
            "pass"
            if kld < th["kld"]["pass_below"]
            else "warn"
            if kld < th["kld"]["warn_below"]
            else "fail"
        )
    top1 = metrics.get("top1_agreement")
    if top1 is not None:
        t = th["top1_agreement"]
        verdicts.append(
            "pass" if top1 > t["pass_above"] else "warn" if top1 > t["warn_above"] else "fail"
        )
    overall = (
        "fail"
        if "fail" in verdicts
        else "warn"
        if "warn" in verdicts
        else "pass"
        if verdicts
        else "unknown"
    )
    return {"verdict": overall, **metrics, "thresholds": th}


def file_size_gb(path: Path) -> float:
    return round(path.stat().st_size / 1e9, 3)


def write_manifest(manifest, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "manifest.json"
    p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return p


def append_measurements(manifest, path: Path) -> None:
    """Local-only predicted-vs-measured log (seed for F9). Nothing is uploaded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for step in manifest.steps:
            for m in step.measurements:
                if m.predicted is None:
                    continue
                fh.write(
                    json.dumps(
                        {
                            "run": manifest.id,
                            "recipe": step.recipe_id,
                            "model": manifest.model.repo,
                            "device": manifest.device.name if manifest.device else None,
                            "kind": m.kind,
                            "predicted": m.predicted,
                            "measured": m.value,
                            "note": m.note,
                            "rightsize": __version__,
                            "toolchain": manifest.toolchain,
                        }
                    )
                    + "\n"
                )
