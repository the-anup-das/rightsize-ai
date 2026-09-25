"""llama.cpp execution adapter (F8, first slice): convert, imatrix, quantize, KL-divergence gate.

Renders the llama.cpp recipes, runs them with streamed logs, measures what happened, and
writes a RunManifest. Heavy imports (huggingface_hub for downloads) happen inside functions.
"""

from __future__ import annotations

import codecs
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
from rightsize.types import GB, GIB, Measurement, RunStep

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
        """Peak in decimal GB, to sit next to the fit engine's prediction in a Measurement."""
        return round(self.peak_mib * 1024**2 / GB, 2) if self.peak_mib else None


# ------------------------------------------------------------------ GPU preflight

#: Names worth reporting when the GPU is busy. Everything with a window is on the GPU too;
#: these are the ones that hold gigabytes and that you can actually unload.
_GPU_HOGS = ("llama-server", "ollama", "lms", "lm studio", "koboldcpp", "comfyui", "python")


def _smi(query: str) -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gpu_memory() -> tuple[float, float] | None:
    """(free_gib, total_gib) for the first NVIDIA GPU, or None when there is none.

    In GiB, the unit nvidia-smi itself uses, so what rightsize prints can be compared with
    what the driver prints. Model sizes are decimal GB; ``preflight_vram`` converts.
    """
    for line in _smi("gpu=memory.free,memory.total"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            free, total = (int(p) * 1024**2 / GIB for p in parts)
            return round(free, 2), round(total, 2)
    return None


def gpu_holders() -> list[str]:
    """Names of inference runtimes currently on the GPU, with sizes when the driver reports
    them. Windows WDDM returns ``[N/A]`` for per-process memory, so names may come alone."""
    found: list[str] = []
    for line in _smi("compute-apps=process_name,used_memory"):
        name, _, used = line.partition(",")
        # nvidia-smi reports the OS's own paths, so split on both separators whatever we run on.
        stem = name.strip().replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".exe")
        if not any(h in stem.lower() for h in _GPU_HOGS):
            continue
        mib = used.strip()
        found.append(f"{stem} ({int(mib) * 1024**2 / GIB:.1f} GiB)" if mib.isdigit() else stem)
    return found


def preflight_vram(need_gb: float, *, what: str, log: Log) -> bool:
    """Log free VRAM before a GPU step; warn when another process leaves too little.

    ``need_gb`` is a model size in decimal GB, as the fit engine reports it; the comparison
    happens in GiB so it lines up with the driver's own numbers.

    Returns True when the step looks safe. A step that starts with just enough memory can
    still die later if the other process grows, so this warns rather than blocks.
    """
    mem = gpu_memory()
    if mem is None:
        return True
    free, total = mem
    need = need_gb * GB / GIB
    if free >= need:
        log(f"gpu: {free:.1f} GiB free of {total:.1f} GiB, {what} needs about {need:.1f} GiB")
        return True
    holders = gpu_holders()
    log(
        f"gpu: only {free:.1f} GiB free of {total:.1f} GiB but {what} needs about "
        f"{need:.1f} GiB. llama.cpp will fail mid-run, often without an error message."
    )
    if holders:
        log(f"gpu: memory is held by {', '.join(holders)} - unload it, or pass --gpu-layers 0")
    return False


def log_tail(path: str | os.PathLike, lines: int = 6, width: int = 300) -> str:
    """Last few lines of a step log, for error messages. llama.cpp prints its last words
    without a trailing newline, so a crash leaves them at the end of the file."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    tail = [ln.strip() for ln in text.replace("\r", "\n").splitlines() if ln.strip()][-lines:]
    return "\n".join(ln[-width:] for ln in tail)


def run_step(
    step: RenderedStep,
    *,
    log_path: Path,
    log: Log | None = None,
    on_text: Callable[[str], None] | None = None,
    cwd: Path | None = None,
    extra_path: Path | None = None,
    sample_vram: bool = False,
) -> tuple[RunStep, str, float | None]:
    """Run a rendered command, stream its output, return (RunStep, captured text, peak VRAM GB).

    Output is read in chunks, not lines, so tools that print progress without newlines
    (llama-perplexity's ``[1]4.97,[2]5.88,`` ...) still drive ``on_text`` live. ``log`` receives
    complete lines; ``on_text`` receives everything captured so far, partial line included.
    """
    assert step.argv, "run_step needs a command recipe"
    env = dict(os.environ)
    if extra_path:
        env["PATH"] = str(extra_path) + os.pathsep + env.get("PATH", "")
    rs = RunStep(recipe_id=step.recipe_id, argv=step.argv, started=_now(), log_path=str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    t0 = time.perf_counter()
    sampler = _VramSampler() if sample_vram else None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(step.argv) + "\n")
        proc = subprocess.Popen(
            step.argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        if sampler:
            sampler.__enter__()
        try:
            assert proc.stdout is not None
            while True:
                data = proc.stdout.read1(65536)
                if not data:
                    break
                text = decoder.decode(data)
                if not text:
                    continue
                fh.write(text)
                fh.flush()
                captured.append(text)
                pending += text
                *lines, pending = pending.replace("\r", "\n").split("\n")
                if log:
                    for line in lines:
                        if line:
                            log(line)
                if on_text:
                    on_text("".join(captured))
            if pending and log:
                log(pending)
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
    return round(path.stat().st_size / GB, 3)


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
