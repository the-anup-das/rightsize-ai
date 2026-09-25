"""Measure this machine's effective memory bandwidth instead of looking it up (F2, F9).

    rightsize bench Qwen/Qwen3-0.6B

Some devices have no published bandwidth to find. NVIDIA states a bus width but no
bandwidth for laptop GPUs, because the memory speed is an OEM choice; a card can be
re-binned; an unusual board may not be in any table. For those, the honest number is not
someone else's spec sheet but what the machine in front of you actually does.

``llama-bench`` reports decode throughput, and the fit engine's speed model is

    tok/s = efficiency * bandwidth / (active weights + KV)

so running it on a model whose size we know and solving for bandwidth gives the effective
figure for this device, memory configuration and driver included. It is cached per device
and used ahead of any table, because a measurement of your machine beats a number about a
model of it.

The same run doubles as a check on the efficiency constant itself: on a device whose
bandwidth we do know, the measurement should land near the published figure, and the gap
is the real efficiency (F9 refits it).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rightsize.errors import RightsizeError
from rightsize.fit.llm import _DECODE_EFFICIENCY, gguf_bpw, kv_cache_gb
from rightsize.types import GB, Device, ModelFacts

Log = Callable[[str], None]

#: Where a measurement is remembered, so it is taken once per machine rather than per run.
CACHE = Path(os.environ.get("RIGHTSIZE_CACHE", Path.home() / ".cache" / "rightsize"))
CACHE_FILE = CACHE / "bandwidth.json"
BENCH_TOKENS = 128
BENCH_REPEATS = 3


class BenchError(RightsizeError):
    pass


def _exe(name: str) -> str:
    import sys

    return f"{name}.exe" if sys.platform == "win32" else name


def _busy_gpu(log: Log) -> list[str]:
    """Other inference runtimes on the GPU right now; they make a reading meaningless."""
    from rightsize.execution.llamacpp import gpu_holders

    holders = gpu_holders()
    if holders:
        log(f"  warning: {', '.join(holders)} is also on the GPU; the reading will be low")
    return holders


def measured_bandwidth(device_name: str) -> float | None:
    """A previously measured figure for this device, if there is one."""
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rec = data.get(device_name)
    return float(rec["bandwidth_gbps"]) if isinstance(rec, dict) else None


def _remember(device_name: str, record: dict[str, Any]) -> None:
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data[device_name] = record
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_llama_bench(
    bench_bin: Path,
    model_gguf: Path,
    *,
    gpu_layers: str = "all",
    tokens: int = BENCH_TOKENS,
    repeats: int = BENCH_REPEATS,
    timeout: float = 900.0,
) -> list[dict[str, Any]]:
    """Decode-only benchmark, as JSON. ``-p 0`` skips prompt processing, which is compute
    bound and would tell us nothing about memory bandwidth."""
    argv = [
        str(bench_bin),
        "-m", str(model_gguf),
        "-p", "0",
        "-n", str(tokens),
        "-ngl", "999" if gpu_layers == "all" else str(gpu_layers),
        "-r", str(repeats),
        "-o", "json",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        raise BenchError(f"llama-bench failed ({proc.returncode}): {' | '.join(tail)}")
    try:
        rows = json.loads(proc.stdout)
    except ValueError as exc:  # llama-bench prints backend banners before the JSON
        start = proc.stdout.find("[")
        if start < 0:
            raise BenchError("llama-bench produced no JSON") from exc
        rows = json.loads(proc.stdout[start:])
    return [r for r in rows if r.get("n_gen")]


def bandwidth_from_throughput(
    facts: ModelFacts, quant: str, tok_per_s: float, *, ctx: int = 512
) -> float:
    """Invert the decode model: bandwidth = tok/s * bytes touched per token / efficiency."""
    bpw, _ = gguf_bpw(quant)
    active = facts.params_active or facts.params_total
    if not active:
        raise BenchError("model parameter count unknown, cannot derive bandwidth")
    active_gb = active * bpw / 8 / GB
    per_token_gb = active_gb + kv_cache_gb(facts, ctx)
    return tok_per_s * per_token_gb / _DECODE_EFFICIENCY


def _published_bandwidth(device: Device) -> float | None:
    """The table figure for this device, ignoring anything we measured earlier.

    resolve() replaces a device's bandwidth with a measurement once one exists, so reading
    it back here would compare a run against its own previous run and call that agreement.
    """
    from rightsize.hardware.db import _bandwidth_for

    gbps, _ = _bandwidth_for(device.name, device.vendor, device.memory_gib)
    return gbps


def measure(
    device: Device,
    facts: ModelFacts,
    quant: str,
    model_gguf: Path,
    *,
    tools: Any = None,
    gpu_layers: str = "all",
    ctx: int = 512,
    save: bool = True,
    log: Log = print,
) -> dict[str, Any]:
    """Measure and remember this device's effective bandwidth. Returns the record."""
    from rightsize.execution.llamacpp import find_tools

    tc = tools if tools is not None else find_tools()
    bench_bin = tc.root / _exe("llama-bench")
    if not bench_bin.exists():
        raise BenchError(f"llama-bench not found in {tc.root}; run 'rightsize tools install'")
    log(f"benchmarking {model_gguf.name} on {device.name} ({BENCH_TOKENS} tokens)")
    busy = _busy_gpu(log)
    rows = run_llama_bench(bench_bin, model_gguf, gpu_layers=gpu_layers)
    if not rows:
        raise BenchError("llama-bench reported no generation result")
    best = max(rows, key=lambda r: float(r.get("avg_ts") or 0))
    tok_per_s = float(best["avg_ts"])
    gbps = bandwidth_from_throughput(facts, quant, tok_per_s, ctx=ctx)
    file_gb = model_gguf.stat().st_size / GB
    record = {
        "bandwidth_gbps": round(gbps, 1),
        "measured_tok_per_s": round(tok_per_s, 2),
        "model": facts.ref.repo,
        "quant": quant,
        "model_file_gb": round(file_gb, 2),
        "efficiency_assumed": _DECODE_EFFICIENCY,
        "llama_cpp": getattr(tc, "version", "unknown"),
        "gpu_busy_during_run": busy or None,
        "note": "derived from llama-bench decode throughput on this machine",
    }
    log(f"  {tok_per_s:.1f} tok/s -> effective bandwidth {gbps:.0f} GB/s")
    published = _published_bandwidth(device)
    if published:
        # A known device tells us how good the efficiency constant is, rather than the
        # other way round. Worth printing: it is the number F9 exists to refit.
        ratio = gbps / published
        record["published_gbps"] = published
        record["efficiency_observed"] = round(_DECODE_EFFICIENCY * ratio, 3)
        observed = _DECODE_EFFICIENCY * ratio
        log(
            f"  published {published:.0f} GB/s, so this decode reached "
            f"{observed:.0%} of peak (model assumes {_DECODE_EFFICIENCY:.0%})"
        )
        if observed < _DECODE_EFFICIENCY * 0.75:
            # Far below is usually the run's fault, not the card's. Say which, because a
            # bad measurement cached as this machine's bandwidth is worse than no
            # measurement: every later estimate inherits it.
            why = []
            if busy:
                why.append(f"the GPU was shared with {', '.join(busy)}")
            if file_gb < 4:
                why.append(
                    f"a {file_gb:.1f} GB model is too small to saturate memory - "
                    "launch overhead dominates, so bigger models read closer to peak"
                )
            for reason in why:
                log(f"  low reading: {reason}")
            record["low_reading"] = why or ["unexplained"]
    if save:
        _remember(device.name, record)
        log(f"  remembered in {CACHE_FILE}")
    return record
