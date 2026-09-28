"""What local runtimes have loaded, and how much GPU memory it holds (F9).

Ollama reports both: /api/ps gives each loaded model's size in VRAM. LM Studio's REST API
reports what is loaded, with its quantization and context length, but not its memory, so
that comes from the GPU driver per process: nvidia-smi on Linux, and on Windows (where
WDDM hides per-process memory from nvidia-smi) the "GPU Process Memory" performance
counters. A reading is attributed to a model only when one model is loaded in that
runtime, so two models never share one measurement.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

LM_STUDIO = "http://localhost:1234"
OLLAMA = "http://localhost:11434"
#: The process LM Studio runs a loaded model in.
LM_STUDIO_ENGINE = "llama-server"


@dataclass(frozen=True)
class Loaded:
    runtime: str  # "lm studio" or "ollama"
    name: str
    quant: str | None
    ctx: int | None
    vram_gb: float | None  # measured, when the runtime reports it
    hub_repo: str | None  # a Hub id to predict from, when the name is one


def _get(url: str, client=None):
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=3)
    try:
        r = client.get(url)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if own:
            client.close()


def lm_studio(base: str = LM_STUDIO, client=None) -> list[Loaded]:
    """Models LM Studio has loaded. Its ids are "publisher/name", often a Hub id."""
    body = _get(f"{base}/api/v0/models", client) or {}
    out = []
    for m in body.get("data", []):
        if m.get("state") != "loaded" or m.get("type") not in ("llm", "vlm"):
            continue
        mid = str(m.get("id", ""))
        out.append(
            Loaded(
                runtime="lm studio",
                name=mid,
                quant=m.get("quantization"),
                ctx=m.get("loaded_context_length"),
                vram_gb=None,
                hub_repo=mid if "/" in mid else None,
            )
        )
    return out


def ollama(base: str = OLLAMA, client=None) -> list[Loaded]:
    """Models Ollama has loaded, with the VRAM it reports for each. A model pulled as
    hf.co/org/repo:QUANT names its Hub repo; library models (qwen3:14b) do not."""
    body = _get(f"{base}/api/ps", client) or {}
    out = []
    for m in body.get("models", []):
        name = str(m.get("name", ""))
        repo = None
        if name.startswith("hf.co/"):
            repo = name.removeprefix("hf.co/").split(":")[0]
        vram = m.get("size_vram")
        out.append(
            Loaded(
                runtime="ollama",
                name=name,
                quant=(m.get("details") or {}).get("quantization_level"),
                ctx=m.get("context_length"),
                vram_gb=round(vram / 1e9, 3) if vram else None,
                hub_repo=repo,
            )
        )
    return out


def _nvidia_smi_processes() -> list[tuple[str, float]]:
    """(process name, GB) per process from nvidia-smi; empty where the driver hides it."""
    from rightsize.execution.llamacpp import _smi

    return parse_nvidia_smi(_smi("compute-apps=process_name,used_memory"))


def parse_nvidia_smi(lines: list[str]) -> list[tuple[str, float]]:
    out = []
    for line in lines:
        name, _, used = line.rpartition(",")
        used = used.strip()
        if not used.isdigit():  # "[N/A]" under Windows WDDM
            continue
        stem = name.strip().replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".exe")
        out.append((stem, int(used) * 1024**2 / 1e9))
    return out


_COUNTER = re.compile(r"pid_(\d+)_.*?=(\d+)$")


def parse_windows_counters(
    lines: list[str], names: dict[int, str], limit_gb: float | None = None
) -> list[tuple[str, float]]:
    """(process name, GB) from "InstanceName=bytes" lines of the GPU Process Memory counter.

    The counter reports impossible values for some processes (terabytes for a browser),
    so readings above the card's memory are dropped."""
    best: dict[int, float] = {}
    for line in lines:
        m = _COUNTER.search(line.strip())
        if not m:
            continue
        pid, gb = int(m.group(1)), int(m.group(2)) / 1e9
        if limit_gb and gb > limit_gb:
            continue
        best[pid] = max(best.get(pid, 0.0), gb)
    return [(names.get(pid, str(pid)), gb) for pid, gb in best.items() if gb > 0]


def _windows_processes(limit_gb: float | None) -> list[tuple[str, float]]:
    script = (
        "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | "
        "ForEach-Object { $_.InstanceName + '=' + [int64]$_.CookedValue }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        tasks = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    names: dict[int, str] = {}
    for row in tasks.stdout.splitlines():
        parts = [p.strip('"') for p in row.split('","')]
        if len(parts) >= 2 and parts[1].isdigit():
            names[int(parts[1])] = parts[0].removesuffix(".exe")
    return parse_windows_counters(out.stdout.splitlines(), names, limit_gb)


def process_vram(name: str, limit_gb: float | None = None) -> list[float]:
    """GB each process called ``name`` holds on the GPU; empty when it cannot be read."""
    rows = _nvidia_smi_processes()
    if not rows and sys.platform == "win32":
        rows = _windows_processes(limit_gb)
    return [gb for proc, gb in rows if proc.lower() == name.lower()]
