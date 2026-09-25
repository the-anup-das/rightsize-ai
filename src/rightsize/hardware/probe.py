"""Detect the local machine as a Device (F2, first slice).

Subprocess and stdlib only: nvidia-smi for NVIDIA, system_profiler/sysctl on macOS, /proc or
wmic for RAM. Bandwidth comes from the presets table when the GPU name matches a preset.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

from rightsize.types import Device, Provenance


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def _os_name() -> str:
    return {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform, "unknown")


def system_ram_gb() -> float | None:
    if sys.platform == "win32":
        out = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
        if out:
            digits = re.findall(r"\d+", out)
            if digits:
                return round(int(digits[0]) / 1e9, 1)
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if out and out.strip().isdigit():
            return round(int(out.strip()) / 1e9, 1)
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out and out.strip().isdigit():
            return round(int(out.strip()) / 1e9, 1)
    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) * 1024 / 1e9, 1)
        except OSError:
            pass
    return None


def nvidia_gpus() -> list[dict]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if not out:
        return gpus
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_gb": round(int(parts[1]) / 1024, 1),
                    "driver": parts[2] if len(parts) > 2 else None,
                }
            )
    return gpus


def _apple_chip() -> str | None:
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return out.strip() if out else None


def detect() -> Device:
    """Best-effort local device. Never imports torch. Raises nothing; unknown fields stay None."""
    ram = system_ram_gb()
    os_name = _os_name()
    prov = Provenance(
        source_url="local://detect", fetched_at="now", note="rightsize.hardware.detect"
    )

    gpus = nvidia_gpus()
    if gpus:
        g = gpus[0]
        dev = Device(
            name=g["name"],
            vendor="nvidia",
            memory_gb=g["memory_gb"],
            system_ram_gb=ram,
            backends=["cuda", "vulkan"],
            os=os_name,
            usable_fraction=0.92,
            provenance=prov,
        )
        _fill_from_preset(dev)
        return dev

    if sys.platform == "darwin" and platform.machine() == "arm64":
        chip = _apple_chip() or "Apple Silicon"
        dev = Device(
            name=chip,
            vendor="apple",
            memory_gb=ram or 8.0,
            system_ram_gb=ram,
            backends=["metal"],
            os=os_name,
            usable_fraction=0.70,
            provenance=prov,
        )
        _fill_from_preset(dev)
        return dev

    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
    return Device(
        name=cpu,
        vendor="cpu",
        memory_gb=ram or 8.0,
        system_ram_gb=ram,
        backends=["cpu"],
        os=os_name,
        usable_fraction=0.80,
        provenance=prov,
    )


def _fill_from_preset(dev: Device) -> None:
    """Copy bandwidth and architecture from a matching preset, if any."""
    from rightsize.hardware.db import get

    try:
        p = get(f"{dev.name} {int(round(dev.memory_gb))}GB")
    except KeyError:
        try:
            p = get(dev.name)
        except KeyError:
            return
    if dev.bandwidth_gbps is None:
        dev.bandwidth_gbps = p.bandwidth_gbps
    if dev.compute_arch is None:
        dev.compute_arch = p.compute_arch
