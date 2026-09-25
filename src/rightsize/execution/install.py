"""Install pinned toolchains into .tools/ (F8). Today: llama.cpp.

    rightsize tools install llama.cpp                 # auto-picks the backend for this machine
    rightsize tools install llama.cpp --backend cpu   # or cuda-12.4, cuda-13.4, vulkan, rocm, ...

Downloads the release zip for the chosen backend, the CUDA runtime zip when needed, and the
source pieces the conversion step imports (``convert_hf_to_gguf.py``, ``conversion/``,
``gguf-py/``) from the same tag, so binaries and converter always match. httpx only.
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from rightsize.errors import RightsizeError

LLAMA_CPP_VERSION = "b11177"  # bump deliberately; recipes carry version_tested
RELEASES = "https://github.com/ggml-org/llama.cpp/releases/download"
ARCHIVE = "https://github.com/ggml-org/llama.cpp/archive/refs/tags"
SOURCE_PATHS = ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion/", "gguf-py/")

Log = Callable[[str], None]


class InstallError(RightsizeError):
    pass


def _os_arch() -> tuple[str, str]:
    os_name = {"win32": "win", "darwin": "macos", "linux": "ubuntu"}.get(sys.platform)
    if os_name is None:
        raise InstallError(f"no llama.cpp prebuilt for platform {sys.platform}")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return os_name, arch


def default_backend() -> str:
    """Pick a backend from what the machine has: CUDA if nvidia-smi works, Metal on Apple
    Silicon, otherwise CPU. Vulkan/ROCm/SYCL are opt-in via --backend."""
    if sys.platform == "darwin":
        return "arm64" if platform.machine().lower() == "arm64" else "x64"
    if shutil.which("nvidia-smi"):
        return "cuda-12.4"
    return "cpu"


def asset_names(version: str, backend: str) -> list[str]:
    os_name, arch = _os_arch()
    if os_name == "macos":
        return [f"llama-{version}-bin-macos-{backend}.zip"]
    names = [f"llama-{version}-bin-{os_name}-{backend}-{arch}.zip"]
    if backend.startswith("cuda-") and os_name == "win":
        names.append(f"cudart-llama-bin-win-{backend}-{arch}.zip")
    return names


def _download(client: httpx.Client, url: str, log: Log) -> bytes:
    log(f"downloading {url.rsplit('/', 1)[-1]}")
    with client.stream("GET", url) as r:
        if r.status_code == 404:
            raise InstallError(f"asset not found: {url}")
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_bytes(1 << 20):
            buf.write(chunk)
    return buf.getvalue()


def install_llama_cpp(
    dest: str | os.PathLike = ".tools/llama.cpp",
    *,
    version: str = LLAMA_CPP_VERSION,
    backend: str | None = None,
    log: Log = print,
    timeout: float = 600.0,
) -> Path:
    """Install binaries + converter for one llama.cpp tag. Idempotent for the same version."""
    dest = Path(dest)
    backend = backend or default_backend()
    marker = dest / "VERSION"
    if marker.exists() and marker.read_text().strip() == version and (dest / "conversion").is_dir():
        log(f"llama.cpp {version} already installed in {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    exe = ".exe" if sys.platform == "win32" else ""
    have_binaries = (
        marker.exists()
        and marker.read_text().strip() == version
        and (dest / f"llama-quantize{exe}").exists()
    )
    with httpx.Client(follow_redirects=True, timeout=timeout) as c:
        if have_binaries:
            log(f"binaries for {version} present, fetching the converter only")
        else:
            for name in asset_names(version, backend):
                data = _download(c, f"{RELEASES}/{version}/{name}", log)
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    z.extractall(dest)
        src = _download(c, f"{ARCHIVE}/{version}.tar.gz", log)
    _extract_source(src, dest, version, log)
    marker.write_text(version + "\n")
    _post_install_check(dest, log)
    return dest


def _extract_source(tar_bytes: bytes, dest: Path, version: str, log: Log) -> None:
    """Copy only the converter and its packages out of the source tarball."""
    for rel in SOURCE_PATHS:
        target = dest / rel.rstrip("/")
        if target.is_dir():
            shutil.rmtree(target)
    prefix = f"llama.cpp-{version}/"
    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix) :]
            if not any(rel == p.rstrip("/") or rel.startswith(p) for p in SOURCE_PATHS):
                continue
            if member.isdir():
                (dest / rel).mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            fh = tar.extractfile(member)
            assert fh is not None
            out.write_bytes(fh.read())
            count += 1
    log(f"extracted {count} converter files ({', '.join(p.rstrip('/') for p in SOURCE_PATHS)})")


def _post_install_check(dest: Path, log: Log) -> None:
    exe = ".exe" if sys.platform == "win32" else ""
    missing = [
        n
        for n in ("llama-quantize", "llama-imatrix", "llama-perplexity")
        if not (dest / f"{n}{exe}").exists()
    ]
    if missing:
        raise InstallError(f"install incomplete, missing {missing} in {dest}")
    if not (dest / "conversion" / "__init__.py").exists():
        raise InstallError("install incomplete: conversion package missing")
    log(f"llama.cpp ready in {dest}")
