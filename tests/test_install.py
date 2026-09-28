"""Toolchain installer: asset naming, source extraction, idempotency. No network."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

from rightsize.execution import install


def test_asset_names_windows_cuda(monkeypatch) -> None:
    monkeypatch.setattr(install.sys, "platform", "win32")
    monkeypatch.setattr(install.platform, "machine", lambda: "AMD64")
    assert install.asset_names("b11177", "cuda-12.4") == [
        "llama-b11177-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    assert install.asset_names("b11177", "cpu") == ["llama-b11177-bin-win-cpu-x64.zip"]


def test_asset_names_linux_and_mac(monkeypatch) -> None:
    monkeypatch.setattr(install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(install.sys, "platform", "linux")
    assert install.asset_names("b11177", "vulkan") == ["llama-b11177-bin-ubuntu-vulkan-x64.zip"]
    monkeypatch.setattr(install.sys, "platform", "darwin")
    assert install.asset_names("b11177", "arm64") == ["llama-b11177-bin-macos-arm64.zip"]


def test_default_backend(monkeypatch) -> None:
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert install.default_backend() == "cuda-12.4"
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert install.default_backend() == "cpu"
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.platform, "machine", lambda: "arm64")
    assert install.default_backend() == "arm64"


def _fake_source_tarball(version: str) -> bytes:
    buf = io.BytesIO()
    prefix = f"llama.cpp-{version}/"
    files = {
        "convert_hf_to_gguf.py": b"print('convert')\n",
        "convert_lora_to_gguf.py": b"print('lora')\n",
        "conversion/__init__.py": b"# pkg\n",
        "conversion/qwen.py": b"# qwen\n",
        "gguf-py/gguf/__init__.py": b"# gguf\n",
        "README.md": b"not wanted\n",
        "src/llama.cpp": b"not wanted either\n",
    }
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_extract_source_copies_only_converter_paths(tmp_path: Path) -> None:
    logs: list[str] = []
    install._extract_source(_fake_source_tarball("b11177"), tmp_path, "b11177", logs.append)
    assert (tmp_path / "convert_hf_to_gguf.py").exists()
    assert (tmp_path / "conversion" / "qwen.py").exists()
    assert (tmp_path / "gguf-py" / "gguf" / "__init__.py").exists()
    assert not (tmp_path / "README.md").exists() and not (tmp_path / "src").exists()
    assert logs and "extracted 5 converter files" in logs[0]


def test_install_is_idempotent_and_skips_binaries(tmp_path: Path, monkeypatch) -> None:
    exe = ".exe" if sys.platform == "win32" else ""
    for n in ("llama-quantize", "llama-imatrix", "llama-perplexity"):
        (tmp_path / f"{n}{exe}").write_bytes(b"")
    (tmp_path / "VERSION").write_text("b11177\n")
    downloads: list[str] = []

    def fake_download(client, url, log):
        downloads.append(url)
        return _fake_source_tarball("b11177")

    monkeypatch.setattr(install, "_download", fake_download)
    monkeypatch.setattr(install.httpx, "Client", lambda **kw: _NullClient())
    install.install_llama_cpp(tmp_path, version="b11177", backend="cpu", log=lambda s: None)
    assert downloads == [f"{install.ARCHIVE}/b11177.tar.gz"], (
        "binaries present: only the source tarball"
    )
    assert (tmp_path / "conversion" / "__init__.py").exists()
    # second call: already complete, nothing downloaded
    downloads.clear()
    install.install_llama_cpp(tmp_path, version="b11177", backend="cpu", log=lambda s: None)
    assert downloads == []


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_install_check_reports_missing(tmp_path: Path) -> None:
    with pytest.raises(install.InstallError, match="missing"):
        install._post_install_check(tmp_path, lambda s: None)
