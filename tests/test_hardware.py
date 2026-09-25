"""Presets, fuzzy lookup and detection parsers (no real hardware required)."""

from __future__ import annotations

import pytest

from rightsize.hardware import db, get, presets, probe, resolve
from rightsize.types import Device


def test_presets_have_provenance_and_bandwidth() -> None:
    table = presets()
    assert len(table) >= 14
    for dev in table.values():
        assert dev.provenance and dev.provenance.source_url.startswith("https://")
        assert dev.bandwidth_gbps and dev.memory_gb > 0


@pytest.mark.parametrize(
    "query,expected",
    [
        ("RTX 4090", "RTX 4090 24GB"),
        ("GeForce RTX 4090 24GB", "RTX 4090 24GB"),
        ("NVIDIA GeForce RTX 4070 Ti SUPER 16GB", "RTX 4070 Ti SUPER 16GB"),
        ("h100", "H100 80GB"),
        ("m4 max", "M4 Max 64GB"),
    ],
)
def test_fuzzy_lookup(query: str, expected: str) -> None:
    assert get(query).name == expected


def test_unknown_preset_raises_with_hint() -> None:
    with pytest.raises(KeyError, match="known"):
        get("Voodoo 3")


def test_resolve_accepts_device_or_name() -> None:
    d = Device(name="x", memory_gb=8)
    assert resolve(d) is d
    assert resolve("RTX 3060").name == "RTX 3060 12GB"


def test_nvidia_smi_parser(monkeypatch) -> None:
    monkeypatch.setattr(
        probe, "_run", lambda cmd, timeout=15.0: "NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 610.88\n"
    )
    gpus = probe.nvidia_gpus()
    assert gpus == [
        {"name": "NVIDIA GeForce RTX 4070 Ti SUPER", "memory_gb": 16.0, "driver": "610.88"}
    ]


def test_detect_fills_bandwidth_from_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "nvidia_gpus",
        lambda: [{"name": "NVIDIA GeForce RTX 4070 Ti SUPER", "memory_gb": 16.0, "driver": "x"}],
    )
    monkeypatch.setattr(probe, "system_ram_gb", lambda: 64.0)
    dev = probe.detect()
    assert dev.vendor == "nvidia" and dev.memory_gb == 16.0 and dev.system_ram_gb == 64.0
    assert dev.bandwidth_gbps == 672 and dev.compute_arch == "ada"


def test_norm_strips_vendor_words() -> None:
    assert db._norm("NVIDIA GeForce RTX 4090 24 GB") == "rtx 4090 24gb"
