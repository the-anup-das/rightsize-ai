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
        assert dev.bandwidth_gbps and dev.memory_gib > 0


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
    with pytest.raises(KeyError, match="no device matches"):
        get("Voodoo 3")
    # a near miss suggests rather than dumping all 250-odd catalogue names
    with pytest.raises(KeyError, match="did you mean"):
        get("RTX 9999")


def test_resolve_accepts_device_or_name() -> None:
    d = Device(name="x", memory_gib=8)
    assert resolve(d) is d
    assert resolve("RTX 3060").name == "RTX 3060 12GB"


def test_nvidia_smi_parser(monkeypatch) -> None:
    monkeypatch.setattr(
        probe, "_run", lambda cmd, timeout=15.0: "NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 610.88\n"
    )
    gpus = probe.nvidia_gpus()
    assert gpus == [
        {"name": "NVIDIA GeForce RTX 4070 Ti SUPER", "memory_gib": 16.0, "driver": "610.88"}
    ]


def test_detect_fills_bandwidth_from_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "nvidia_gpus",
        lambda: [{"name": "NVIDIA GeForce RTX 4070 Ti SUPER", "memory_gib": 16.0, "driver": "x"}],
    )
    monkeypatch.setattr(probe, "system_ram_gib", lambda: 64.0)
    dev = probe.detect()
    assert dev.vendor == "nvidia" and dev.memory_gib == 16.0 and dev.system_ram_gib == 64.0
    assert dev.bandwidth_gbps == 672 and dev.compute_arch == "ada"


def test_norm_strips_vendor_words() -> None:
    assert db._norm("NVIDIA GeForce RTX 4090 24 GB") == "rtx 4090 24gb"


def test_device_memory_is_gib_and_converts_to_decimal_gb() -> None:
    """A "16 GB" card holds 16 GiB, which is 17.18 decimal GB.

    Presets and detection speak GiB, because that is what the vendor and nvidia-smi say
    (this card reports 16376 MiB). Every model size in rightsize is decimal GB, so the fit
    engine reads Device.memory_gb. Treating the 16 as decimal made every verdict on this
    card 7% pessimistic, which is exactly the margin borderline models live in.
    """
    dev = get("RTX 4070 Ti SUPER")
    assert dev.memory_gib == 16
    assert dev.memory_gb == 17.18
    assert pytest.approx(dev.memory_gib, abs=0.01) == 16376 / 1024
    assert Device(name="cpu box", memory_gib=8, system_ram_gib=64).system_ram_gb == 68.719
    assert Device(name="no ram", memory_gib=8).system_ram_gb is None


def test_catalog_is_ingested_and_joined_to_bandwidth() -> None:
    """The catalogue comes from Hugging Face's SKU table; bandwidth comes from ours.

    HF publishes memory, TFLOPS and compute capability for ~250 accelerators but no memory
    bandwidth, so the two files are joined on the device name. A device we have no
    bandwidth for still resolves - the fit engine gives a memory verdict without a tok/s.
    """
    cat = db.catalog()
    assert len(cat) > 200, "the whole HF table, one entry per memory option"
    assert cat["RTX 4090 24GB"].bandwidth_gbps == 1008.0
    # A card nobody has curated and no list covers keeps None rather than a guess
    assert cat["A800 80GB"].bandwidth_gbps is None, "not covered anywhere, and it says so"
    assert cat["RTX 4090 24GB"].compute_arch == "sm_89"
    for dev in cat.values():
        assert dev.provenance and dev.provenance.source_url.startswith("https://")


def test_bandwidth_is_computed_from_bus_width_and_speed() -> None:
    """256-bit GDDR6X at 21 Gbps is 672 GB/s, which is what this card actually does.

    Storing the two facts the vendor publishes, rather than a bandwidth figure copied out
    of someone's database, means the number carries its own derivation.
    """
    table = db.bandwidth()
    ti = table[db._norm("RTX 4070 Ti Super")]
    assert ti["gbps"] == 672.0
    assert ti["how"] == "256-bit GDDR6X at 21 Gbps"
    assert ti["formula_id"] == "mem.bandwidth.bus_x_rate.v0"
    # HBM and unified memory do not divide cleanly, so those are the vendor's own figure
    assert table[db._norm("H100 80GB")]["how"] == "stated by the source"
    assert table[db._norm("Apple M4 Max")]["gbps"] == 546.0


def test_lookup_prefers_presets_then_falls_through_to_the_catalog() -> None:
    assert get("RTX 4070 Ti SUPER").usable_fraction == 0.92  # curated preset
    desktop = get("RTX 5080")
    assert desktop.name == "RTX 5080 16GB", "a bare name means the desktop card, not Mobile"


# ---------------------------------------------------------------- Hub profile hardware

#: The shape https://huggingface.co/api/users/<name>/overview really returns.
_OVERVIEW = {
    "hardwareItems": [
        {"sku": ["GPU", "NVIDIA", "RTX 3090"], "mem": 24, "num": 2},
        {"sku": ["Apple Silicon", "-", "Apple M4 Max"], "mem": 64, "num": 1, "isPrimary": True},
        {"sku": ["GPU", "NVIDIA", "RTX 4090"], "mem": 48, "num": 1},
    ]
}


def _stub(payload, status=200):
    import httpx

    return httpx.MockTransport(lambda request: httpx.Response(status, json=payload))


def test_from_hf_reads_saved_hardware() -> None:
    from rightsize.hardware.hf import from_hf as impl

    devices = impl("someone", transport=_stub(_OVERVIEW))
    assert devices[0].name == "Apple M4 Max 64GB", "the primary machine comes first"
    assert devices[0].bandwidth_gbps == 546.0, "resolved against our own tables"
    assert [d.name for d in devices[1:3]] == ["RTX 3090 24GB #1", "RTX 3090 24GB #2"]
    # 48GB is not a stock 4090; keep the catalogue record but take the user's word on size
    odd = devices[-1]
    assert odd.memory_gib == 48 and odd.vendor == "nvidia"
    assert all(d.provenance and "profile" in (d.provenance.note or "") for d in devices)


def test_from_hf_is_loud_when_there_is_nothing_to_read() -> None:
    from rightsize.hardware.hf import HubHardwareError
    from rightsize.hardware.hf import from_hf as impl

    with pytest.raises(HubHardwareError, match="no Hugging Face profile"):
        impl("ghost", transport=_stub({}, status=404))
    with pytest.raises(HubHardwareError, match="no hardware saved"):
        impl("empty", transport=_stub({"hardwareItems": []}))


def test_ingested_bandwidth_covers_cards_nobody_curated() -> None:
    """Wikipedia's GPU lists fill in what Hugging Face's table and our hand list do not."""
    cat = db.catalog()
    assert cat["RTX 3090 24GB"].bandwidth_gbps == 936.0  # 384-bit GDDR6X at 19.5 Gbps
    assert cat["RTX 5080 16GB"].bandwidth_gbps == 960.0  # 256-bit GDDR7 at 30
    assert cat["RX 7900 XTX 24GB"].bandwidth_gbps == 960.0  # 384-bit GDDR6 at 20
    assert "GDDR6X at 19.5 Gbps" in cat["RTX 3090 24GB"].provenance.note


def test_bandwidth_never_crosses_vendors() -> None:
    """_norm drops the vendor word, so "Apple M4" and a Mobility Radeon M4 collide.

    Whichever page a record came from is recorded, and a device only takes a number from
    its own vendor's list. Without this an Apple chip silently reported a Radeon's memory.
    """
    assert db._bandwidth_for("M4", "amd")[0] != db._bandwidth_for("Apple M4", "apple")[0]
    gbps, hit = db._bandwidth_for("Apple M4", "apple")
    assert gbps == 120.0 and hit["how"] == "stated by the source"
    assert db._bandwidth_for("H100 80GB", "nvidia")[0] == 3350.0


def test_a_source_that_is_consistently_wrong_is_overridden_by_hand() -> None:
    """The cross-check catches a source contradicting itself, not one that is just wrong.

    Wikipedia's GPU list gives the H200 3360 GB/s - the H100's figure - and its bus width
    and clock agree with that, so the arithmetic reproduces the error faithfully. NVIDIA
    states 4.8 TB/s. bandwidth.yaml records the vendor's number and flags the disagreement
    so the ingest does not keep failing on it.
    """
    dev = db.catalog()["H200 141GB"]
    assert dev.bandwidth_gbps == 4800.0
    assert "nvidia.com" in dev.provenance.source_url


def test_size_variants_do_not_borrow_each_others_bandwidth() -> None:
    """An A100 40GB is 1555 GB/s and an 80GB is 2039; a 3060 is 240 at 8GB and 360 at 12.

    The join asks for "<model> <size>GB" before the bare model name, because a name on its
    own does not identify the card when the memory differs.
    """
    cat = db.catalog()
    assert cat["A100 40GB"].bandwidth_gbps == 1555.2
    assert cat["A100 80GB"].bandwidth_gbps == 2039.0
    assert cat["RTX 3060 8GB"].bandwidth_gbps == 240.0
    assert cat["RTX 3060 12GB"].bandwidth_gbps == 360.0
