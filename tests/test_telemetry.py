"""Calibration (F9): consent, what a record may contain, the recorders, and the refit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from rightsize import telemetry
from rightsize.telemetry import record as rec
from rightsize.telemetry import sources
from rightsize.telemetry.calibrate import calibrate
from rightsize.types import Device, ModelFacts

FIXTURES = Path(__file__).parent / "fixtures" / "facts"
RTX = Device(name="RTX 4070 Ti SUPER 16GB", vendor="nvidia", memory_gib=16,
             compute_capability=8.9, os="windows")

# LM Studio's /api/v0/models, trimmed from a live response on 2026-09-25
LM_STUDIO_MODELS = {"data": [
    {"id": "openai/gpt-oss-20b", "object": "model", "type": "llm", "publisher": "openai",
     "arch": "gpt-oss", "compatibility_type": "gguf", "quantization": "MXFP4",
     "state": "loaded", "max_context_length": 131072, "loaded_context_length": 8192},
    {"id": "google/gemma-4-e2b", "object": "model", "type": "vlm", "quantization": "Q4_K_M",
     "state": "not-loaded", "max_context_length": 131072},
    {"id": "text-embedding-nomic", "object": "model", "type": "embeddings",
     "state": "loaded"},
]}
# Ollama's /api/ps, the documented example plus a Hub pull
OLLAMA_PS = {"models": [
    {"name": "mistral:latest", "size": 5137025024, "size_vram": 5137025024,
     "details": {"format": "gguf", "family": "llama", "parameter_size": "7.2B",
                 "quantization_level": "Q4_0"}},
    {"name": "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M", "size": 3_900_000_000,
     "size_vram": 3_600_000_000, "context_length": 4096,
     "details": {"quantization_level": "Q4_K_M"}},
]}


@pytest.fixture(autouse=True)
def _homes(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGHTSIZE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("RIGHTSIZE_DATA_HOME", str(tmp_path / "data"))


def _facts(name: str, **extra) -> ModelFacts:
    fx = ModelFacts.model_validate_json((FIXTURES / name).read_text(encoding="utf-8"))
    return fx.model_copy(update={"extra": {**fx.extra, **extra}})


def _client(payload: dict) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))


# ---------------------------------------------------------------- consent


def test_recording_is_off_until_turned_on() -> None:
    assert telemetry.status()["enabled"] is False
    telemetry.enable()
    assert telemetry.enabled() and telemetry.status()["since"]
    telemetry.disable()
    assert not telemetry.enabled()


def test_nothing_is_written_while_off() -> None:
    record = rec.build(source="manual", device=RTX, runtime="llama.cpp", facts=None,
                       quant="Q4_K_M", ctx=8192, predicted=None, measured_vram_gb=5.0)
    assert rec.append(record) is False and rec.read() == []
    telemetry.enable()
    assert rec.append(record) is True and len(rec.read()) == 1


def test_the_consent_copy_names_what_is_never_recorded() -> None:
    for word in ("paths", "host names", "prompts", "private models", "Nothing is sent"):
        assert word in telemetry.WHAT_IS_RECORDED


# ---------------------------------------------------------------- what a record may hold


def test_a_record_carries_no_local_names_or_private_repos() -> None:
    """A device named after its host and a model from a private repo go in; neither comes
    out. Only fields on the allowlist can be set at all."""
    local = Device(name=r"DESKTOP-7QK2 C:\Users\jdoe\gpu", vendor="nvidia",
                   memory_gib=16, os="windows")
    private = _facts("Qwen__Qwen3-4B.json", private=True)
    record = rec.build(source="manual", device=local, runtime="llama.cpp", facts=private,
                       quant="Q4_K_M", ctx=8192, predicted=None, measured_vram_gb=3.1)
    text = record.model_dump_json()
    assert "jdoe" not in text.lower() and "desktop" not in text.lower() and "\\\\" not in text
    assert record.model.repo is None, "private repo: no id"
    assert record.device.name == "nvidia 16 GiB"
    with pytest.raises(ValueError):
        rec.RecordModel(repo=None, path="C:/models/x.gguf")  # unknown field refused


def test_a_public_repo_and_a_known_card_are_kept() -> None:
    record = rec.build(source="manual", device=RTX, runtime="llama.cpp",
                       facts=_facts("Qwen__Qwen3-4B.json", private=False), quant="Q4_K_M",
                       ctx=8192, predicted=None, measured_vram_gb=3.1)
    assert record.model.repo == "Qwen/Qwen3-4B" and record.device.name == RTX.name


def test_export_writes_what_show_reads(tmp_path) -> None:
    telemetry.enable()
    rec.append(rec.build(source="manual", device=RTX, runtime="llama.cpp", facts=None,
                         quant=None, ctx=None, predicted=None, measured_vram_gb=1.0))
    out = tmp_path / "mine.jsonl"
    assert rec.export(out) == 1
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["measured"]["vram_gb"] == 1.0


# ---------------------------------------------------------------- recorders


def test_lm_studio_lists_loaded_chat_models_only() -> None:
    got = sources.lm_studio(client=_client(LM_STUDIO_MODELS))
    assert [(m.name, m.quant, m.ctx, m.hub_repo) for m in got] == [
        ("openai/gpt-oss-20b", "MXFP4", 8192, "openai/gpt-oss-20b")
    ]


def test_ollama_reports_vram_and_hub_pulls_name_their_repo() -> None:
    got = sources.ollama(client=_client(OLLAMA_PS))
    assert got[0].vram_gb == pytest.approx(5.137, abs=0.001) and got[0].hub_repo is None
    assert got[1].hub_repo == "Qwen/Qwen3-4B-GGUF" and got[1].ctx == 4096


def test_a_runtime_that_is_not_running_gives_nothing() -> None:
    def refuse(request):
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    assert sources.lm_studio(client=client) == [] and sources.ollama(client=client) == []


def test_process_memory_parsers() -> None:
    smi = ["/usr/bin/llama-server, 11191", r"C:\Windows\explorer.exe, [N/A]"]
    assert sources.parse_nvidia_smi(smi) == [("llama-server", pytest.approx(11.735, abs=0.01))]
    counters = [
        "pid_46820_luid_0x00000000_0x0000c8ce_phys_0=11734614016",
        "pid_16424_luid_0x00000000_0x0000c8ce_phys_0=2149731844915",  # a browser, 2 TB: noise
        "pid_2192_luid_0x00000000_0x0000c8ce_phys_0=0",
    ]
    names = {46820: "llama-server", 16424: "firefox"}
    assert sources.parse_windows_counters(counters, names, limit_gb=17.2) == [
        ("llama-server", pytest.approx(11.73, abs=0.01))
    ]


# ---------------------------------------------------------------- calibrate


def test_calibrate_compares_and_records_when_allowed(monkeypatch) -> None:
    monkeypatch.setattr(sources, "process_vram", lambda name, limit_gb=None: [11.735])
    lms = sources.lm_studio(client=_client(LM_STUDIO_MODELS))
    facts = _facts("openai__gpt-oss-20b.json", private=False)
    rows = calibrate(RTX, lm_studio=lms, ollama=[], facts_for=lambda repo: facts)
    assert rows[0].predicted_gb and rows[0].measured_gb == pytest.approx(11.735)
    assert rows[0].error is not None and abs(rows[0].error) < 0.2
    assert rec.read() == [], "recording off: compared, not written"
    telemetry.enable()
    calibrate(RTX, lm_studio=lms, ollama=[], facts_for=lambda repo: facts)
    assert rec.read()[0].model.repo == "openai/gpt-oss-20b"


def test_two_lm_studio_models_are_not_given_one_reading(monkeypatch) -> None:
    monkeypatch.setattr(sources, "process_vram", lambda name, limit_gb=None: [11.7])
    two = [sources.Loaded("lm studio", n, "Q4_K_M", 4096, None, None) for n in ("a/x", "b/y")]
    rows = calibrate(RTX, lm_studio=two, ollama=[], facts_for=lambda repo: None)
    assert all(r.measured_gb is None for r in rows)


# ---------------------------------------------------------------- refit


def _refit():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import refit_constants

    return refit_constants


def test_refit_recovers_known_constants() -> None:
    """Records made with overhead = 0.4 GB + 5% of weights: the fit finds those."""
    refit = _refit()
    telemetry.enable()
    for w in (2.0, 5.0, 9.0, 14.0, 20.0):
        record = rec.build(source="manual", device=RTX, runtime="llama.cpp", facts=None,
                           quant="Q4_K_M", ctx=8192, predicted=None,
                           measured_vram_gb=w + 0.3 + 0.4 + 0.05 * w)
        record.predicted = rec.RecordNumbers(vram_gb=w + 0.3 + 0.75 + 0.02 * w,
                                             weights_gb=w, kv_gb=0.3)
        rec.append(record)
    (proposal,) = refit.refit(rec.read(), {"llama.cpp": (0.75, 0.02)})
    assert proposal["new"] == (pytest.approx(0.4, abs=0.01), pytest.approx(0.05, abs=0.001))
    assert proposal["error_after"] < proposal["error_before"]


def test_refit_writes_only_with_enough_records(tmp_path) -> None:
    refit = _refit()
    data = tmp_path / "overheads.yaml"
    data.write_text(Path(refit.DATA).read_text(encoding="utf-8"), encoding="utf-8")
    few = [{"runtime": "lm studio", "records": 1, "new": (0.18, 0.02),
            "error_before": 0.05, "error_after": 0.0}]
    assert refit.write(few, data) == []
    many = [{**few[0], "records": 6}]
    assert refit.write(many, data) == ["lm studio"]
    text = data.read_text(encoding="utf-8")
    assert "fixed_gb: 0.18" in text and "refit from 6 records" in text
    assert "fixed_gb: 0.75" in text, "other runtimes untouched"
