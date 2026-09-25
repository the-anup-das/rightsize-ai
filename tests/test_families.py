"""Diffusion, audio, vision and embedding estimators against published measurements (F3).

Every row in data/runtimes/diffusers/memory.yaml and data/runtimes/whisper/memory.yaml is
replayed here with the facts the catalog recorded for that model, so a change to the
formulas, the constants or the catalog that moves a prediction outside the error the row
admits to fails a test that names the measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightsize._data import load_yaml
from rightsize.errors import NotImplementedYet
from rightsize.fit import estimate
from rightsize.fit import formats as fmt
from rightsize.fit.diffusion import constants
from rightsize.types import GB, GIB, Device, Family, ModelFacts, Verdict

FIXTURES = Path(__file__).parent / "fixtures" / "facts"
H100 = Device(name="H100", vendor="nvidia", memory_gib=80)
RTX_3070_TI = Device(name="RTX 3070 Ti", vendor="nvidia", memory_gib=8)
I7_12700K = Device(name="i7-12700K", vendor="cpu", memory_gib=64)


def _facts(repo: str) -> ModelFacts:
    path = FIXTURES / (repo.replace("/", "__") + ".json")
    return ModelFacts.model_validate_json(path.read_text(encoding="utf-8"))


DIFFUSION_ROWS = load_yaml("runtimes/diffusers/memory.yaml")["measurements"]
WHISPER_ROWS = load_yaml("runtimes/whisper/memory.yaml")["measurements"]["rows"]


# ---------------------------------------------------------------- formats


@pytest.mark.parametrize(
    ("name", "bpw"),
    [("nf4", 4.5), ("nf4-dq", 4.127), ("int4", 4.25), ("awq", 4.156), ("gptq", 4.156),
     ("mlx-4bit", 4.5), ("mlx-8bit", 8.5), ("nvfp4", 4.5), ("mxfp4", 4.25), ("fp8", 8.0),
     ("int8", 8.0), ("bf16", 16.0), ("fp32", 32.0)],
)
def test_format_bits_follow_their_block_layout(name: str, bpw: float) -> None:
    assert fmt.bpw(name) == pytest.approx(bpw)


def test_format_names_match_through_aliases_and_case() -> None:
    assert fmt.lookup("bnb-4bit").id == "nf4"
    assert fmt.lookup("BF16").id == "bf16"
    assert fmt.lookup("float8_e4m3fn").id == "fp8"


def test_gguf_names_use_llama_cpp_or_the_tensor_type() -> None:
    """An LLM's Q2_K file mixes in larger types; a diffusion GGUF applies Q2_K throughout.
    city96's FLUX.1-dev Q2_K file is 2.71 bpw: the tensor figure (2.625) is 3% off, the
    LLM figure (3.16) 16%."""
    assert fmt.bpw("Q2_K") > 3.0
    assert fmt.bpw("Q2_K", gguf="tensor") == pytest.approx(2.625)
    assert fmt.bpw("Q4_K_S", gguf="tensor") == pytest.approx(4.5)


def test_an_unknown_format_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="nf4"):
        fmt.lookup("int3-magic")


# ---------------------------------------------------------------- diffusion


@pytest.mark.parametrize("row", DIFFUSION_ROWS, ids=[r["id"] for r in DIFFUSION_ROWS])
def test_diffusion_reproduces_published_peaks(row: dict) -> None:
    r = estimate(
        _facts(row["model"]), row["quant"] or None, H100, offload=row["offload"],
        resolution=tuple(row["resolution"]), batch=row["batch"],
        vae_slicing=row.get("vae_slicing") or None,
    )
    # the published counters are torch's, which never see the CUDA context
    predicted = (r.vram_gb - constants()["cuda_context_gb"]) * GB / GIB
    error = predicted / row["peak_gib"] - 1
    assert abs(error) <= row["tolerance"], f"{row['id']}: {error:+.1%}"


PREQUANTIZED = [
    r for r in DIFFUSION_ROWS
    if r.get("after_loading_gib") and r["offload"] == "none"
    and all(f.upper().startswith("Q") or f == "bf16" for f in r["quant"].values())
]


@pytest.mark.parametrize("row", PREQUANTIZED, ids=[r["id"] for r in PREQUANTIZED])
def test_diffusion_weights_match_memory_after_loading(row: dict) -> None:
    """Without quantization at load time, memory after loading is the weights. BF16 and
    the GGUF rows check the weight arithmetic alone, out of sample: nothing was fitted."""
    r = estimate(_facts(row["model"]), row["quant"] or None, H100)
    weights = r.breakdown["weights"] * GB / GIB
    assert weights == pytest.approx(row["after_loading_gib"], rel=0.05)


def test_flux_components_from_a_gated_repo() -> None:
    """FLUX.1-dev is gated: the files are listed but not served, so the catalog counted
    each component from its size. The post states the BF16 sizes: 23.8, 9.52, 0.246, 0.168."""
    parts = _facts("black-forest-labs/FLUX.1-dev").extra["pipeline"]["components"]
    gb = {name: c["params"] * 2 / 1e9 for name, c in parts.items()}
    assert gb["transformer"] == pytest.approx(23.8, rel=0.01)
    assert gb["text_encoder_2"] == pytest.approx(9.52, rel=0.01)
    assert gb["text_encoder"] == pytest.approx(0.246, rel=0.01)
    assert gb["vae"] == pytest.approx(0.168, rel=0.01)


def test_offloading_lowers_the_gpu_peak_and_moves_weights_to_ram() -> None:
    fx = _facts("black-forest-labs/FLUX.1-dev")
    card = Device(name="RTX 4070", vendor="nvidia", memory_gib=12, system_ram_gib=64)
    none, model, seq = (estimate(fx, "bf16", card, offload=o)
                        for o in ("none", "model", "sequential"))
    assert none.vram_gb > model.vram_gb > seq.vram_gb
    assert none.ram_gb == 0 and model.ram_gb == pytest.approx(none.breakdown["weights"], rel=0.01)
    assert none.verdict is Verdict.offload
    assert any("enable_sequential_cpu_offload" in n for n in none.notes)


def test_offloading_saves_nothing_on_unified_memory() -> None:
    fx = _facts("black-forest-labs/FLUX.1-dev")
    mac = Device(name="M4 Pro", vendor="apple", memory_gib=24, unified_memory=True)
    r = estimate(fx, "bf16", mac, offload="model")
    assert r.vram_gb >= r.breakdown["weights"]
    assert any("unified memory" in n for n in r.notes)


def test_bitsandbytes_loads_every_quantized_component_before_offloading() -> None:
    r = estimate(_facts("black-forest-labs/FLUX.1-dev"), "nf4", H100, offload="model",
                 text_encoder_quant="nf4")
    assert r.breakdown["phase.load"] == max(
        v for k, v in r.breakdown.items() if k.startswith("phase.")
    )


def test_guidance_doubles_unet_samples_but_not_flux() -> None:
    sdxl = estimate(_facts("stabilityai/stable-diffusion-xl-base-1.0"), None, H100)
    flux = estimate(_facts("black-forest-labs/FLUX.1-dev"), None, H100)
    assert any("classifier-free guidance" in n for n in sdxl.notes)
    assert not any("classifier-free guidance" in n for n in flux.notes)


def _act(r, phase: str) -> float:
    """A phase's activations: with nothing offloaded, every phase also holds all weights."""
    return r.breakdown[f"phase.{phase}"] - r.breakdown["weights"]


def test_vae_slicing_and_resolution_move_the_decode() -> None:
    fx = _facts("stabilityai/stable-diffusion-xl-base-1.0")
    one = estimate(fx, None, H100)
    four = estimate(fx, None, H100, batch=4)
    sliced = estimate(fx, None, H100, batch=4, vae_slicing=True)
    small = estimate(fx, None, H100, resolution=(512, 512))
    assert _act(four, "decode") == pytest.approx(4 * _act(one, "decode"), rel=0.01)
    assert _act(sliced, "decode") == pytest.approx(_act(one, "decode"), rel=0.01)
    assert _act(small, "decode") == pytest.approx(_act(one, "decode") / 4, rel=0.01)


def test_video_counts_frames_and_says_it_is_less_sure() -> None:
    fx = _facts("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    image = estimate(fx, None, H100, resolution=(832, 480))
    video = estimate(fx, None, H100, resolution=(832, 480), frames=81)
    assert _act(video, "denoise") > 10 * _act(image, "denoise")
    assert video.confidence < image.confidence


def test_diffusion_takes_a_mapping_per_component() -> None:
    r = estimate(_facts("black-forest-labs/FLUX.1-dev"),
                 {"transformer": "Q8_0", "text_encoder_2": "nf4"}, H100)
    assert r.breakdown["weights.transformer"] == pytest.approx(11.9e9 * 8.5 / 8 / 1e9, rel=0.01)
    assert r.breakdown["weights.text_encoder_2"] < 3.0


# ---------------------------------------------------------------- audio


@pytest.mark.parametrize(
    "row", WHISPER_ROWS,
    ids=[f"{r['runtime']}-{r['device']}-{r['precision']}-b{r['batch']}" for r in WHISPER_ROWS],
)
def test_whisper_reproduces_the_benchmark(row: dict) -> None:
    fx = _facts(row["model"])
    device = RTX_3070_TI if row["device"] == "gpu" else I7_12700K
    r = estimate(fx, row["precision"], device, runtime=row["runtime"], batch=row["batch"])
    assert r.vram_gb * 1000 == pytest.approx(row["memory_mb"], rel=0.05)


@pytest.mark.parametrize(
    ("repo", "quant", "file_bytes"),
    [
        ("openai/whisper-large-v3", "f16", 3_095_033_483),
        ("openai/whisper-large-v3", "q5_0", 1_081_140_203),
        ("openai/whisper-large-v3-turbo", "q5_0", 574_041_195),
        ("openai/whisper-large-v3-turbo", "q8_0", 874_188_075),
        ("openai/whisper-small", "q8_0", 264_464_607),
        ("openai/whisper-small", "q5_1", 190_085_487),
    ],
)
def test_whisper_cpp_weights_match_the_published_files(repo, quant, file_bytes) -> None:
    """ggerganov/whisper.cpp's ggml files, sizes from the Hub. The F3 plan's golden
    "large-v3 q5_0 = 547 MB" was the turbo file in MiB (574 MB); large-v3 q5_0 is 1.08 GB."""
    r = estimate(_facts(repo), quant, RTX_3070_TI, runtime="whisper.cpp")
    assert r.breakdown["weights"] * 1e9 == pytest.approx(file_bytes, rel=0.05)


def test_audio_runtime_follows_the_device_and_the_format() -> None:
    fx = _facts("openai/whisper-large-v3")
    mac = Device(name="M4", vendor="apple", memory_gib=16, unified_memory=True)
    assert "whisper.cpp" in estimate(fx, None, mac).notes[0]
    assert "faster-whisper" in estimate(fx, "int8", RTX_3070_TI).notes[0]
    assert "whisper.cpp" in estimate(fx, "q5_0", RTX_3070_TI).notes[0]


def test_non_whisper_audio_is_weights_plus_an_allowance() -> None:
    fx = _facts("openai/whisper-small").model_copy(update={"extra": {"model_type": "kokoro"}})
    r = estimate(fx, "fp32", RTX_3070_TI)
    assert r.formula_id == "audio.weights.v0" and r.confidence <= 0.3


# ---------------------------------------------------------------- vision and embeddings


def test_embedding_counts_a_batch_of_tokens() -> None:
    fx = _facts("BAAI/bge-m3")
    assert fx.family is Family.embedding
    small = estimate(fx, None, RTX_3070_TI, batch=1)
    big = estimate(fx, None, RTX_3070_TI, batch=64, seq_len=8192)
    assert big.breakdown["activations"] > 100 * small.breakdown["activations"]
    assert any("32x" in n for n in small.notes)


def test_vit_tokens_come_from_image_and_patch_size() -> None:
    r = estimate(_facts("google/vit-base-patch16-224"), None, RTX_3070_TI)
    assert any("197 patches" in n for n in r.notes)  # (224/16)^2 + the class token


# ---------------------------------------------------------------- routing


def test_each_family_gets_its_estimator_and_ignores_other_arguments() -> None:
    ids = {
        repo: estimate(_facts(repo), None, H100, ctx=4096, runtime=None, offload="none").formula_id
        for repo in ("black-forest-labs/FLUX.1-dev", "openai/whisper-small", "BAAI/bge-m3",
                     "google/vit-base-patch16-224", "Qwen/Qwen3-4B")
    }
    assert ids == {
        "black-forest-labs/FLUX.1-dev": "diffusion.components.v0",
        "openai/whisper-small": "audio.whisper.v0",
        "BAAI/bge-m3": "encoder.weights.v0",
        "google/vit-base-patch16-224": "encoder.weights.v0",
        "Qwen/Qwen3-4B": "llm.gguf.analytic.v0",
    }


def test_fine_tuning_other_families_says_it_is_not_modelled() -> None:
    with pytest.raises(NotImplementedYet):
        estimate(_facts("black-forest-labs/FLUX.1-dev"), None, H100, mode="lora")


def test_cli_estimate_for_a_diffusion_pipeline(monkeypatch, capsys) -> None:
    import rightsize.catalog as catalog
    from rightsize.cli import main

    monkeypatch.setattr(catalog, "facts", lambda repo, rev="main": _facts(repo))
    assert main(["--json", "estimate", "black-forest-labs/FLUX.1-dev", "--device",
                 "RTX 4070 12GB", "--quant", "nf4", "--offload", "none", "--offload",
                 "model"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["results"]) == {"nf4/none", "nf4/model"}
    assert out["results"]["nf4/model"]["vram_gb"] < out["results"]["nf4/none"]["vram_gb"]
