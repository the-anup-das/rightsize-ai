"""Published quantizations of a model, found through the Hub's base-model links (F1)."""

from __future__ import annotations

import json

import httpx
import pytest

from rightsize.catalog import variants
from rightsize.catalog.variants import classify, matches_base, quant_of

BASE = "Qwen/Qwen3-4B"
LISTED = [  # as /api/models?filter=base_model:quantized:Qwen/Qwen3-4B returns them
    {"id": "unsloth/Qwen3-4B-GGUF", "downloads": 462_007, "library_name": "transformers",
     "tags": ["transformers", "gguf", "qwen3", "unsloth", f"base_model:quantized:{BASE}"]},
    {"id": "tuner/Qwen3-4B-Roleplay-v2-GGUF", "downloads": 424_827,
     "library_name": "llama.cpp", "tags": ["llama.cpp", "gguf", "qlora"]},
    {"id": "Qwen/Qwen3-4B-GGUF", "downloads": 404_833, "tags": ["gguf"]},
    {"id": "Qwen/Qwen3-4B-AWQ", "downloads": 245_475, "library_name": "transformers",
     "tags": ["transformers", "safetensors", "4-bit", "awq"]},
    {"id": "unsloth/Qwen3-4B-unsloth-bnb-4bit", "downloads": 60_411,
     "library_name": "transformers", "tags": ["4-bit", "bitsandbytes"]},
    {"id": "lmstudio-community/Qwen3-4B-MLX-4bit", "downloads": 24_348, "library_name": "mlx",
     "tags": ["mlx", "4-bit"]},
    {"id": "someone/Qwen3-4B-q4f16_1-MLC", "downloads": 10, "library_name": "mlc-llm",
     "tags": []},
]
FILES = {
    "unsloth/Qwen3-4B-GGUF": {"Qwen3-4B-Q4_K_M.gguf": 2_497_281_312,
                              "Qwen3-4B-UD-Q4_K_XL.gguf": 2_550_000_000, "README.md": 1},
    "Qwen/Qwen3-4B-GGUF": {"Qwen3-4B-Q4_K_M.gguf": 2_497_280_000,
                           "Qwen3-4B-Q8_0.gguf": 4_280_000_000},
    "tuner/Qwen3-4B-Roleplay-v2-GGUF": {"Qwen3-4B-Roleplay-v2-Q4_K_M.gguf": 2_500_000_000},
    "Qwen/Qwen3-4B-AWQ": {"model.safetensors": 2_670_000_000, "config.json": 1},
    "unsloth/Qwen3-4B-unsloth-bnb-4bit": {"model.safetensors": 3_550_000_000},
    "lmstudio-community/Qwen3-4B-MLX-4bit": {"model.safetensors": 2_260_000_000},
    "someone/Qwen3-4B-q4f16_1-MLC": {"params_shard_0.bin": 32_000_000},
}


def _hub(seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        path = request.url.path
        if path == "/api/models":
            return httpx.Response(200, json=LISTED)
        repo = path.removeprefix("/api/models/")
        if repo == "unsloth/Qwen3-4B-GGUF" and request.url.params.get("blobs") != "true":
            return httpx.Response(200, json={"tags": [f"base_model:quantized:{BASE}"]})
        if repo == BASE and request.url.params.get("blobs") != "true":
            return httpx.Response(200, json={"tags": ["text-generation"]})
        if repo in FILES:
            return httpx.Response(200, json={"siblings": [
                {"rfilename": f, "size": n} for f, n in FILES[repo].items()]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_each_published_quantization_is_found_and_sized() -> None:
    found = variants(BASE, transport=_hub(), base_params=4_022_468_096)
    by = {(v.ref.repo, v.quant): v for v in found}
    q4 = by[("Qwen/Qwen3-4B-GGUF", "Q4_K_M")]
    assert q4.format == "gguf" and q4.official and q4.ref.file == "Qwen3-4B-Q4_K_M.gguf"
    assert q4.bits_per_weight == pytest.approx(4.97, abs=0.01)
    assert ("unsloth/Qwen3-4B-GGUF", "UD-Q4_K_XL") in by, "one variant per GGUF file"
    assert by[("Qwen/Qwen3-4B-AWQ", None)].format == "awq"
    assert by[("unsloth/Qwen3-4B-unsloth-bnb-4bit", "4bit")].format == "bnb"
    mlx = by[("lmstudio-community/Qwen3-4B-MLX-4bit", "4bit")]
    assert mlx.format == "mlx" and mlx.runtimes == ["mlx-lm", "lm studio"]
    assert by[("someone/Qwen3-4B-q4f16_1-MLC", "q4f16_1")].format == "mlc", "by library name"


def test_official_then_known_publishers_come_first_within_a_format() -> None:
    found = variants(BASE, transport=_hub(), base_params=4_022_468_096)
    gguf_repos = [v.ref.repo for v in found if v.format == "gguf"]
    assert gguf_repos[0] == "Qwen/Qwen3-4B-GGUF", "official, though fewer downloads"
    assert gguf_repos.index("unsloth/Qwen3-4B-GGUF") < gguf_repos.index(
        "tuner/Qwen3-4B-Roleplay-v2-GGUF")
    assert [v.format for v in found] == sorted(
        (v.format for v in found),
        key=["gguf", "mlx", "compressed-tensors", "modelopt", "awq", "gptq", "bnb", "nvfp4",
             "fp8", "torchao", "exl3", "exl2", "mlc", "litert", "openvino", "onnx"].index)


def test_a_fine_tune_calling_itself_a_quantization_is_flagged() -> None:
    found = variants(BASE, transport=_hub(), base_params=4_022_468_096)
    tuned = [v for v in found if v.publisher == "tuner"]
    assert tuned and not any(v.name_matches_base for v in tuned)
    assert all(v.name_matches_base for v in found if v.publisher != "tuner")


def test_a_quantized_repo_leads_back_to_its_base() -> None:
    seen: list[httpx.Request] = []
    variants("unsloth/Qwen3-4B-GGUF", transport=_hub(seen), files=False)
    listing = next(r for r in seen if r.url.path == "/api/models")
    assert listing.url.params.get("filter") == f"base_model:quantized:{BASE}"


def test_one_format_is_narrowed_on_the_hub() -> None:
    seen: list[httpx.Request] = []
    found = variants(BASE, formats=["gguf"], transport=_hub(seen), files=False)
    listing = next(r for r in seen if r.url.path == "/api/models")
    assert listing.url.params.get_list("filter") == [f"base_model:quantized:{BASE}", "gguf"]
    assert {v.format for v in found} == {"gguf"}
    assert not any(r.url.params.get("blobs") for r in seen), "files=False lists no files"


def test_variants_are_cached_and_served_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path))
    first = variants(BASE, transport=_hub(), use_cache=True, base_params=1)
    again = variants(BASE, offline=True)
    assert again == first
    with pytest.raises(FileNotFoundError, match="offline"):
        variants("Qwen/Qwen3-8B", offline=True)


@pytest.mark.parametrize(("repo", "tags", "library", "fmt", "quant"), [
    ("RedHatAI/Qwen3-8B-quantized.w4a16", ["compressed-tensors"], None,
     "compressed-tensors", "w4a16"),
    ("RedHatAI/Qwen3-8B-FP8-dynamic", ["compressed-tensors", "fp8"], None,
     "compressed-tensors", "FP8-dynamic"),
    ("nvidia/Qwen3-8B-NVFP4", ["modelopt", "nvfp4"], "Model Optimizer", "modelopt", "NVFP4"),
    ("x/Qwen3-8B-GPTQ-Int4", ["gptq"], "transformers", "gptq", "Int4"),
    ("x/Qwen3-8B-exl3_4.0bpw", ["exl3"], None, "exl3", "4.0bpw"),
    ("litert-community/Qwen3-8B", [], "litert-lm", "litert", None),
    ("x/Qwen3-8B-something", [], None, None, None),
])
def test_formats_are_told_by_tag_library_then_name(repo, tags, library, fmt, quant) -> None:
    f = classify(repo, tags, library)
    assert (f["id"] if f else None) == fmt
    assert quant_of(f, repo) == quant


@pytest.mark.parametrize(("repo", "base", "same"), [
    ("bartowski/Qwen_Qwen3-4B-GGUF", "Qwen/Qwen3-4B", True),
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
     "meta-llama/Llama-3.1-8B-Instruct", True),
    ("mlc-ai/Qwen3-8B-q4f16_1-MLC", "Qwen/Qwen3-8B", True),
    ("async0x42/Qwen3-8B-exl3_4.0bpw", "Qwen/Qwen3-8B", True),
    ("unsloth/DeepSeek-V3-0324-GGUF", "deepseek-ai/DeepSeek-V3-0324", True),
    ("mobilint/EAGLE3-Qwen3-8B", "Qwen/Qwen3-8B", False),
    ("unsloth/Qwen3-8B-128K-GGUF", "Qwen/Qwen3-8B", False),
    ("google/gemma-3-4b-it-qat-q4_0-gguf", "google/gemma-3-4b-it", False),
])
def test_names_are_compared_without_their_format_words(repo, base, same) -> None:
    assert matches_base(repo, base) is same


def test_the_cli_lists_them_and_hides_look_alikes(monkeypatch, capsys) -> None:
    import rightsize.catalog as catalog
    from rightsize.cli import main

    found = variants(BASE, transport=_hub(), base_params=4_022_468_096)
    monkeypatch.setattr(catalog, "variants", lambda *a, **k: found)
    assert main(["variants", BASE]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-4B-GGUF" in out and "Roleplay" in out.split("more whose names")[1]
    assert main(["--json", "variants", BASE, "--all"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(found)


def test_an_offline_miss_is_one_line_not_a_traceback(capsys) -> None:
    from rightsize.cli import main

    assert main(["--offline", "variants", "nobody/nothing"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("rightsize variants: no cached variants for nobody/nothing")
