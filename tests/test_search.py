"""Curated model lists and what counts as one copy of a model's weights (F1)."""

from __future__ import annotations

import json

import httpx
import pytest

from rightsize.catalog import curated, facts, search
from rightsize.catalog.weights import pick_copy

CANDIDATES = {"models": [
    {"repo": "Qwen/Qwen3-8B", "publisher": "Qwen", "tasks": ["chat"], "created_at": "2025-04-27",
     "downloads_30d": 900, "license": "apache-2.0",
     "facts": {"params_total": 8_190_735_360}},
    {"repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "publisher": "Qwen", "tasks": ["coding"],
     "created_at": "2024-09-17", "downloads_30d": 700, "license": "apache-2.0",
     "facts": {"params_total": 7_615_616_512}},
]}
FAMILIES = {
    "tasks": {"diffusion": ["text-to-image", "text-to-video"],
              "audio": ["automatic-speech-recognition", "text-to-speech"],
              "vision": ["object-detection"], "embedding": ["sentence-similarity"]},
    "models": [
        {"repo": "black-forest-labs/FLUX.1-dev", "family": "diffusion", "tasks": ["text-to-image"],
         "publisher": "black-forest-labs", "params_total": 16_871_000_000, "license": "other",
         "gated": True, "created_at": "2024-07-31", "downloads_30d": 800},
        {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "family": "diffusion",
         "tasks": ["text-to-image"], "publisher": "stabilityai", "params_total": 3_469_000_000,
         "license": "openrail++", "gated": False, "created_at": "2023-07-25",
         "downloads_30d": 2000},
        {"repo": "openai/whisper-small", "family": "audio",
         "tasks": ["automatic-speech-recognition"], "publisher": "openai",
         "params_total": 241_734_912, "license": "apache-2.0", "gated": False,
         "created_at": "2022-09-26", "downloads_30d": 1500},
    ],
}


@pytest.fixture(autouse=True)
def _lists(monkeypatch):
    docs = {"models/candidates.yaml": CANDIDATES, "models/families.yaml": FAMILIES}
    monkeypatch.setattr(curated, "load_yaml", lambda rel: docs[rel])


def test_a_family_and_a_size_bound() -> None:
    found = search("diffusion", max_params=13e9)
    assert [e.repo for e in found] == ["stabilityai/stable-diffusion-xl-base-1.0"]


def test_a_task_implies_its_family() -> None:
    assert [e.repo for e in search(task="coding")] == ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert [e.repo for e in search(task="automatic-speech-recognition")] == [
        "openai/whisper-small"]


def test_most_downloaded_first_and_the_other_filters() -> None:
    assert [e.repo for e in search("diffusion")] == [
        "stabilityai/stable-diffusion-xl-base-1.0", "black-forest-labs/FLUX.1-dev"]
    assert [e.repo for e in search("diffusion", include_gated=False)] == [
        "stabilityai/stable-diffusion-xl-base-1.0"]
    assert [e.repo for e in search(license="apache-2.0", min_params=1e9)] == [
        "Qwen/Qwen3-8B", "Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert [e.repo for e in search(publisher="OPENAI")] == ["openai/whisper-small"]
    assert len(search(limit=2)) == 2


def test_the_cli_lists_tasks_and_results(capsys) -> None:
    from rightsize.cli import main

    assert main(["search", "--task", "list"]) == 0
    assert "text-to-image" in capsys.readouterr().out
    assert main(["--json", "search", "--family", "audio"]) == 0
    assert [e["repo"] for e in json.loads(capsys.readouterr().out)] == ["openai/whisper-small"]


# ---------------------------------------------------------------- one copy of the weights


def test_the_same_weights_sharded_twice_count_once() -> None:
    """LTX-2.5's transformer ships a 4-shard and an 8-shard set of the same 19B weights;
    counting both made it 38B."""
    sizes = {f"transformer/m-0000{i}-of-00004.safetensors": 9_500_000_000 for i in range(1, 5)}
    sizes |= {f"transformer/m-0000{i}-of-00008.safetensors": 4_750_000_000 for i in range(1, 9)}
    files, variant, others = pick_copy(list(sizes), sizes)
    assert len(files) == 4 and variant is None and others == [], "a resharding is no choice"


def test_several_checkpoints_at_the_root_are_one_choice_among_them() -> None:
    """LTX-2.3 keeps dev, distilled and a LoRA side by side; adding them made a 22B model
    78B. One is counted and the rest are named."""
    sizes = {"ltx-2.3-22b-dev.safetensors": 46_150_000_123,
             "ltx-2.3-22b-distilled.safetensors": 46_150_000_500,
             "ltx-2.3-22b-distilled-lora-384.safetensors": 7_610_000_000}
    files, _, others = pick_copy(list(sizes), sizes)
    assert files == ["ltx-2.3-22b-dev.safetensors"], "near-equal sizes tie; the shorter name"
    assert len(others) == 2


def test_a_config_with_per_stage_lists_still_gives_facts() -> None:
    """SegFormer's config lists heads per stage ([1, 2, 5, 8]); facts() used to fail on it."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/nvidia/segformer-b0":
            return httpx.Response(200, json={
                "pipeline_tag": "image-segmentation",
                "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.safetensors"}],
                "safetensors": {"parameters": {"F32": 3_753_206}},
            })
        if request.url.path.endswith("/config.json"):
            return httpx.Response(200, json={"model_type": "segformer",
                                             "num_attention_heads": [1, 2, 5, 8],
                                             "hidden_sizes": [32, 64, 160, 256]})
        return httpx.Response(404)

    fx = facts("nvidia/segformer-b0", transport=httpx.MockTransport(handler))
    assert fx.family.value == "vision" and fx.params_total == 3_753_206
    assert fx.num_kv_heads is None and fx.head_dim is None
