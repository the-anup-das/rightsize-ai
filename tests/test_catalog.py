"""Hub catalog against a fake transport: no network, header-only reads."""

from __future__ import annotations

import json
import struct

import httpx
import pytest

from rightsize.catalog import facts, safetensors_header

CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 36,
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}


def _safetensors_bytes(tensors: dict[str, tuple[str, list[int]]]) -> bytes:
    header: dict = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * 2
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * offset


SHARD = _safetensors_bytes(
    {
        "model.embed_tokens.weight": ("BF16", [1000, 2560]),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [2560, 2560]),
        "model.norm.weight": ("F32", [2560]),
    }
)
EXPECTED_PARAMS = 1000 * 2560 + 2560 * 2560 + 2560


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/models/Qwen/Qwen3-4B":
        return httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "config.json"},
                    {"rfilename": "model.safetensors"},
                    {"rfilename": "README.md"},
                ],
                "pipeline_tag": "text-generation",
                "cardData": {"license": "apache-2.0"},
                "gated": False,
            },
        )
    if path.endswith("/config.json"):
        return httpx.Response(200, json=CONFIG)
    if path.endswith("/model.safetensors"):
        rng = request.headers["Range"].split("=")[1]
        a, b = (int(x) for x in rng.split("-"))
        assert b - a + 1 <= 8 or a == 8, "only header ranges may be requested"
        return httpx.Response(206, content=SHARD[a : b + 1])
    return httpx.Response(404)


def test_safetensors_header_reads_only_the_header() -> None:
    with httpx.Client(transport=httpx.MockTransport(_handler)) as c:
        hdr = safetensors_header(
            c, "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/model.safetensors"
        )
    assert set(hdr) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    }


def test_facts_from_headers_and_config() -> None:
    fx = facts("Qwen/Qwen3-4B", transport=httpx.MockTransport(_handler))
    assert fx.params_total == EXPECTED_PARAMS
    assert fx.dtype == "BF16"
    assert (fx.num_layers, fx.num_kv_heads, fx.head_dim, fx.context_max) == (36, 8, 128, 40960)
    assert fx.family.value == "llm"
    assert fx.license == "apache-2.0"
    assert fx.confidence == 1.0
    assert fx.extra["params_by_dtype"] == {"BF16": EXPECTED_PARAMS - 2560, "F32": 2560}


def test_hub_parameter_summary_spares_the_shard_headers() -> None:
    """The model-info response already counts parameters per dtype. Reading every shard
    header instead cost DeepSeek-V3 163 requests; with the summary there are none."""
    shard_reads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/Qwen/Qwen3-4B":
            body = _handler(request).json()
            body["safetensors"] = {"parameters": {"BF16": 4_022_468_096}, "total": 4_022_468_096}
            return httpx.Response(200, json=body)
        if request.url.path.endswith(".safetensors"):
            shard_reads.append(request.url.path)
        return _handler(request)

    fx = facts("Qwen/Qwen3-4B", transport=httpx.MockTransport(handler))
    assert fx.params_total == 4_022_468_096
    assert shard_reads == [], "headers are only the fallback"


def test_a_cache_from_an_older_catalog_is_refetched(tmp_path, monkeypatch) -> None:
    """A warm cache written before facts() learned a field would otherwise answer forever:
    it kept describing LFM2 as all-attention, 166% over what llama.cpp allocates."""
    from rightsize.catalog import hub

    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path))
    path = hub._cache_path("Qwen/Qwen3-4B", "main")
    fresh = facts("Qwen/Qwen3-4B", transport=httpx.MockTransport(_handler))
    hub._write_cache(path, fresh.model_dump(mode="json"))
    assert hub._read_cache(path, ttl_s=3600) is not None, "same version: served from cache"

    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["_cache_version"] = hub.CACHE_VERSION - 1
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert hub._read_cache(path, ttl_s=3600) is None, "older version: refetch"


def test_moe_active_parameters_count_the_always_on_part() -> None:
    """Qwen3-30B-A3B: 128 experts of 3 x 2048 x 768 in 48 layers, 8 routed per token. Only
    idle experts are subtracted; attention and embeddings run every token. Scaling the whole
    model by 8/128 gave 1.9B, where the name and the model card say 3.3B."""
    from rightsize.catalog.hub import _moe_active

    tc = {"moe_intermediate_size": 768, "num_hidden_layers": 48}
    active, how = _moe_active(30_532_122_624, tc, experts=128, k=8, hidden=2048)
    assert active / 1e9 == pytest.approx(3.35, abs=0.02)
    assert "idle routed experts" in how


def test_moe_prediction_layers_are_not_active() -> None:
    """DeepSeek-V3.1 ships a multi-token-prediction layer that plain decoding never runs;
    without subtracting it the active count is 51B against a published 37B."""
    from rightsize.catalog.hub import _moe_active

    tc = {
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "num_nextn_predict_layers": 1,
    }
    active, how = _moe_active(684_531_386_000, tc, experts=256, k=8, hidden=7168)
    assert 36e9 < active < 41e9
    assert "prediction layer" in how


# ---------------------------------------------------------------- pipelines and files


def _pipeline_hub(gated: bool):
    """A FLUX-shaped repo: a single-file checkpoint at the root that duplicates the
    transformer, the diffusers folders, an fp16 variant, and an ONNX export folder."""
    tensors = {"w": ("BF16", [1000, 1000])}
    body = _safetensors_bytes(tensors)
    sizes = {
        "model_index.json": 400,
        "flux1-dev.safetensors": 2_000_000,          # the same transformer, single file
        "transformer/config.json": 300,
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 1_000_000,
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 1_000_000,
        "text_encoder/model.safetensors": 400_000,     # fp32 default ...
        "text_encoder/model.fp16.safetensors": 200_000,  # ... and its fp16 variant
        "vae/diffusion_pytorch_model.safetensors": 100_000,
        "vae_1_0/diffusion_pytorch_model.safetensors": 100_000,  # a spare, not in the index
        "unet_onnx/model.safetensors": 999_999,        # an export, not a component
    }
    index = {
        "_class_name": "FluxPipeline",
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/models/org/pipe":
            assert request.url.params.get("blobs") == "true", "sizes come with blobs=true"
            return httpx.Response(200, json={
                "siblings": [{"rfilename": f, "size": s} for f, s in sizes.items()],
                "pipeline_tag": "text-to-image", "library_name": "diffusers", "gated": gated,
                # the Hub's summary counts one file set: here the root checkpoint only
                "safetensors": {"parameters": {"BF16": 1_000_000}, "total": 1_000_000},
            })
        if gated and "/resolve/" in path:
            return httpx.Response(401)
        if path.endswith("/model_index.json"):
            return httpx.Response(200, json=index)
        if path.endswith("/transformer/config.json"):
            return httpx.Response(200, json={"_class_name": "FluxTransformer2DModel",
                                             "num_attention_heads": 24, "attention_head_dim": 128})
        if path.endswith(".safetensors"):
            a, b = (int(x) for x in request.headers["Range"].split("=")[1].split("-"))
            return httpx.Response(206, content=body[a: b + 1])
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_a_pipeline_counts_each_component_once() -> None:
    fx = facts("org/pipe", transport=_pipeline_hub(gated=False))
    comps = fx.extra["pipeline"]["components"]
    assert set(comps) == {"transformer", "text_encoder", "vae"}, "index names the components"
    assert fx.family.value == "diffusion"
    assert comps["transformer"]["role"] == "denoiser" and comps["vae"]["role"] == "vae"
    assert comps["transformer"]["params"] == 2 * 1_000_000, "two shards, headers read"
    assert comps["text_encoder"]["files"] == ["text_encoder/model.safetensors"], "one variant"
    assert comps["transformer"]["config"]["num_attention_heads"] == 24
    assert fx.params_total == 4 * 1_000_000, "not the Hub summary, not the root checkpoint"
    assert fx.confidence == 1.0


def test_a_gated_pipeline_is_counted_from_file_sizes() -> None:
    fx = facts("org/pipe", transport=_pipeline_hub(gated=True))
    comps = fx.extra["pipeline"]["components"]
    assert set(comps) == {"transformer", "text_encoder", "vae"}, "known folder names"
    assert comps["transformer"]["params"] == 1_000_000, "2 MB at 2 bytes per parameter"
    assert comps["text_encoder"]["params"] == 100_000, "twice its fp16 variant, so fp32"
    assert "file sizes" in comps["text_encoder"]["counted_from"]
    assert fx.confidence < 1.0


def test_a_repo_with_only_pytorch_files_is_sized_from_them() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/models/org/emb":
            files = {"config.json": 500, "pytorch_model.bin": 2_271_145_830,
                     "colbert_linear.pt": 2_100_674}
            return httpx.Response(200, json={
                "siblings": [{"rfilename": f, "size": s} for f, s in files.items()],
                "pipeline_tag": "sentence-similarity", "library_name": "sentence-transformers",
            })
        if request.url.path.endswith("/config.json"):
            return httpx.Response(200, json={"model_type": "xlm-roberta", "hidden_size": 1024,
                                             "torch_dtype": "float32"})
        return httpx.Response(404)

    fx = facts("org/emb", transport=httpx.MockTransport(handler))
    assert fx.family.value == "embedding"
    assert fx.params_total == 2_271_145_830 // 4  # bge-m3: 568M
    assert fx.extra["shards"] == ["pytorch_model.bin"], "the small extra head is not the model"
    assert fx.confidence < 1.0
