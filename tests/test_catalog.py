"""Hub catalog against a fake transport: no network, header-only reads."""

from __future__ import annotations

import json
import struct

import httpx

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
