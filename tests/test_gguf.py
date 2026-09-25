"""GGUF headers read over HTTP Range, and the facts and estimates built on them (F1)."""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import httpx
import pytest
import yaml

from rightsize.catalog import facts, gguf
from rightsize.catalog.gguf import GgufError, LongArray
from rightsize.fit import estimate, kv
from rightsize.hardware import resolve
from rightsize.types import ModelFacts

ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / "tests" / "fixtures" / "gguf"
FACTS = ROOT / "tests" / "fixtures" / "facts"

_CODES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
          12: "d"}


def _gguf(metadata, tensors=(), *, order="<", version=3, alignment=32, data=True) -> bytes:
    """A GGUF file as llama.cpp writes one: header, padding, zeroed tensor data."""
    def u32(x): return struct.pack(order + "I", x)
    def u64(x): return struct.pack(order + "Q", x)
    def s(x): return u64(len(x.encode())) + x.encode()

    def val(t, v):
        if t == 8:
            return s(v)
        if t == 9:
            et, items = v
            return u32(et) + u64(len(items)) + b"".join(val(et, i) for i in items)
        return struct.pack(order + _CODES[t], v)

    types = gguf.tensor_types()
    out = b"GGUF" + u32(version) + u64(len(tensors)) + u64(len(metadata))
    for key, t, v in metadata:
        out += s(key) + u32(t) + val(t, v)
    offset = 0
    for name, shape, tid in tensors:
        out += s(name) + u32(len(shape)) + b"".join(u64(d) for d in shape) + u32(tid) + u64(offset)
        n = 1
        for d in shape:
            n *= d
        nbytes = n // types[tid][1] * types[tid][2] if tid in types else 100
        offset += -(-nbytes // alignment) * alignment
    out += b"\0" * (-len(out) % alignment)
    return out + (b"\0" * offset if data else b"")


# ---------------------------------------------------------------- the format


EVERY_TYPE = [
    ("general.architecture", 8, "llama"),
    ("u8", 0, 7), ("i8", 1, -7), ("u16", 2, 65535), ("i16", 3, -300),
    ("u32", 4, 4_000_000_000), ("i32", 5, -5), ("f32", 6, 0.5), ("bool", 7, True),
    ("u64", 10, 2**40), ("i64", 11, -(2**40)), ("f64", 12, 0.25),
    ("per_layer", 9, (4, [8, 0, 8])),
    ("names", 9, (8, ["a", "b"])),
    ("nested", 9, (9, [(4, [1]), (8, ["x"])])),
    ("tokenizer.ggml.tokens", 9, (8, [f"tok{i}" for i in range(1000)])),
    ("tokenizer.ggml.scores", 9, (6, [0.0] * 5000)),
]


def test_every_value_type_reads_back() -> None:
    h = gguf.from_bytes(_gguf(EVERY_TYPE, [("w", (256, 4), 12)]))
    md = h.metadata
    assert (md["u8"], md["i8"], md["u16"], md["i16"]) == (7, -7, 65535, -300)
    assert (md["u32"], md["i32"], md["f32"], md["bool"]) == (4_000_000_000, -5, 0.5, True)
    assert (md["u64"], md["i64"], md["f64"]) == (2**40, -(2**40), 0.25)
    assert md["per_layer"] == [8, 0, 8] and md["names"] == ["a", "b"]
    assert md["nested"] == [[1], ["x"]]
    assert md["tokenizer.ggml.tokens"] == LongArray("string", 1000), "walked, not kept"
    assert md["tokenizer.ggml.scores"] == LongArray("float32", 5000)
    assert h.tensors[0].shape == (256, 4) and h.tensors[0].type_id == 12
    assert h.data_offset % 32 == 0


def test_a_big_endian_file_reads_the_same() -> None:
    little = gguf.from_bytes(_gguf(EVERY_TYPE, [("w", (256, 4), 12)]))
    big = gguf.from_bytes(_gguf(EVERY_TYPE, [("w", (256, 4), 12)], order=">"))
    assert big.byteorder == "big"
    assert big.metadata == little.metadata and big.tensors == little.tensors


def test_long_numeric_arrays_are_stepped_over_without_fetching_them() -> None:
    """A 4 MB array sits between the keys and the tensor table; the reader jumps it."""
    blob = _gguf([("general.architecture", 8, "llama"), ("big", 9, (6, [0.0] * 1_000_000)),
                  ("after", 4, 42)], [("w", (32,), 0)], data=False)
    fetched: list[int] = []

    def fetch(start: int, stop: int) -> bytes:
        fetched.append(stop - start)
        return blob[start:stop]

    h = gguf.parse(fetch, first=256)
    assert h.metadata["after"] == 42 and h.metadata["big"] == LongArray("float32", 1_000_000)
    assert sum(fetched) < 50_000, f"fetched {sum(fetched)} of {len(blob)} bytes"


def test_a_type_newer_than_the_table_is_sized_from_the_offsets() -> None:
    blob = _gguf([("general.architecture", 8, "llama")],
                 [("a", (256,), 12), ("b", (64,), 99), ("c", (32,), 0)])
    h = gguf.from_bytes(blob)
    s = gguf.summarize([h])
    assert s["unknown_types"] == [99]
    assert s["bytes_by_type"] == {"Q4_K": 144, "type 99": 128, "F32": 128}, \
        "the gap to the next tensor, alignment padding included"


@pytest.mark.parametrize(("blob", "message"), [
    (b"GGUF" + struct.pack("<I", 1) + b"\0" * 64, "version 1"),
    (b"PK\x03\x04" + b"\0" * 64, "not a GGUF"),
    (_gguf(EVERY_TYPE)[:200], "ends inside"),
])
def test_what_is_not_a_readable_header_says_why(blob: bytes, message: str) -> None:
    with pytest.raises(GgufError, match=message):
        gguf.from_bytes(blob)


def test_the_type_table_agrees_with_the_bits_per_weight_table() -> None:
    """ggml_types.yaml is generated from llama.cpp's gguf-py; gguf_bpw.yaml was written from
    huggingface.js. Two sources, one number per type."""
    bpw = yaml.safe_load((ROOT / "data" / "quants" / "gguf_bpw.yaml").read_text())["tensor_types"]
    for tid, (name, block, size) in gguf.tensor_types().items():
        if name in bpw:
            assert size * 8 / block == pytest.approx(bpw[name]), (tid, name)


# ---------------------------------------------------------------- real headers


def _head(name: str) -> gguf.Header:
    return gguf.from_bytes((HEADS / f"{name}.head").read_bytes())


def test_a_real_header_from_llama_cpps_test_models() -> None:
    """stories260K, the model llama.cpp's CI runs: header bytes only (tests/fixtures/gguf)."""
    h = _head("stories260K")
    s = gguf.summarize([h])
    v = gguf.config_view(h)
    assert (h.version, h.architecture, len(h.tensors)) == (3, "llama", 48)
    assert s["params_total"] == 292_800 and s["bytes_by_type"] == {"F32": 1_171_200}
    assert s["tied_embeddings"] is False and s["input_embedding_bytes"] == 512 * 64 * 4
    assert (v["num_hidden_layers"], v["num_attention_heads"], v["num_key_value_heads"],
            v["head_dim"], v["vocab_size"]) == (5, 8, 4, 8, 512)


def test_the_big_endian_copy_of_that_header_matches() -> None:
    little, big = _head("stories260K"), _head("stories260K-be")
    assert big.byteorder == "big" and big.tensors == little.tensors
    assert gguf.summarize([big]) == gguf.summarize([little])


def test_a_split_part_carries_its_own_tensors() -> None:
    h = _head("stories15M-q8_0-00002-of-00003")
    assert (h.metadata["split.no"], h.metadata["split.count"]) == (1, 3)
    assert len(h.tensors) == 7 and "Q8_0" in gguf.summarize([h])["bytes_by_type"]


# ---------------------------------------------------------------- files in a repo


@pytest.mark.parametrize(("path", "label", "model"), [
    ("Qwen3-4B-Q4_K_M.gguf", "Q4_K_M", "qwen3-4b"),
    ("Qwen3-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL", "qwen3-4b"),
    ("Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", "Q4_K_M", "mistral-7b-instruct-v0.3"),
    ("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "Q4_K_M", "qwen2.5-7b-instruct"),
    ("gpt-oss-20b-MXFP4.gguf", "MXFP4", "gpt-oss-20b"),
    ("DeepSeek-R1-0528-Qwen3-8B-IQ4_XS.gguf", "IQ4_XS", "deepseek-r1-0528-qwen3-8b"),
    ("UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00004.gguf", "UD-IQ1_S", "deepseek-r1"),
    ("model-fp16.gguf", "F16", "model"),
    ("stories260K.gguf", None, "stories260k"),
])
def test_quantization_and_model_from_the_file_name(path, label, model) -> None:
    assert gguf.quant_label(path) == label
    assert gguf.identity(path) == model


def test_the_repos_own_model_is_told_from_a_draft_beside_it() -> None:
    """ggml-org/gpt-oss-20b-GGUF also holds an EAGLE3 draft model in BF16 and Q8_0; those
    are not gpt-oss-20b at 16 or 8 bits."""
    sizes = {"gpt-oss-20b-MXFP4.gguf": 12_109_566_624,
             "eagle3-gpt-oss-20b-BF16.gguf": 1_722_588_800,
             "eagle3-gpt-oss-20b-Q8_0.gguf": 921_488_000, "README.md": 462}
    groups, others = gguf.weight_files(sizes, "ggml-org/gpt-oss-20b-GGUF")
    assert list(groups) == ["MXFP4"] and len(others) == 2
    draft, _ = gguf.weight_files(sizes, "ggml-org/gpt-oss-20b-GGUF", model="eagle3-gpt-oss-20b")
    assert set(draft) == {"BF16", "Q8_0"}


def test_split_parts_group_and_side_files_stay_out() -> None:
    sizes = {f"Q4_K_M/Big-Q4_K_M-0000{i}-of-00003.gguf": 10 for i in (1, 2, 3)}
    sizes |= {"Big-Q8_0.gguf": 40, "mmproj-Big-F16.gguf": 5, "Big.imatrix.gguf": 1}
    groups, _ = gguf.weight_files(sizes, "org/Big-GGUF")
    assert groups["Q4_K_M"] == {"files": sorted(f for f in sizes if "Q4_K_M/" in f), "bytes": 30}
    assert set(groups) == {"Q4_K_M", "Q8_0"}, "no projector, no importance matrix"
    assert gguf.extras(sizes, "mmproj") == {"mmproj-Big-F16.gguf": 5}
    assert gguf.default_quant(groups) == "Q4_K_M"
    assert gguf.default_quant({"Q8_0": {}, "BF16": {}}) == "Q8_0"


# ---------------------------------------------------------------- facts from a GGUF repo


def _tiny(file_type: int = 15) -> bytes:
    """A two-layer llama in Q4_K with tied embeddings (no output.weight)."""
    meta = [
        ("general.architecture", 8, "llama"),
        ("general.file_type", 4, file_type),
        ("llama.block_count", 4, 2),
        ("llama.context_length", 4, 4096),
        ("llama.embedding_length", 4, 256),
        ("llama.feed_forward_length", 4, 512),
        ("llama.attention.head_count", 4, 4),
        ("llama.attention.head_count_kv", 4, 2),
        ("tokenizer.ggml.tokens", 9, (8, [f"t{i}" for i in range(256)])),
    ]
    tensors = [("token_embd.weight", (256, 256), 12), ("output_norm.weight", (256,), 0)]
    for i in range(2):
        tensors += [(f"blk.{i}.attn_q.weight", (256, 256), 12),
                    (f"blk.{i}.ffn_down.weight", (512, 256), 14)]
    return _gguf(meta, tensors)


TINY = _tiny()
TINY_HEADER = gguf.from_bytes(TINY).data_offset
LISTING = {"Tiny-Q4_K_M.gguf": len(TINY), "Tiny-Q8_0.gguf": 2 * len(TINY),
           "mmproj-Tiny-F16.gguf": 1000, "README.md": 10}


def _hub(*, gated: bool = False, sha: str = "abc", seen: list[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if path == "/api/models/org/Tiny-GGUF/revision/main":
            return httpx.Response(200, json={"id": "org/Tiny-GGUF", "sha": sha})
        if path == "/api/models/org/Tiny-GGUF":
            return httpx.Response(200, json={
                "sha": sha, "gated": gated, "pipeline_tag": "text-generation",
                "cardData": {"license": "apache-2.0", "base_model": "org/Tiny"},
                "siblings": [{"rfilename": f, "size": n} for f, n in LISTING.items()],
                "gguf": {"total": 123_456, "architecture": "llama", "context_length": 4096},
            })
        if gated:
            return httpx.Response(401)
        if path.endswith("/Tiny-Q4_K_M.gguf"):
            a, b = (int(x) for x in request.headers["Range"].split("=")[1].split("-"))
            return httpx.Response(206, content=TINY[a:b + 1],
                                  headers={"Content-Range": f"bytes {a}-{b}/{len(TINY)}"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_a_gguf_repo_is_read_from_its_q4_k_m_header() -> None:
    seen: list[str] = []
    fx = facts("org/Tiny-GGUF", transport=_hub(seen=seen))
    assert not any(p.endswith("Q8_0.gguf") for p in seen), "one header, not every file"
    assert fx.ref.file == "Tiny-Q4_K_M.gguf" and fx.dtype == "Q4_K_M"
    assert (fx.num_layers, fx.num_kv_heads, fx.head_dim, fx.context_max) == (2, 2, 64, 4096)
    assert fx.params_total == 256 * 256 * 3 + 256 + 2 * 512 * 256
    assert fx.extra["tie_word_embeddings"] is True and fx.base_model == "org/Tiny"
    assert fx.extra["attention"]["model_type"] == "llama" and fx.confidence == 1.0
    files = fx.extra["gguf_files"]
    tensor_bytes = fx.extra["gguf"]["tensor_bytes"]
    assert files["Q4_K_M"] == {"files": ["Tiny-Q4_K_M.gguf"], "bytes": len(TINY),
                               "weights_bytes": tensor_bytes, "exact": True}
    assert files["Q8_0"]["weights_bytes"] == 2 * len(TINY) - TINY_HEADER
    assert fx.extra["mmproj"] == {"mmproj-Tiny-F16.gguf": 1000}


def test_an_estimate_uses_the_file_it_was_read_from() -> None:
    fx = facts("org/Tiny-GGUF", transport=_hub())
    dev = resolve("RTX 4070")
    r = estimate(fx, None, dev, ctx=4096)
    assert r.breakdown["weights"] == round(fx.extra["gguf"]["tensor_bytes"] / 1e9, 3)
    assert any("the repo's Q4_K_M file (its header)" in n for n in r.notes)
    other = estimate(fx, "Q6_K", dev, ctx=4096)
    assert any("no Q6_K file" in n for n in other.notes), "a quantization the repo lacks"


def test_a_named_file_must_be_in_the_repo() -> None:
    with pytest.raises(FileNotFoundError, match="Tiny-Q4_K_M.gguf"):
        facts("org/Tiny-GGUF", file="Tiny-Q2_K.gguf", transport=_hub())
    with pytest.raises(ValueError, match="selects a GGUF"):
        facts("org/Tiny-GGUF", file="model.safetensors", transport=_hub())


def test_a_gated_gguf_repo_falls_back_to_the_hubs_summary() -> None:
    fx = facts("org/Tiny-GGUF", transport=_hub(gated=True))
    assert fx.params_total == 123_456 and fx.confidence == 0.7 and fx.num_layers is None
    with pytest.raises(ValueError, match="HF_TOKEN"):
        estimate(fx, None, resolve("RTX 4070"))


def test_model_first_plans_for_a_gguf_repo_say_what_to_do_instead() -> None:
    from rightsize.errors import NotImplementedYet
    from rightsize.rules.recommend import recommend_for_model

    fx = facts("org/Tiny-GGUF", transport=_hub())
    with pytest.raises(NotImplementedYet, match="rightsize estimate org/Tiny-GGUF"):
        recommend_for_model(fx, "RTX 4070")


# ---------------------------------------------------------------- the cache


def test_expired_facts_are_kept_while_the_commit_is_unchanged(tmp_path, monkeypatch) -> None:
    """After the TTL, one ~100-byte request for the commit id decides whether to reread the
    headers: 13 MB for gpt-oss-20b's tokenizer alone."""
    from rightsize.catalog import hub

    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path))
    seen: list[str] = []
    facts("org/Tiny-GGUF", transport=_hub(seen=seen), use_cache=True)
    path = hub._cache_path("org/Tiny-GGUF", "main")
    old = time.time() - 30 * 24 * 3600
    os.utime(path, (old, old))

    seen.clear()
    facts("org/Tiny-GGUF", transport=_hub(seen=seen), use_cache=True)
    assert seen == ["/api/models/org/Tiny-GGUF/revision/main"], "only the commit was asked"
    assert path.stat().st_mtime > old, "and the entry is fresh again"

    os.utime(path, (old, old))
    seen.clear()
    fx = facts("org/Tiny-GGUF", transport=_hub(seen=seen, sha="def"), use_cache=True)
    assert "/api/models/org/Tiny-GGUF" in seen, "a new commit is read again"
    assert fx.extra["sha"] == "def"


def test_facts_for_a_commit_id_do_not_go_stale(tmp_path, monkeypatch) -> None:
    from rightsize.catalog import hub

    monkeypatch.setenv("RIGHTSIZE_CACHE_DIR", str(tmp_path))
    commit = "0123456789abcdef0123456789abcdef01234567"
    fx = facts("org/Tiny-GGUF", transport=_hub(), use_cache=True)
    path = hub._cache_path("org/Tiny-GGUF", commit)
    hub._write_cache(path, fx.model_dump(mode="json"))
    old = time.time() - 365 * 24 * 3600
    os.utime(path, (old, old))
    seen: list[str] = []
    facts("org/Tiny-GGUF", commit, transport=_hub(seen=seen), use_cache=True)
    assert seen == []


# ---------------------------------------------------------------- the same model, two sources

PAIRS = [
    ("Qwen/Qwen3-4B", "unsloth/Qwen3-4B-GGUF"),
    ("openai/gpt-oss-20b", "ggml-org/gpt-oss-20b-GGUF"),
    ("unsloth/gemma-3-270m-it", "unsloth/gemma-3-270m-it-GGUF"),
    ("LiquidAI/LFM2-350M", "LiquidAI/LFM2-350M-GGUF"),
    ("ibm-granite/granite-4.0-h-350m", "ibm-granite/granite-4.0-h-350m-GGUF"),
    ("tiiuae/Falcon-H1-1.5B-Instruct", "tiiuae/Falcon-H1-1.5B-Instruct-GGUF"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF"),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", "mradermacher/DeepSeek-V2-Lite-Chat-GGUF"),
]


def _fixture(repo: str) -> ModelFacts:
    raw = (FACTS / f"{repo.replace('/', '__')}.json").read_text(encoding="utf-8")
    return ModelFacts.model_validate(json.loads(raw))


@pytest.mark.parametrize(("config_repo", "gguf_repo"), PAIRS)
def test_a_gguf_header_and_its_config_give_the_same_cache(config_repo, gguf_repo) -> None:
    """Recorded from the live Hub: plain GQA, sliding windows, MLA and four kinds of hybrid.
    llama.cpp reads the GGUF, so the header is what it will allocate for; the config is what
    the model was converted from. The KV rules must read both the same way."""
    a, b = _fixture(config_repo), _fixture(gguf_repo)
    assert kv.layout(a).rule == kv.layout(b).rule
    for ctx in (4096, 32768, 131072):
        assert kv.kv_breakdown(a, ctx) == kv.kv_breakdown(b, ctx), ctx


def test_a_split_gguf_counts_every_part() -> None:
    fx = _fixture("unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF")
    assert len(fx.extra["gguf"]["files"]) == 3
    assert fx.params_total / 1e9 == pytest.approx(235.1, abs=0.1)
    assert fx.params_active / 1e9 == pytest.approx(22.2, abs=0.1), "A22B, from the tensors"


def test_the_gguf_drops_what_llama_cpp_does_not_run() -> None:
    """Qwen3-Next's checkpoint carries a multi-token-prediction layer the converter leaves
    out: the GGUF is what llama.cpp loads, 1.65B parameters lighter."""
    hf = _fixture("Qwen/Qwen3-Next-80B-A3B-Instruct")
    gg = _fixture("unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF")
    assert (hf.params_total - gg.params_total) / 1e9 == pytest.approx(1.65, abs=0.01)
