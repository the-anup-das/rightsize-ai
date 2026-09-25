"""GGUF headers over HTTP Range: the metadata and the tensor table, never the weights (F1).

A GGUF file is a header, then the tensor data. The header is

    magic "GGUF" | version u32 (2 or 3) | tensor count u64 | metadata count u64
    metadata:  key (string) | value type (u32) | value
    tensors:   name (string) | n_dims (u32) | dims (u64 each) | type (u32) | offset (u64)
    zero padding to general.alignment (32), then the data

in little-endian, unless the version bytes show the file was written big-endian. Strings are
a u64 length and UTF-8 bytes; arrays are an element type (u32), a count (u64) and the
elements. That is the layout llama.cpp's own reader (gguf-py's GGUFReader) parses at the
tag we pin, and the type numbers come from the same source (data/quants/ggml_types.yaml).

A header is kilobytes for gigabytes of weights, with one exception: the tokenizer's token
and merge lists live in it, several MB for a 150k vocabulary. Those lists are walked, not
kept, and long numeric arrays (token types, scores) are stepped over without being fetched.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from rightsize._data import load_yaml

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
#: Arrays up to this long are kept: per-layer values (KV heads, sliding-window flags) are,
#: tokenizer lists are not.
KEEP_ARRAY = 4096
KEEP_STRINGS = 64
_FIRST_FETCH = 1 << 20
_MAX_FETCH = 16 << 20

_STRING, _ARRAY = 8, 9
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
           12: "d"}
_TYPE_NAME = {0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
              6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64",
              12: "float64"}

#: llama.cpp architectures that are text encoders: their GGUFs are embedding models.
ENCODER_ARCHITECTURES = frozenset({
    "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert", "jina-bert-v2",
    "jina-bert-v3",
})


class GgufError(ValueError):
    """The bytes are not a GGUF header this reader understands."""


@dataclass(frozen=True)
class LongArray:
    """An array too long to keep, such as a tokenizer list: its element type and length."""

    type: str
    count: int


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int  # from the start of the data section

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Header:
    version: int
    byteorder: str
    metadata: dict[str, Any]
    tensors: list[Tensor]
    data_offset: int  # where the tensor data starts: the header's size, padding included
    file_size: int | None = None
    fetched: int = 0  # bytes downloaded to read it

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    def arch(self, key: str) -> Any:
        """The per-architecture value ``<architecture>.<key>``."""
        return self.metadata.get(f"{self.architecture}.{key}")


class _Cursor:
    """Reads a file front to back through ``fetch(start, stop)``, which returns the bytes
    [start, stop), or fewer at the end of the file. Each fetch asks for twice as much as the
    last, so a header of any size costs a handful of requests."""

    def __init__(self, fetch: Callable[[int, int], bytes], first: int) -> None:
        self._fetch = fetch
        self._buf = b""
        self._start = 0
        self._next = first
        self.pos = 0
        self.fetched = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        have = self._start + len(self._buf)
        if end > have or self.pos < self._start:
            keep = self._buf[self.pos - self._start:] if self._start <= self.pos < have else b""
            start = self.pos + len(keep)
            data = self._fetch(start, start + max(end - start, self._next))
            self.fetched += len(data)
            self._next = min(self._next * 2, _MAX_FETCH)
            self._buf, self._start = keep + data, self.pos
            if end > self._start + len(self._buf):
                raise GgufError("the file ends inside its header")
        i = self.pos - self._start
        self.pos = end
        return self._buf[i:i + n]


class _Reader:
    def __init__(self, cur: _Cursor, order: str) -> None:
        self.cur = cur
        self.order = order
        self._u64 = struct.Struct(order + "Q")

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        s = struct.Struct(self.order + fmt)
        return s.unpack(self.cur.take(s.size))

    def _length(self) -> int:
        (n,) = self._u64.unpack(self.cur.take(8))
        if n > 1 << 30:
            raise GgufError(f"implausible string length {n}; not a GGUF header")
        return n

    def string(self) -> str:
        return self.cur.take(self._length()).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            return self.array()
        code = _SCALAR.get(vtype)
        if code is None:
            raise GgufError(f"unknown metadata value type {vtype}")
        return self.unpack(code)[0]

    def array(self) -> Any:
        etype, count = self.unpack("IQ")
        if etype == _STRING:
            if count <= KEEP_STRINGS:
                return [self.string() for _ in range(count)]
            for _ in range(count):
                n = self._length()  # (not "pos += _length()": that reads pos first)
                self.cur.pos += n  # stepped over, never decoded
            return LongArray("string", count)
        if etype == _ARRAY:  # each element carries its own type and count
            return [self.array() for _ in range(count)]
        code = _SCALAR.get(etype)
        if code is None:
            raise GgufError(f"unknown array element type {etype}")
        if count <= KEEP_ARRAY:
            return list(self.unpack(f"{count}{code}"))
        self.cur.pos += count * struct.calcsize(code)
        return LongArray(_TYPE_NAME[etype], count)


def parse(fetch: Callable[[int, int], bytes], *, first: int = _FIRST_FETCH) -> Header:
    """Parse a GGUF header from a byte source; see the module docstring for the layout."""
    cur = _Cursor(fetch, first)
    if cur.take(4) != MAGIC:
        raise GgufError("not a GGUF file: the first four bytes are not 'GGUF'")
    raw = cur.take(4)
    order = "<"
    (version,) = struct.unpack("<I", raw)
    if version & 0xFFFF == 0:  # gguf-py's test: a big-endian file reads as 0x03000000
        order = ">"
        (version,) = struct.unpack(">I", raw)
    if version == 1:
        raise GgufError("GGUF version 1 files (2023) are not supported; convert the model again")
    if version not in (2, 3):
        raise GgufError(f"unknown GGUF version {version}")
    r = _Reader(cur, order)
    n_tensors, n_kv = r.unpack("QQ")
    if n_tensors > 1 << 20 or n_kv > 1 << 20:
        raise GgufError("implausible tensor or metadata count; not a GGUF header")
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        (vtype,) = r.unpack("I")
        metadata[key] = r.value(vtype)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        (n_dims,) = r.unpack("I")
        if n_dims > 8:
            raise GgufError(f"tensor {name!r} has {n_dims} dimensions; not a GGUF header")
        dims = r.unpack(f"{n_dims}Q")
        type_id, offset = r.unpack("IQ")
        tensors.append(Tensor(name, tuple(dims), type_id, offset))
    align = int(metadata.get("general.alignment") or DEFAULT_ALIGNMENT)
    data_offset = -(-cur.pos // align) * align
    return Header(version, "big" if order == ">" else "little", metadata, tensors, data_offset,
                  fetched=cur.fetched)


def from_bytes(data: bytes) -> Header:
    """Parse a header held in memory (tests, local files)."""
    h = parse(lambda start, stop: data[start:stop])
    h.file_size = len(data) if len(data) > h.data_offset else None
    return h


def read_header(client: httpx.Client, url: str) -> Header:
    """Read a remote file's header with Range requests; the weights are never fetched."""
    seen: dict[str, int] = {}

    def fetch(start: int, stop: int) -> bytes:
        with client.stream("GET", url, headers={"Range": f"bytes={start}-{stop - 1}"}) as r:
            if r.status_code == 416:  # asked past the end of the file
                return b""
            r.raise_for_status()
            if r.status_code == 206:
                total = r.headers.get("content-range", "").rpartition("/")[2]
                if total.isdigit():
                    seen["size"] = int(total)
                return r.read()
            # A 200 means the server ignored the range. Read only what was asked for, then
            # hang up, rather than download gigabytes of weights.
            out = bytearray()
            for chunk in r.iter_bytes():
                out += chunk
                if len(out) >= stop:
                    break
            return bytes(out[start:stop])

    h = parse(fetch)
    h.file_size = seen.get("size")
    return h


# ---------------------------------------------------------------- types and sizes


def tensor_types() -> dict[int, tuple[str, int, int]]:
    """type id -> (name, weights per block, bytes per block)."""
    doc = load_yaml("quants/ggml_types.yaml")
    return {t["id"]: (t["name"], t["block_size"], t["type_size"]) for t in doc["tensor_types"]}


def file_type_name(h: Header) -> str | None:
    """general.file_type as llama-quantize names it (15 is Q4_K_M)."""
    ft = h.metadata.get("general.file_type")
    if ft is None:
        return None
    names = {t["id"]: t["name"] for t in load_yaml("quants/ggml_types.yaml")["file_types"]}
    return names.get(int(ft), f"file type {ft}")


def tensor_bytes(h: Header, types: dict[int, tuple[str, int, int]]) -> list[int | None]:
    """Bytes of each tensor. Known types from their block size; a type newer than our table
    from the gap to the next tensor's offset, which needs the file size for the last one
    and includes up to ``alignment - 1`` bytes of padding."""
    out: list[int | None] = []
    for t in h.tensors:
        spec = types.get(t.type_id)
        out.append(t.n_elements // spec[1] * spec[2] if spec else None)
    if None in out:
        order = sorted(range(len(h.tensors)), key=lambda i: h.tensors[i].offset)
        data_end = h.file_size - h.data_offset if h.file_size else None
        for pos, i in enumerate(order):
            if out[i] is None:
                nxt = (h.tensors[order[pos + 1]].offset if pos + 1 < len(order) else data_end)
                out[i] = nxt - h.tensors[i].offset if nxt is not None else None
    return out


_BLOCK = re.compile(r"^blk\.(\d+)\.")


def summarize(headers: list[Header]) -> dict[str, Any]:
    """Parameters and bytes per tensor type across a file's parts, and the tensors the fit
    engine treats specially: the input embedding (llama.cpp keeps it in system RAM) and the
    output head (absent when tied to the embedding)."""
    types = tensor_types()
    params_by: dict[str, int] = {}
    bytes_by: dict[str, int] = {}
    named: dict[str, int | None] = {}
    unknown: set[int] = set()
    total_params = total_bytes = 0
    exact = True
    for h in headers:
        for t, nb in zip(h.tensors, tensor_bytes(h, types), strict=True):
            name = types[t.type_id][0] if t.type_id in types else f"type {t.type_id}"
            if t.type_id not in types:
                unknown.add(t.type_id)
            params_by[name] = params_by.get(name, 0) + t.n_elements
            total_params += t.n_elements
            if nb is None:
                exact = False
            else:
                bytes_by[name] = bytes_by.get(name, 0) + nb
                total_bytes += nb
            if t.name in ("token_embd.weight", "output.weight"):
                named[t.name] = nb
    return {
        "params_total": total_params,
        "params_by_type": dict(sorted(params_by.items(), key=lambda kv: -kv[1])),
        "bytes_by_type": dict(sorted(bytes_by.items(), key=lambda kv: -kv[1])),
        "tensor_bytes": total_bytes if exact else None,
        "input_embedding_bytes": named.get("token_embd.weight"),
        "output_bytes": named.get("output.weight"),
        "tied_embeddings": "output.weight" not in named,
        "tensor_count": sum(len(h.tensors) for h in headers),
        "unknown_types": sorted(unknown),
    }


def active_params(headers: list[Header], total: int) -> tuple[int | None, str]:
    """Parameters a mixture-of-experts model runs per token, counted from its tensors:
    routed expert tensors (``*_exps``) at the share of experts used, plus every tensor of
    the multi-token-prediction blocks plain decoding never runs. Shared experts, attention
    and embeddings run for every token."""
    head = headers[0]
    experts, k = head.arch("expert_count"), head.arch("expert_used_count")
    if not (experts and k):
        return None, ""
    blocks = head.arch("block_count") or 0
    nextn = head.arch("nextn_predict_layers") or 0
    idle = 0.0
    for h in headers:
        for t in h.tensors:
            m = _BLOCK.match(t.name)
            if nextn and m and int(m.group(1)) >= blocks - nextn:
                idle += t.n_elements
            elif "_exps." in t.name:
                idle += t.n_elements * (1 - k / experts)
    how = f"total minus idle routed experts ({k} of {experts} used), from the GGUF's tensors"
    if nextn:
        how += f" and {nextn} prediction layer{'s' if nextn > 1 else ''}"
    return int(total - idle), how


# ---------------------------------------------------------------- architecture


def _layers(value: Any, n: int) -> list[Any] | None:
    return list(value[:n]) if isinstance(value, list) and len(value) >= n else None


def _one(value: Any) -> Any:
    """A per-layer array's representative value: the largest non-zero one."""
    if isinstance(value, list):
        nonzero = [v for v in value if isinstance(v, (int, float)) and v]
        return max(nonzero) if nonzero else None
    return value


def config_view(h: Header) -> dict[str, Any]:
    """The GGUF's architecture metadata under the names config.json uses, so the fit
    engine's KV rules read a GGUF exactly as they read the model it was converted from.

    Hybrid models mark their recurrent layers with zero KV heads in a per-layer array
    (llama.cpp: a layer is recurrent when n_head_kv(il) == 0); those arrays become
    ``layer_types``. Multi-token-prediction blocks are counted in block_count but are not
    decoder layers, as in the config."""
    g = h.arch
    blocks = g("block_count")
    nextn = g("nextn_predict_layers") or 0
    n = blocks - nextn if blocks else None
    heads = _one(g("attention.head_count"))
    kv_raw = g("attention.head_count_kv")
    hidden = g("embedding_length")
    kv_lora = g("attention.kv_lora_rank")
    tokens = h.metadata.get("tokenizer.ggml.tokens")
    vocab = g("vocab_size") or (tokens.count if isinstance(tokens, LongArray)
                               else len(tokens) if isinstance(tokens, list) else None)
    view: dict[str, Any] = {
        "model_type": h.architecture,
        "num_hidden_layers": n,
        "num_attention_heads": heads,
        "num_key_value_heads": _one(kv_raw) or heads,
        "head_dim": g("attention.key_length") or (hidden // heads if hidden and heads else None),
        "hidden_size": hidden,
        "intermediate_size": _one(g("feed_forward_length")),
        "max_position_embeddings": g("context_length"),
        "vocab_size": vocab,
        "sliding_window": g("attention.sliding_window"),
        "num_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "moe_intermediate_size": g("expert_feed_forward_length"),
        "kv_lora_rank": kv_lora,
        "qk_rope_head_dim": g("rope.dimension_count") if kv_lora else None,
        "full_attention_interval": g("full_attention_interval"),
        "mamba_d_state": g("ssm.state_size"),
        "mamba_d_ssm": g("ssm.inner_size"),
        "mamba_n_groups": g("ssm.group_count"),
        "mamba_d_conv": g("ssm.conv_kernel"),
        "conv_L_cache": g("shortconv.l_cache"),
    }
    kv_layers = _layers(kv_raw, n) if n else None
    swa = g("attention.sliding_window_pattern")
    swa_layers = _layers(swa, n) if n else None
    if kv_layers and any(v == 0 for v in kv_layers):
        view["layer_types"] = [
            ("sliding_attention" if swa_layers and swa_layers[i] else "full_attention")
            if v else "recurrent"
            for i, v in enumerate(kv_layers)
        ]
    elif swa_layers:
        view["layer_types"] = ["sliding_attention" if s else "full_attention" for s in swa_layers]
    elif isinstance(swa, int) and not isinstance(swa, bool):
        view["sliding_window_pattern"] = swa
    return view


def base_model(h: Header) -> str | None:
    """The Hub repo the GGUF says it was converted from, if it names one."""
    url = h.metadata.get("general.base_model.0.repo_url") or ""
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        return None
    return url[len(prefix):].strip("/") or None


# ---------------------------------------------------------------- files in a repo

_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
_QUANT = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|B?F(?:P)?(?:16|32)|MXFP4(?:_MOE)?|NVFP4"
    r"|TQ\d_\d))(?=[-_.]|$)",
    re.I,
)
#: Files that sit beside the weights but are not them: vision projectors and the
#: importance matrices some publishers upload.
_NOT_WEIGHTS = ("mmproj", "imatrix")
#: The file a repo is sized at when none is named: the usual default of LM Studio and
#: Ollama first, then the nearest alternatives.
PREFERRED = ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "Q4_0", "MXFP4", "MXFP4_MOE",
             "Q5_K_M", "Q6_K", "Q8_0", "BF16", "F16", "F32")


def quant_label(path: str) -> str | None:
    """The quantization a file's name states: Q4_K_M, UD-Q4_K_XL, IQ4_XS, BF16, MXFP4.
    The last match wins, since model names come first and quant labels last."""
    name = _SPLIT.sub(".gguf", path.rsplit("/", 1)[-1])
    found = [m.group(1).upper() for m in _QUANT.finditer(name[: -len(".gguf")])]
    if not found:
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        found = [m.group(1).upper() for m in _QUANT.finditer(folder.rsplit("/", 1)[-1])]
    label = found[-1] if found else None
    return {"FP16": "F16", "FP32": "F32", "BFP16": "BF16"}.get(label, label) if label else None


def identity(path: str) -> str:
    """Which model a file holds: its name without the quantization label, split suffix
    and extension. ``gpt-oss-20b-MXFP4.gguf`` and ``eagle3-gpt-oss-20b-BF16.gguf`` sit in
    one repo but are two models (the second is a speculative-decoding draft)."""
    name = _SPLIT.sub(".gguf", path.rsplit("/", 1)[-1])[: -len(".gguf")]
    label = quant_label(path)
    if label:
        spans = [m.span(1) for m in _QUANT.finditer(name)]
        if spans:
            a, b = spans[-1]
            name = name[:a] + name[b:]
    return name.strip("-_. ").lower()


def weight_files(
    sizes: dict[str, int | None], repo: str | None = None, *, model: str | None = None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The repo's GGUF weights by quantization, split parts grouped, and the GGUF files that
    hold some other model: ``({"Q4_K_M": {"files": [...], "bytes": 2497281312}}, [...])``.

    The model is the identity named like the repo (``unsloth/Qwen3-4B-GGUF`` holds
    ``Qwen3-4B-*.gguf``); failing that, the one with the most quantizations. ``model``
    (an identity) picks another. Within it, a quantization stated twice keeps the shorter
    file name."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(sizes, key=lambda p: (len(p), p)):
        base = path.rsplit("/", 1)[-1].lower()
        if not base.endswith(".gguf") or any(w in base for w in _NOT_WEIGHTS):
            continue
        stem = _SPLIT.sub("", path)
        label = quant_label(path) or stem.rsplit("/", 1)[-1].removesuffix(".gguf")
        groups = found.setdefault(identity(path), {})
        g = groups.setdefault(label, {"stem": stem, "files": [], "bytes": 0})
        if g["stem"] != stem:
            continue
        g["files"].append(path)
        g["bytes"] += sizes.get(path) or 0
    if not found:
        return {}, []
    wanted = re.sub(r"[-_.]gguf$", "", (repo or "").rsplit("/", 1)[-1], flags=re.I).lower()
    main = model if model in found else wanted if wanted in found else max(
        found, key=lambda k: (len(found[k]), sum(g["bytes"] for g in found[k].values())))
    for groups in found.values():
        for g in groups.values():
            g["files"].sort()
            del g["stem"]
    others = sorted(f for k, groups in found.items() if k != main
                    for g in groups.values() for f in g["files"])
    return found[main], others


def extras(sizes: dict[str, int | None], kind: str) -> dict[str, int]:
    """Side files by name, such as ``extras(sizes, "mmproj")``: the vision projector an
    image-capable GGUF needs loaded beside the weights."""
    return {f: s or 0 for f, s in sorted(sizes.items())
            if f.lower().endswith(".gguf") and kind in f.rsplit("/", 1)[-1].lower()}


def default_quant(groups: dict[str, Any]) -> str:
    for q in PREFERRED:
        if q in groups:
            return q
    return sorted(groups)[0]


def group_of(path: str, groups: dict[str, dict[str, Any]]) -> str | None:
    """The quantization a named file belongs to, split parts included."""
    for label, g in groups.items():
        if path in g["files"]:
            return label
    return None
