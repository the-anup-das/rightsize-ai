"""Which files hold a model's weights, and how many parameters they carry (F1).

A repo often holds the same weights more than once: a single-file checkpoint at the root
beside the diffusers folders, fp32 and fp16 variants side by side, ONNX and OpenVINO
exports, a spare VAE. Summing every file counts a model two or three times, so each
folder contributes one copy, and a diffusers pipeline contributes only the folders its
model_index.json names.

Parameters come from the safetensors headers when they can be read. A gated repo lists its
files and their sizes to anyone but serves the files only after its terms are accepted, so
without a token the count falls back to file size over bytes per parameter, and says so.
"""

from __future__ import annotations

import json
import re
import struct
from typing import Any

import httpx

DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "I4": 0.5,
    "U4": 0.5,
}

_SHARD = re.compile(r"-\d{5}-of-\d{5}$")
# Variant tags diffusers and transformers put before the extension. A closed list, so a
# version number in a file name (sd_xl_base_1.0) is not read as a variant.
_VARIANT = re.compile(r"\.(fp16|fp32|bf16|fp8|non_ema|ema_only|ema)$", re.IGNORECASE)
_VARIANT_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}

#: Component keys kept from each pipeline component's config.json: what the diffusion
#: estimator needs to count tokens and activations. Absent keys fall back to defaults.
COMPONENT_KEYS = (
    "_class_name",
    "num_layers",
    "num_single_layers",
    "num_attention_heads",
    "attention_head_dim",
    "hidden_size",
    "d_model",
    "patch_size",
    "patch_size_t",
    "in_channels",
    "latent_channels",
    "z_dim",
    "block_out_channels",
    "sample_size",
    "guidance_embeds",
    "force_upcast",
    "temporal_compression_ratio",
    "spatial_compression_ratio",
    "scale_factor_temporal",
    "scale_factor_spatial",
    "torch_dtype",
)


def safetensors_header(client: httpx.Client, url: str) -> dict[str, Any]:
    """Fetch only the JSON header of a safetensors file (two small Range requests)."""
    r = client.get(url, headers={"Range": "bytes=0-7"})
    r.raise_for_status()
    (n,) = struct.unpack("<Q", r.content[:8])
    r = client.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    r.raise_for_status()
    return json.loads(r.content[:n])


def count_params(header: dict[str, Any]) -> tuple[int, dict[str, int]]:
    total = 0
    by_dtype: dict[str, int] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in info["shape"]:
            n *= d
        total += n
        by_dtype[info["dtype"]] = by_dtype.get(info["dtype"], 0) + n
    return total, by_dtype


def variant_groups(files: list[str]) -> dict[str | None, list[str]]:
    """Files grouped by variant tag; None is the default set."""
    groups: dict[str | None, list[str]] = {}
    for f in files:
        stem = _SHARD.sub("", f.rsplit(".", 1)[0])
        m = _VARIANT.search(stem)
        groups.setdefault(m.group(1).lower() if m else None, []).append(f)
    return groups


def pick_variant(files: list[str]) -> tuple[list[str], str | None]:
    """The one copy to count: the default set if there is one, else fp16, then bf16."""
    groups = variant_groups(files)
    for v in (None, "fp16", "bf16"):
        if v in groups:
            return sorted(groups[v]), v
    if not groups:
        return [], None
    v = sorted(groups, key=str)[0]
    return sorted(groups[v]), v


def _bytes_per_param(groups: dict[str | None, list[str]], chosen: str | None,
                     sizes: dict[str, int | None]) -> tuple[float, str]:
    """Bytes per parameter for a set of files whose headers cannot be read."""
    if chosen in _VARIANT_BYTES:
        return _VARIANT_BYTES[chosen], f"the {chosen} variant"
    if chosen is None and "fp16" in groups:
        default = sum(sizes.get(f) or 0 for f in groups[None])
        half = sum(sizes.get(f) or 0 for f in groups["fp16"])
        if half and default > 1.5 * half:
            return 4, "twice the size of its fp16 variant, so fp32"
        return 2, "the size of its fp16 variant"
    return 2, "assumed 16-bit, the usual for current checkpoints"


def count_files(
    client: httpx.Client,
    base: str,
    files: list[str],
    sizes: dict[str, int | None],
    groups: dict[str | None, list[str]] | None = None,
    chosen: str | None = None,
) -> tuple[int, dict[str, int], str]:
    """(params, params by dtype, how they were counted) for one copy of the weights."""
    total, by_dtype = 0, {}
    try:
        for f in files:
            n, d = count_params(safetensors_header(client, f"{base}/{f}"))
            total += n
            for k, v in d.items():
                by_dtype[k] = by_dtype.get(k, 0) + v
        return total, by_dtype, "safetensors headers"
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (401, 403):
            raise
    nbytes = sum(sizes.get(f) or 0 for f in files)
    per, why = _bytes_per_param(groups or {chosen: files}, chosen, sizes)
    params = int(nbytes / per)
    label = {4: "F32", 2: "BF16", 1: "F8_E4M3"}.get(int(per), "unknown")
    return params, {label: params}, f"file sizes ({why}); the headers are gated"


def _get_json(client: httpx.Client, url: str) -> dict[str, Any] | None:
    try:
        r = client.get(url)
        r.raise_for_status()
        out = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def role(name: str, cls: str | None) -> str:
    """What a pipeline component does, which decides when it must be on the GPU."""
    n, c = name.lower(), (cls or "")
    if n.startswith(("transformer", "unet", "prior")) or c.endswith(
        ("Transformer2DModel", "Transformer3DModel", "UNet2DConditionModel", "UNet2DModel")
    ):
        return "denoiser"
    if n.startswith("text_encoder"):
        return "text_encoder"
    if n.startswith("vae") or c.startswith("Autoencoder"):
        return "vae"
    return "other"


_COMPONENT_FOLDER = re.compile(r"^(transformer(_\d)?|unet|prior|text_encoder(_\d)?|vae)$")


def pipeline_components(
    client: httpx.Client, base: str, sizes: dict[str, int | None]
) -> dict[str, Any]:
    """The components of a diffusers pipeline, each counted once.

    model_index.json names the components; the other folders (a spare VAE, ONNX and
    OpenVINO exports) are left out. When the index is gated, folders with the usual
    component names stand in for it."""
    index = _get_json(client, f"{base}/model_index.json")
    folders = sorted({f.split("/")[0] for f in sizes if f.count("/") == 1
                      and f.endswith(".safetensors")})
    if index:
        names = [k for k in folders if isinstance(index.get(k), list)]
        classes = {k: index[k][1] for k in names if len(index[k]) == 2}
    else:
        names = [f for f in folders if _COMPONENT_FOLDER.match(f)]
        classes = {}
    components: dict[str, Any] = {}
    for name in names:
        files = [f for f in sizes if f.startswith(name + "/") and f.count("/") == 1
                 and f.endswith(".safetensors")]
        groups = variant_groups(files)
        chosen_files, chosen = pick_variant(files)
        params, by_dtype, how = count_files(client, base, chosen_files, sizes, groups, chosen)
        config = None
        if f"{name}/config.json" in sizes:
            config = _get_json(client, f"{base}/{name}/config.json")
        cls = classes.get(name) or (config or {}).get("_class_name")
        components[name] = {
            "role": role(name, cls),
            "class": cls,
            "params": params,
            "params_by_dtype": by_dtype,
            "bytes": sum(sizes.get(f) or 0 for f in chosen_files),
            "files": chosen_files,
            "counted_from": how,
            "config": {k: config[k] for k in COMPONENT_KEYS if k in (config or {})},
        }
    return {
        "class": (index or {}).get("_class_name"),
        "index_readable": index is not None,
        "components": components,
    }


_TORCH_FILE = re.compile(r"^pytorch_model([.-].*)?\.bin$")
_OTHER_TORCH = (".pth", ".pt", ".ckpt", ".nemo")


def torch_weights(sizes: dict[str, int | None], torch_dtype: str | None) -> dict[str, Any] | None:
    """Parameters of a repo that ships no safetensors, from its PyTorch files' sizes.

    pytorch_model*.bin when there is one (one variant), else the largest .pth/.pt/.ckpt/
    .nemo at the root: Kokoro's voice packs are small .pt files beside one real checkpoint.
    PyTorch saves float32 unless told otherwise, so that is the assumption when the config
    names no dtype."""
    root = {f: s or 0 for f, s in sizes.items() if "/" not in f}
    bins = [f for f in root if _TORCH_FILE.match(f)]
    if bins:
        files, _ = pick_variant(bins)
    else:
        others = [f for f in root if f.endswith(_OTHER_TORCH)]
        if not others:
            return None
        files = [max(others, key=lambda f: root[f])]
    dtype = (torch_dtype or "float32").lower()
    per = 2 if dtype in ("float16", "bfloat16", "half") else 4
    nbytes = sum(root[f] for f in files)
    return {
        "params": int(nbytes / per),
        "files": sorted(files),
        "bytes": nbytes,
        "counted_from": f"file sizes at {per} bytes per parameter ({dtype})",
    }
