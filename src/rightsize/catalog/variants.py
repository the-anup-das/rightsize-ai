"""Quantized copies of a model that already exist on the Hub (F1).

A model card can name its base model and how it relates to it (``base_model_relation:
quantized``), and the Hub turns that into a filter: ``base_model:quantized:Qwen/Qwen3-4B``
lists the GGUF, MLX, AWQ, FP8 and bitsandbytes repos made from Qwen3-4B, most downloaded
first. Each is told apart by the conventions in data/models/variants.yaml (Hub tag, library
name, a pattern in the repo name), so a new format or a publisher's naming habit is a data
edit. GGUF repos hold one file per quantization; each becomes its own variant, sized from
the file listing.

The link is only as good as the card that declares it. Fine-tunes and speculative-decoding
drafts sometimes call themselves quantizations too, so every variant says whether its name
is the base model's name plus format words (``name_matches_base``).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from rightsize._data import load_yaml
from rightsize.catalog import gguf
from rightsize.types import ModelRef, Variant

HUB = "https://huggingface.co"
_EXPAND = ("downloads", "library_name", "tags", "gated")


def _table() -> dict[str, Any]:
    return load_yaml("models/variants.yaml")


def classify(repo: str, tags: list[str] | None, library: str | None) -> dict[str, Any] | None:
    """The format record a repo matches: by Hub tag, then library, then name; the first
    format in the table wins at each step."""
    formats = _table()["formats"]
    have = {t.lower() for t in tags or []}
    for f in formats:
        if have & {t.lower() for t in f.get("tags") or []}:
            return f
    lib = (library or "").lower()
    for f in formats:
        if lib and lib in {x.lower() for x in f.get("libraries") or []}:
            return f
    name = repo.rsplit("/", 1)[-1]
    for f in formats:
        if any(re.search(p, name, re.I) for p in f.get("name_patterns") or []):
            return f
    return None


def quant_of(fmt: dict[str, Any] | None, repo: str) -> str | None:
    """The quantization a repo's name states, by its format's patterns."""
    name = repo.rsplit("/", 1)[-1]
    for p in (fmt or {}).get("quant_patterns") or []:
        m = re.search(p, name, re.I)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    return None


def _tokens(name: str) -> list[str]:
    text = name.lower()
    for p in _table()["strip_patterns"]:
        text = re.sub(rf"(?<![a-z0-9])(?:{p})(?![a-z0-9])", "-", text)
    return [t for t in re.split(r"[-_. ]+", text) if t]


def matches_base(repo: str, base: str) -> bool:
    """Whether a variant's name is its base model's name plus format words.

    Qwen_Qwen3-4B-GGUF (bartowski puts the org first) and Qwen3-4B-unsloth-bnb-4bit match
    Qwen/Qwen3-4B; a fine-tune published as a quantization, such as
    Qwen3-4B-Roleplay-v2-GGUF, does not."""
    org, _, name = base.partition("/")
    want = _tokens(name)
    have = _tokens(repo.rsplit("/", 1)[-1])
    if have == want:
        return True
    extra = have[: len(have) - len(want)]
    org_words = set(re.split(r"[-_. ]+", org.lower()))
    return have[len(extra):] == want and bool(extra) and set(extra) <= org_words


def base_of(client: httpx.Client, repo: str) -> str:
    """The model a quantized repo was made from, by its card's base_model link; the repo
    itself when it declares none."""
    r = client.get(f"{HUB}/api/models/{repo}", params={"expand[]": ["tags", "cardData"]})
    r.raise_for_status()
    for tag in r.json().get("tags") or []:
        if tag.startswith("base_model:quantized:"):
            return tag.removeprefix("base_model:quantized:")
    return repo


def _cache_path(key: str) -> Path:
    base = Path(os.environ.get("RIGHTSIZE_CACHE_DIR", Path.home() / ".cache" / "rightsize"))
    return base / "variants" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', key)}.json"


def _params(base: str) -> int | None:
    from rightsize.catalog.hub import facts

    try:
        return facts(base).params_total
    except (httpx.HTTPError, FileNotFoundError, ValueError):
        return None


def variants(
    repo: str,
    *,
    formats: list[str] | None = None,
    limit: int = 20,
    files: bool = True,
    token: str | None = None,
    offline: bool = False,
    ttl_s: float = 24 * 3600,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
    use_cache: bool | None = None,
    base_params: int | None = None,
) -> list[Variant]:
    """Quantized copies of ``repo`` (or of its base, when ``repo`` is itself a quantized
    copy), most downloaded first within each format.

    ``formats`` keeps only those formats (ids from data/models/variants.yaml). ``limit``
    caps the repos listed, not the files: a GGUF repo contributes one variant per
    quantization. ``files=False`` skips the per-repo file listing, so there are no sizes
    and each GGUF repo is one variant. Cached for a day; ``offline`` answers from the cache
    only."""
    from rightsize.catalog.hub import _headers

    offline = offline or os.environ.get("RIGHTSIZE_OFFLINE") == "1"
    use_cache = transport is None if use_cache is None else use_cache
    key = f"{repo}__{'+'.join(sorted(formats or []))}__{limit}__{int(files)}"
    path = _cache_path(key)
    if use_cache:
        try:
            if offline or time.time() - path.stat().st_mtime <= ttl_s:
                rows = json.loads(path.read_text(encoding="utf-8"))
                return [Variant.model_validate(r) for r in rows]
        except (OSError, ValueError):
            pass
    if offline:
        raise FileNotFoundError(f"no cached variants for {repo} and offline=True")

    table = _table()
    by_id = {f["id"]: f for f in table["formats"]}
    known = {p["name"].lower() for p in table["publishers"]}
    with httpx.Client(headers=_headers(token), timeout=timeout, follow_redirects=True,
                      transport=transport) as c:
        base = base_of(c, repo)
        params: list[tuple[str, str]] = [
            ("filter", f"base_model:quantized:{base}"), ("sort", "downloads"),
            ("direction", "-1"), ("limit", str(limit)),
            *(("expand[]", e) for e in _EXPAND),
        ]
        if formats and len(formats) == 1 and (by_id.get(formats[0]) or {}).get("tags"):
            params.append(("filter", by_id[formats[0]]["tags"][0]))  # narrowed on the Hub
        r = c.get(f"{HUB}/api/models", params=params)
        r.raise_for_status()
        rows = r.json()
        total = base_params if base_params is not None else (
            _params(base) if files and transport is None else None)
        out: list[Variant] = []
        for row in rows:
            rid = row["id"]
            fmt = classify(rid, row.get("tags"), row.get("library_name"))
            if formats and (fmt is None or fmt["id"] not in formats):
                continue
            common = {
                "format": fmt["id"] if fmt else "unknown",
                "publisher": rid.split("/")[0],
                "official": rid.split("/")[0].lower() == base.split("/")[0].lower(),
                "known_publisher": rid.split("/")[0].lower() in known,
                "name_matches_base": matches_base(rid, base),
                "downloads": row.get("downloads"),
                "runtimes": list((fmt or {}).get("runtimes") or []),
                "gated": bool(row.get("gated")),
            }
            if not files:
                out.append(Variant(ref=ModelRef(repo=rid), quant=quant_of(fmt, rid), **common))
                continue
            info = c.get(f"{HUB}/api/models/{rid}", params={"blobs": "true"})
            if info.status_code >= 400:
                continue
            sizes = {s["rfilename"]: s.get("size") for s in info.json().get("siblings") or []}
            if common["format"] == "gguf":
                groups, _ = gguf.weight_files(sizes, rid)
                for q, g in groups.items():
                    out.append(Variant(
                        ref=ModelRef(repo=rid, file=g["files"][0]), quant=q,
                        size_bytes=g["bytes"], bits_per_weight=_bpw(g["bytes"], total),
                        files=g["files"], **common,
                    ))
                continue
            nbytes = sum(s or 0 for f, s in sizes.items() if f.endswith(".safetensors")) or None
            out.append(Variant(
                ref=ModelRef(repo=rid), quant=quant_of(fmt, rid), size_bytes=nbytes,
                bits_per_weight=_bpw(nbytes, total), **common,
            ))
    order = {f["id"]: i for i, f in enumerate(table["formats"])}
    out.sort(key=lambda v: (order.get(v.format, len(order)), not v.official,
                            not v.known_publisher, -(v.downloads or 0), v.ref.repo,
                            v.size_bytes or 0))
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([v.model_dump(mode="json") for v in out]),
                            encoding="utf-8")
        except OSError:
            pass
    return out


def _bpw(nbytes: int | None, params: int | None) -> float | None:
    return round(nbytes * 8 / params, 2) if nbytes and params else None
