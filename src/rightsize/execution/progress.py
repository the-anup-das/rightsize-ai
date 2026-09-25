"""Turn the tools' own output into progress. Each parser sees the text as it streams (including
the partial last line) and returns (current, total | None) or None. Totals are announced by the
tools themselves; the convert step gets its total from the safetensors index.

Formats (llama.cpp b11177):
  llama-quantize    [  12/ 311]  blk.0.attn_k.weight - [...], type = bf16, converting to q4_K ..
  llama-imatrix     compute_imatrix: computing over 140 chunks, n_ctx=512 ...   then  [1]4.9,[2]5.8,
  llama-perplexity  perplexity: calculating perplexity over 100 chunks ...      then  [1]4.9,[2]5.8,
  convert           INFO:hf-to-gguf:blk.0.attn_k.weight, torch.bfloat16 --> BF16, shape = {...}
"""

from __future__ import annotations

import re
from collections.abc import Callable

Parser = Callable[[str], tuple[int, int | None] | None]

_QUANT = re.compile(r"\[\s*(\d+)/\s*(\d+)\]")
_CHUNKS_TOTAL = re.compile(r"over\s+(\d+)\s+chunks")
# a chunk counts as done once its value is printed: "[3]6.23," yes, a trailing "[4]" no
_CHUNK = re.compile(r"\[(\d+)\](?:[-\d.]+|nan|inf)")
_CONVERT_TENSOR = re.compile(r"-->\s+[A-Z0-9_]+,\s+shape")


def quantize_parser() -> Parser:
    def parse(text: str) -> tuple[int, int | None] | None:
        matches = _QUANT.findall(text)
        if not matches:
            return None
        cur, total = matches[-1]
        return int(cur), int(total)

    return parse


def chunks_parser() -> Parser:
    """For llama-imatrix and llama-perplexity: total from the announcement, progress from [n]."""
    state: dict[str, int | None] = {"total": None}

    def parse(text: str) -> tuple[int, int | None] | None:
        if state["total"] is None:
            m = _CHUNKS_TOTAL.search(text)
            if m:
                state["total"] = int(m.group(1))
        nums = _CHUNK.findall(text)
        if not nums:
            return (0, state["total"]) if state["total"] else None
        return int(nums[-1]), state["total"]

    return parse


def convert_parser(total_tensors: int | None) -> Parser:
    def parse(text: str) -> tuple[int, int | None] | None:
        n = len(_CONVERT_TENSOR.findall(text))
        return (n, total_tensors) if n else None

    return parse


def parser_for(recipe_id: str, *, total_tensors: int | None = None) -> Parser | None:
    if recipe_id == "llama.cpp/quantize":
        return quantize_parser()
    if recipe_id in ("llama.cpp/imatrix", "llama.cpp/kld-base", "llama.cpp/kld-eval"):
        return chunks_parser()
    if recipe_id == "llama.cpp/convert":
        return convert_parser(total_tensors)
    return None
