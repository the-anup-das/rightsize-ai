"""Bits per weight for any weight-format name the fit engine accepts (F3).

Two tables answer: GGUF and ggml types (quants/gguf_bpw.yaml, plus llama.cpp's measured
file types), and every other format (quants/formats.yaml: bitsandbytes, torchao, AWQ,
GPTQ, MLX, FP8, NVFP4, MXFP4, 16- and 32-bit floats). Names are matched without regard to
case, and through each format's aliases, so "nf4", "bnb-4bit" and "4bit" are one format.
"""

from __future__ import annotations

from dataclasses import dataclass

from rightsize._data import load_yaml


@dataclass(frozen=True)
class Format:
    id: str
    bpw: float
    source: str  # where the figure came from, for the estimate's notes


def _formats() -> list[dict]:
    return load_yaml("quants/formats.yaml")["formats"]


def known() -> list[str]:
    """Every non-GGUF format id, for error messages and CLI help."""
    return [f["id"] for f in _formats()]


def lookup(name: str, *, gguf: str = "file") -> Format:
    """The format a name refers to.

    ``gguf`` says which GGUF figure to use. ``"file"`` is llama.cpp's measured file type,
    right for an LLM whose quant name is a recipe of several tensor types (Q4_K_M keeps
    some tensors at Q6_K). ``"tensor"`` is the block size of the named type itself, right
    for files that apply one type throughout, as the diffusion GGUFs do: city96's
    FLUX.1-dev files come out within 0.1 bpw of it, where the LLM figure for Q2_K is 16%
    high.
    """
    key = name.strip().lower()
    for f in _formats():
        if key == f["id"] or key in (a.lower() for a in f.get("aliases", [])):
            return Format(f["id"], float(f["bpw"]), f["layout"])
    if gguf == "tensor":
        found = _gguf_tensor(name)
        if found:
            return found
    from rightsize.fit.llm import gguf_bpw

    try:
        bpw, where = gguf_bpw(name)
    except KeyError:
        raise KeyError(
            f"unknown weight format {name!r}; use a GGUF type (Q4_K_M, Q8_0, ...) or one of "
            + ", ".join(known())
        ) from None
    return Format(name.upper(), bpw, f"GGUF, {where}")


def _gguf_tensor(name: str) -> Format | None:
    tensor = load_yaml("quants/gguf_bpw.yaml")["tensor_types"]
    q = name.strip().upper()
    base = q.rsplit("_", 1)[0] if q.endswith(("_S", "_M", "_L", "_XS", "_XXS")) else q
    for candidate in (q, base):
        if candidate in tensor:
            return Format(q, float(tensor[candidate]), f"GGUF tensor type {candidate}")
    return None


def bpw(name: str, *, gguf: str = "file") -> float:
    return lookup(name, gguf=gguf).bpw
