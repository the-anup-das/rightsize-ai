"""Generate the example notebooks in notebooks/ from plain Python (no jupyter needed).

Run:  uv run python scripts/make_notebooks.py
Each notebook is a walkthrough of one feature. Cells are kept short and runnable top to bottom.
Outputs are not stored; run them locally to see results.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS: dict[str, list[dict]] = {}

NOTEBOOKS["00-quickstart.ipynb"] = [
    md("""
# Rightsize quickstart

Pick a model, pick your hardware, get a runnable plan. This notebook runs the whole first slice:
detect the machine, predict memory and speed for a few GGUF quants, then (optionally) produce and
evaluate the files.

Setup once from the repo root:

```bash
uv sync --group dev --extra llamacpp      # torch CPU + transformers for conversion
rightsize tools install llama.cpp         # pinned llama.cpp binaries + converter into .tools/
```
"""),
    code("""
import rightsize
rightsize.__version__
"""),
    md("## 1. What machine is this?"),
    code("""
from rightsize.hardware import detect
dev = detect()
dev
"""),
    md("## 2. What does the model need? (no download; reads Hub headers only)"),
    code("""
from rightsize.catalog import facts
fx = facts("Qwen/Qwen3-1.7B")
fx.params_total / 1e9, fx.num_layers, fx.num_kv_heads, fx.head_dim, fx.dtype
"""),
    code("""
from rightsize.fit import estimate, predicted_file_gb
for q in ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]:
    r = estimate(fx, q, dev, ctx=8192)
    print(f"{q:7s} file {predicted_file_gb(fx, q):.2f} GB  vram {r.vram_gb:.2f} GB  {r.verdict.value:7s}  {r.speed} {r.speed_unit}")
"""),
    md("""
## 3. Make the files (needs llama.cpp and the `llamacpp` extra)

`quantize_model` renders each step from the recipe registry, runs it, and records what it measured
next to what it predicted. Start with a dry run to see the commands.
"""),
    code("""
from rightsize.execution import quantize_model
m = quantize_model("Qwen/Qwen3-1.7B", ["Q4_K_M"], dry_run=True)
[s.recipe_id for s in m.steps]
"""),
    code("""
# Real run: downloads ~4 GB, converts on CPU, quantizes, then measures KL divergence vs the 16-bit file.
# m = quantize_model("Qwen/Qwen3-1.7B", ["Q4_K_M", "Q8_0"], imatrix=True, evaluate=True, eval_chunks=50)
# m.gate, m.artifacts
"""),
    md(
        "Same thing from the shell: `rightsize quantize Qwen/Qwen3-1.7B --quant Q4_K_M --imatrix --eval`."
    ),
]

NOTEBOOKS["01-model-catalog.ipynb"] = [
    md("""
# F1. Model catalog: facts without downloading

`facts()` reads `config.json` and the safetensors **headers** of a Hub repo over HTTP Range requests
(a few KB), never the weights. Results are cached for 7 days under `~/.cache/rightsize/models/`.
"""),
    code("""
from rightsize.catalog import facts
fx = facts("Qwen/Qwen3-4B")
fx.model_dump(exclude={"extra"})
"""),
    md("Everything the KV-cache formula needs is here: layers, KV heads, head dim, max context."),
    code("""
{k: fx.extra[k] for k in ("model_type", "params_by_dtype", "shards", "vocab_size", "tie_word_embeddings")}
"""),
    md("Gated repos work with `HF_TOKEN` in the environment. Offline mode serves the cache only:"),
    code("""
facts("Qwen/Qwen3-4B", offline=True).params_total
"""),
    md("""
## How the header read works

A safetensors file starts with 8 bytes (little-endian length N) followed by N bytes of JSON that
lists every tensor's dtype and shape. Two Range requests are enough to count parameters.
"""),
    code("""
import httpx
from rightsize.catalog import safetensors_header
with httpx.Client(follow_redirects=True) as c:
    hdr = safetensors_header(c, "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors")
list(hdr.items())[:3]
"""),
]

NOTEBOOKS["02-hardware.ipynb"] = [
    md("""
# F2. Hardware: catalogue, presets, detection and measurement

Four ways to name a device, in the order they are trusted: a bandwidth you measured on this
machine, a curated preset, the ingested catalogue, and whatever you pass by hand.
"""),
    md("""
## The catalogue

259 accelerators, ingested from Hugging Face's SKU table by `scripts/ingest_hf_hardware.py`,
joined to bandwidth from `scripts/ingest_bandwidth.py`. Every record keeps its source URL.
"""),
    code("""
from rightsize.hardware import catalog, presets, get, detect, from_hf
cat = catalog()
have = [d for d in cat.values() if d.bandwidth_gbps]
print(f"{len(cat)} devices, {len(have)} with a memory bandwidth")
"""),
    md("""
Bandwidth is computed from the two facts a vendor publishes, so it carries its derivation
rather than being copied from anyone's table.
"""),
    code("""
for name in ("RTX 4090 24GB", "Apple M4 Max 64GB", "Arc B580 12GB", "A100 80GB"):
    d = cat[name]
    print(f"{name:20s} {d.bandwidth_gbps:>7.1f} GB/s   {d.provenance.note or d.provenance.source_url[:48]}")
"""),
    md("""
Size variants are separate records, because a model name on its own does not identify a
card: an A100 40GB is 1555 GB/s and an 80GB is 2039.
"""),
    code("""
for name in ("A100 40GB", "A100 80GB", "RTX 3060 8GB", "RTX 3060 12GB"):
    print(f"{name:16s} {cat[name].bandwidth_gbps:>7.1f} GB/s")
"""),
    md("""
## Presets and lookup

Presets are the curated setups, with the OS and usable fraction someone actually checked.
`get()` searches those first, then falls through to the catalogue.
"""),
    code("""
for name, d in presets().items():
    print(f"{name:24s} {d.memory_gib:6.0f} GiB {d.bandwidth_gbps:6.0f} GB/s  {d.compute_arch or '':10s} {d.provenance.source_url}")
"""),
    code("""
get("4090"), get("RTX 5080").name, get("RTX 5080").bandwidth_gbps
"""),
    md("""
## Units

Device memory is GiB, as the vendor and `nvidia-smi` state it. Model sizes are decimal GB,
as Hugging Face lists them. `memory_gb` converts, and the fit engine only uses that.
"""),
    code("""
d = get("RTX 4070 Ti SUPER")
print(f"{d.memory_gib} GiB is {d.memory_gb} GB; usable at {d.usable_fraction:.0%} = {d.memory_gb * d.usable_fraction:.2f} GB")
"""),
    md("""
## Detection, and your own hardware

`detect()` reads this machine. `from_hf("username")` reads the hardware someone saved on
their Hugging Face profile, which resolves against the same catalogue.
"""),
    code("""
detect()
"""),
    code("""
# needs network; any public profile with saved hardware works
# for d in from_hf("julien-c"):
#     print(f"{d.name:24s} {d.memory_gib:>6} GiB  {d.bandwidth_gbps or '?'} GB/s")
"""),
    md("""
## When there is no number to look up

NVIDIA publishes a bus width but no bandwidth for laptop GPUs, because the memory speed is
the laptop maker's choice. `rightsize bench` measures it instead, by running `llama-bench`
and solving the speed model backwards, then remembers it for this machine:

```
tok/s = efficiency * bandwidth / (active weights + KV)
```
"""),
    code("""
from rightsize.execution.bench import bandwidth_from_throughput, measured_bandwidth
from rightsize.catalog import facts
# a real run on an RTX 4070 Ti SUPER: 362.7 tok/s on Qwen3-1.7B Q4_K_M at ctx 512
fx = facts("Qwen/Qwen3-1.7B")
print(f"{bandwidth_from_throughput(fx, 'Q4_K_M', 362.7, ctx=512):.0f} GB/s derived, published 672")
print("measured on this machine:", measured_bandwidth(detect().name))
"""),
    md(
        "Presets live in `data/hardware/presets.yaml`, bandwidth in `bandwidth.yaml` "
        "(hand-checked) and `bandwidth_wikipedia.yaml` (generated). Add a card with its "
        "source URL and it becomes usable everywhere."
    ),
]

NOTEBOOKS["03-fit-engine.ipynb"] = [
    md("""
# F3. Fit engine: where the memory goes

`estimate()` returns a `FitResult` with a breakdown (weights, KV cache, overhead), a verdict,
a bandwidth-bound speed estimate, a confidence and the `formula_id` that produced it.
"""),
    code("""
from rightsize.catalog import facts
from rightsize.fit import estimate, kv_cache_gb, gguf_bpw
from rightsize.hardware import get
fx = facts("Qwen/Qwen3-4B")
dev = get("RTX 4070 Ti SUPER")
r = estimate(fx, "Q4_K_M", dev, ctx=8192)
r
"""),
    md(
        "Real bits per weight, not nominal: this is why predicted file sizes match `llama-quantize` output."
    ),
    code("""
{q: gguf_bpw(q) for q in ["Q4_K", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "IQ4_XS"]}
"""),
    md("## Context length changes the answer"),
    code("""
for ctx in [2048, 8192, 32768, 131072]:
    r = estimate(fx, "Q4_K_M", dev, ctx=ctx)
    print(f"ctx {ctx:6d}: kv {r.breakdown['kv_cache']:.2f} GB  total {r.vram_gb:.2f} GB  {r.verdict.value}")
"""),
    md("## Golden check: Llama 3.1 70B KV cache at 128K context is about 43 GB"),
    code("""
from rightsize.types import ModelFacts, ModelRef, Family
llama70 = ModelFacts(ref=ModelRef(repo="meta-llama/Llama-3.1-70B"), family=Family.llm,
                     params_total=70_553_706_496, num_layers=80, num_kv_heads=8, head_dim=128)
kv_cache_gb(llama70, 131072)
"""),
]

NOTEBOOKS["05-recipes.ipynb"] = [
    md("""
# F5. Recipes: the registry renders commands

A recipe is YAML in `data/recipes/<framework>/<stage>.yaml` with typed inputs and a template.
Rendering needs nothing installed; running is a separate, opt-in step.
"""),
    code("""
from rightsize.registry import all_recipes, get, render
for r in all_recipes().values():
    print(f"{r.id:22s} {r.stage:9s} tested {r.version_tested}  {r.source_doc_url}")
"""),
    code("""
step = render(get("llama.cpp/quantize"), quantize_bin="llama-quantize", input_gguf="model-bf16.gguf",
              output_gguf="model-Q4_K_M.gguf", quant="Q4_K_M", imatrix="model-imatrix.gguf")
print(step.text)
step.argv
"""),
    md(
        "Inputs are validated against the recipe: unknown quant types or missing paths fail before anything runs."
    ),
    code("""
from rightsize.registry import TemplateError
try:
    render(get("llama.cpp/quantize"), quantize_bin="q", input_gguf="a", output_gguf="b", quant="Q4_K_ULTRA")
except TemplateError as e:
    print(e)
"""),
]

NOTEBOOKS["08-quantize-and-evaluate.ipynb"] = [
    md("""
# F8. Quantize, evaluate, record

The end-to-end path on a small model. Steps: predict, download, convert to 16-bit GGUF, compute an
importance matrix, quantize, then measure KL divergence and peak VRAM against the 16-bit reference.
Every predicted number is written next to the measured one in `runs/<run>/manifest.json`.

Needs: `uv sync --extra llamacpp` and `rightsize tools install llama.cpp`. Takes a few minutes for Qwen3-0.6B.
"""),
    code("""
from rightsize.execution import quantize_model
m = quantize_model("Qwen/Qwen3-0.6B", ["Q4_K_M", "Q8_0"], imatrix=True, evaluate=True, eval_chunks=20)
m.status, m.gate
"""),
    md("## Predicted vs measured"),
    code("""
for step in m.steps:
    for meas in step.measurements:
        if meas.predicted is not None:
            err = (meas.value - meas.predicted) / meas.predicted * 100
            print(f"{step.recipe_id:22s} {meas.kind:14s} {meas.note or '':16s} predicted {meas.predicted:8.3f}  measured {meas.value:8.3f}  {err:+.1f}%")
"""),
    md("## The quality gate"),
    code("""
import json
json.dumps({q: {k: v for k, v in g.items() if k != "thresholds"} for q, g in m.gate.items()}, indent=2)
"""),
    md(
        "Measurements also append to `runs/measurements.jsonl`, the local seed for the calibration loop (F9). Nothing is uploaded."
    ),
    code("""
from pathlib import Path
print(Path("runs/measurements.jsonl").read_text()[:800])
"""),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
