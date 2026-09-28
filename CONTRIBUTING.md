# Contributing to Rightsize

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:the-anup-das/rightsize-ai.git
cd rightsize-ai
uv sync --group dev --extra mcp
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Ground rules (enforced by tests in `tests/budgets/`)

1. **No ML dependencies in the core.** `rightsize` depends on `httpx`, `pydantic`, `pyyaml` only. Anything heavier goes behind an extra and is imported inside the function that needs it.
2. **No network or file I/O at import time.**
3. **Data is data.** Hardware specs, quant tables, rules and recipes live under `data/` as JSON/YAML with `source_url` and `fetched_at` on every record. Do not hard-code numbers in Python.
4. **Every estimate is explainable.** A `FitResult` always carries `confidence` and `formula_id`; a fired rule always carries `source_url`.

## Adding a framework or toolkit

You do not write Python. A framework is one folder, `data/recipes/<name>/`:

- `framework.yaml` says what the toolkit is for, where it runs, how it installs and how its
  steps join a plan (`data/schema/framework.schema.json`). A trainer declares a `finetune`
  role: the modes it has a recipe for, the recipe, what it writes (a merged model or an
  adapter), how QLoRA holds the base weights, and `default_for`, the device vendors it is the
  default trainer on (`"*"` for any vendor no other framework names; `priority` breaks a
  tie). `defaults` gives the file and binary names its recipes read when a plan does not set
  them, with `{slug}` for the model and `{quant}` for the quantization.
- one YAML file per recipe (`data/schema/recipe.schema.json`): typed inputs, the command or
  config template, hardware constraints and the doc URL you took the flags from.

For example, a trainer that should become the default on NVIDIA needs this and one recipe:

```yaml
name: zoomtune
title: ZoomTune
summary: what it is for, in a sentence
stages: [finetune]
hardware: {vendors: [nvidia]}
install: {kind: pip, check: zoomtune, line: pip install zoomtune}
finetune:
  modes: [lora, qlora]
  recipe: zoomtune/train
  qlora_quant: {method: bnb, variant: nf4}
  writes: merged
  default_for: [nvidia]
  priority: 20
homepage: https://example.org/zoomtune
source_doc_url: https://example.org/zoomtune/docs
```

CI validates both files, renders every recipe, checks that each `defaults` key is an input
one of the framework's recipes reads, and regenerates `docs/frameworks/`
(`python -m rightsize.registry.docs`). `tests/test_frameworks.py` shows a trainer added this
way taking over plans without any change to rightsize.

Third-party packages ship the same folder layout through the `rightsize.recipes` entry point
(a directory, or a callable returning dicts: recipes have a `template`, descriptors do not).
A plugin may add frameworks and recipes but never replace a bundled one.

A recipe also says how it runs: a config recipe names the `file` it is written to and the
`run` line that executes it (Python configs default to `python {file}`), and `writes` lists
the inputs that name what the step produces, which `rightsize run` checks and measures. List
the Python packages in `framework.yaml`'s `install.packages` and `rightsize tools install
<name>` gives the toolkit its own environment. A toolkit whose steps need more than a
command (binaries of its own, a download first) can register a runner under the
`rightsize.runners` entry point; `execution/runner.py` has the protocol and llama.cpp's.

A recipe that makes a weight format lists it under `targets`, and `rightsize quantize MODEL
--to NAME` then runs it:

```yaml
targets:
  - {name: fp8, size_from: fp8, embedding_bits: 16, head_bits: 16}
```

`size_from` names the `data/quants/formats.yaml` entry that predicts the output size;
`inputs` sets the recipe inputs that select the format; `embedding_bits` and `head_bits`
say what the toolkit leaves unquantized (llm-compressor and bitsandbytes skip everything but
Linear layers); `weights` is a glob for the files the prediction covers when the output
folder holds more than the quantized model. Run it once and compare the manifest's measured
size with the prediction before marking the recipe `verified: run`.

## Adding a rule

Rules live in `data/rules/*.yaml` and require `id`, `applies_to`, `condition`, `effect`, `source_url` and a `test`. A rule without a test fails schema validation. See `docs/plans/F04-rules-engine.md`.

## Feature work

Each feature has a plan with a TODO checklist in `docs/plans/`. Pick an unchecked item, open a PR that ticks it. Keep PRs to one feature.

## Commit style

Imperative subject, under 72 characters, body explains why. Reference the feature (`F3:`) when applicable.
