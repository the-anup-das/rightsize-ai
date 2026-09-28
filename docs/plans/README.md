# Rightsize plans

One file per major feature. Each has the same sections: Goal, Why it beats what exists, Public API, Data, Design, MVP scope, Follow-up research, Tests, Reuse, TODO. Tick TODO boxes in the PR that lands them.

**Scope of this repo:** the SDK (Python package, CLI, MCP server, data seed). The web platform is a separate project built on the SDK contract described in F06.

| # | Plan | Phase | Status |
|---|---|---|---|
| 00 | [Competitor landscape](00-competitors.md) | research | revised 2026-09-24 (two passes) |
| 01 | [Library and toolkit inventory](01-libraries.md) | research | done 2026-09-25 |
| 02 | [Quantization concepts and method selection](02-quantization-concepts.md) | reference | done 2026-09-25 |
| F1 | [Model catalog](F01-model-catalog.md) | 1 | facts from config, safetensors and GGUF headers; published quantizations (`variants`); curated lists and `search` for all five families; cache revalidated by commit |
| F2 | [Hardware database and detection](F02-hardware.md) | 1 | 259 devices, bandwidth for 201, detection, `bench`, HF profile import |
| F3 | [Fit engine](F03-fit-engine.md) | 1 | LLM weights, KV, speed, fine-tune memory; diffusion, audio, vision, embeddings |
| F4 | [Rules and ranking](F04-rules-engine.md) | 1 | recommend both flows, 31 rules |
| F5 | [Framework registry and recipes](F05-framework-registry.md) | 1 | 24 recipes over 14 frameworks; plugins; generated framework pages |
| F6 | [Surfaces: SDK, CLI, MCP](F06-surfaces.md) | 1 | SDK, CLI and MCP server over the same functions; Plan JSON Schema |
| F7 | [Cloud rental fallback](F07-cloud-fallback.md) | 1 | offers from SkyPilot's catalog; job estimate; plans that rent |
| F8 | [Execution and evaluation gate](F08-execution-eval.md) | 2 | llama.cpp adapter and KL gate, run end to end on Qwen3-1.7B; `rightsize run` carries out a plan (Unsloth QLoRA to GGUF, run here); `quantize --to` makes FP8, W4A16, AWQ, NF4, OpenVINO, Model Optimizer, ONNX and CTranslate2 outputs, each run here and gated |
| F9 | [Calibration loop](F09-calibration.md) | 2 | `calibrate` against Ollama / LM Studio; opt-in local records; overhead refit |
| F10 | [Cloud provider connectors](F10-cloud-connectors.md) | 3 | planned (local-only until then) |

## Build order

| Run | Delivers | Features |
|---|---|---|
| 0 | repo, placeholder package 0.0.1, CI with lightweight budgets, these plans | this |
| 1 | data seed: GPU table, GGUF bits-per-weight, quant x hardware matrices, KV overrides, schemas, `data update`; verify all items marked **(verify)**. Done except the two F5 **(verify)** items | F2, F3, F4 data |
| 2 | catalog + LLM estimator with golden tests; `rightsize estimate` end to end for a GGUF vs Ollama `/api/ps` | F1, F3 (LLM) |
| 3 | hardware DB, presets, detection; 40 rules; ranking; `rightsize recommend` both flows | F2, F4 |
| 4 | 14 MVP recipes, `Plan.render`, generated framework docs, MCP server, Plan JSON Schema | F5, F6 |
| 5 | diffusion / audio / vision / embedding estimators; cloud fallback; tag `v0.1.0` | F3 (rest), F7 |
| 6+ | execution adapters (local only), eval gate, calibration | F8, F9 |
| phase 3 | cloud provider connectors: run a plan step on a rented GPU | F10 |


**Built out of order, on purpose.** The first real work was F8 (quantize and evaluate a model
end to end), which the table puts in run 6. Getting one real result through the whole pipeline
early paid for itself: it found the llama.cpp converter packaging, the GiB/GB mix-up and the
sdist shipping without data. The cost is that the SDK can execute a plan it cannot yet
recommend - F4 is the gap that matters most.

The lessons that cut across features, each with the measurement that taught it, are in
[docs/lessons.md](../lessons.md); the item-by-item record stays in each plan's TODO.

## Lightweight rules (all features)

1. Core deps: `httpx`, `pydantic`, `pyyaml` only. Enforced by `tests/budgets/`.
2. Numbers and tables live in `data/`, with `source_url` and `fetched_at`.
3. Frameworks are recipes that render commands; execution is an opt-in extra loaded lazily.
4. `import rightsize` is stdlib-only (lazy exports); no I/O at import time.
5. Model metadata is fetched on demand via HTTP Range and cached.

## Conventions

- Items marked **(verify)** were not confirmed from a primary source and must be checked before use.
- Every formula gets a `formula_id` and a golden test against a published number.
- Every rule gets a `source_url` and a test. No exceptions.
