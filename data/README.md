# rightsize-data (seed)

Everything Rightsize knows that is a *number* or a *table* lives here, not in Python. This folder is the seed of the `rightsize-data` package, which will get its own version and weekly release once the first ingest lands (run 1 of the roadmap). The core package pins a minimum data version and refreshes with `rightsize data update` into `~/.cache/rightsize/`.

| Folder | Contents | Source (to ingest) | Owner |
|---|---|---|---|
| `hardware/` | GPUs, CPUs, Apple chips: memory, bandwidth, arch, tflops, msrp, usable fraction; presets | HF `hardware.ts`, TechPowerUp API (licence to verify), `dbgpu`, Apple newsroom | F2 |
| `quants/` | Real bits-per-weight per GGUF type; per-architecture KV overrides (MLA, sliding window, hybrid); MLX/bnb/AWQ/GPTQ/FP8 descriptors | `@huggingface/gguf` `quant-descriptions.ts`, model configs | F1, F3 |
| `rules/` | Hard gates and penalties with `source_url` and a test each | HF/vLLM/SGLang quantization matrices, toolkit docs | F4 |
| `recipes/` | Renderable command/config templates per framework and stage | Each toolkit's docs, pinned by version | F5 |
| `quality/` | Base-model quality and quant penalty tables | Unsloth KL tables, Artificial Analysis, own `llama-perplexity` runs | F4 |
| `cloud/` | GPU rental price ladder (hourly cache) | ComputePrices API, RunPod GraphQL | F7 |
| `schema/` | JSON Schemas every file above must validate against | this repo | all |

## Rules for every record

- `source_url` and `fetched_at` are required. No number without a source.
- Prefer ingest scripts (`scripts/ingest_*.py`, to come) over hand edits; hand tables are marked `"hand": true`.
- Changing data never requires a code release.
