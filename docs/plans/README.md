# Rightsize plans

One file per major feature. Each has the same sections: Goal, Why it beats what exists, Public API, Data, Design, MVP scope, Follow-up research, Tests, Reuse, TODO. Tick TODO boxes in the PR that lands them.

**Scope of this repo:** the SDK (Python package, CLI, MCP server, data seed). The web platform is a separate project built on the SDK contract described in F06.

| # | Plan | Phase | Status |
|---|---|---|---|
| 00 | [Competitor landscape](00-competitors.md) | research | done 2026-09-24 |
| F1 | [Model catalog](F01-model-catalog.md) | 1 | planned |
| F2 | [Hardware database and detection](F02-hardware.md) | 1 | planned |
| F3 | [Fit engine](F03-fit-engine.md) | 1 | planned |
| F4 | [Rules and ranking](F04-rules-engine.md) | 1 | planned |
| F5 | [Framework registry and recipes](F05-framework-registry.md) | 1 | planned |
| F6 | [Surfaces: SDK, CLI, MCP](F06-surfaces.md) | 1 | placeholder CLI shipped |
| F7 | [Cloud rental fallback](F07-cloud-fallback.md) | 1 | planned |
| F8 | [Execution and evaluation gate](F08-execution-eval.md) | 2 | planned |
| F9 | [Calibration loop](F09-calibration.md) | 2 | planned |

## Build order

| Run | Delivers | Features |
|---|---|---|
| 0 | repo, placeholder package 0.0.1, CI with lightweight budgets, these plans | this |
| 1 | data seed: GPU table, GGUF bits-per-weight, quant x hardware matrices, KV overrides, schemas, `data update`; verify all items marked **(verify)** | F2, F3, F4 data |
| 2 | catalog + LLM estimator with golden tests; `rightsize estimate` end to end for a GGUF vs Ollama `/api/ps` | F1, F3 (LLM) |
| 3 | hardware DB, presets, detection; 40 rules; ranking; `rightsize recommend` both flows | F2, F4 |
| 4 | 14 MVP recipes, `Plan.render`, generated framework docs, MCP server, Plan JSON Schema | F5, F6 |
| 5 | diffusion / audio / vision / embedding estimators; cloud fallback; tag `v0.1.0` | F3 (rest), F7 |
| 6+ | execution adapters, eval gate, calibration | F8, F9 |

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
