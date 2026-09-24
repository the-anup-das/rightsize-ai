# F6. Surfaces: SDK, CLI, MCP server

Package: `rightsize` (top level), `rightsize.cli`, `rightsize.mcp_server`. Phase 1. Depends on: F1–F5, F7. Also defines the contract the web platform builds on.

## Goal

Expose the same functions three ways: a Python SDK, a CLI with `--json` on every command, and an MCP server so MCP clients (coding agents, IDE assistants, and the official Hugging Face MCP via a Gradio Space) can call Rightsize.

## Why it beats what exists

Most web calculators are UI-only and gguf-parser-go is Go. The MCP servers that exist (local-ai-mcp, gpu-container) are LLM-inference only. Rightsize is the only agent-callable planner that covers fine-tuning, five model families and recipe output. See 00-competitors.md section 4.

## Public API

```python
import rightsize

rightsize.recommend(task, finetune_device, target_device, family, mode="qlora",
                    pinned_framework=None, quality_floor=None, ctx=8192) -> list[Plan]
rightsize.recommend_for_model(model, finetune_device, target_device, mode="qlora", ...) -> list[Plan]
rightsize.estimate(model, quant, runtime, device, mode="infer", ctx=8192) -> FitResult
rightsize.detect() -> Device

plan.render(framework=None) -> list[RenderedStep]
plan.to_json() -> str
Plan.schema_json() -> str          # the platform's TypeScript types come from this
```

Devices accept a `Device`, a preset name (`"RTX 4090"`), or `"detect"`.

**CLI** (argparse, stdlib): `rightsize recommend | estimate | detect | frameworks [name] | plan render <plan.json> --framework X | data update`, all with `--json`. Placeholder subcommands exist today and print the plan they belong to.

**MCP** (`[mcp]` extra, stdio transport): tools `recommend`, `estimate_memory`, `list_hardware`, `detect_hardware`, `list_frameworks`, `render_recipe`. Input and output schemas are generated from the Pydantic types so SDK, CLI and MCP never drift. Also deployed as a Gradio MCP Space in the platform repo so the official HF MCP server can call it.

## Design

- `import rightsize` stays stdlib-only via PEP 562 lazy exports (already implemented; enforced by `tests/budgets/test_imports.py`).
- Public functions are pure: JSON-serialisable inputs, `Plan` outputs, no global state except the on-disk cache.
- CLI prints a compact human table by default; `--json` emits the `Plan` list verbatim. Exit codes: 0 plans found, 2 nothing fits, 1 error.
- `rightsize data update` downloads the latest `rightsize-data` release into `~/.cache/rightsize/data/` and pins its version; `--offline` everywhere.
- MCP tool descriptions are written for agent consumption (what to pass, what comes back, when to call).

## SDK contract for the platform (out of scope here, listed for completeness)

The platform needs, and only needs: `Plan.schema_json()`, `recommend` / `estimate` as pure functions, the MCP server, `hardware.from_hf(username)`, and the registry docs generator. Nothing in this repo assumes a web front end.

## MVP scope

All three surfaces over F1–F5; `data update`; README quickstart with three examples (hardware-first, model-first, estimate).

## Follow-up research

- MCP tool schema conventions that the common MCP clients render best (argument naming, enums, description length).
- Gradio MCP Space constraints (timeouts, cold start) for the platform.

## Tests

- CLI smoke and `--json` shape tests (exist for placeholders; extend per command).
- MCP: tool list and schema snapshot; one round-trip per tool with fixtures.
- Budgets: `import rightsize` stdlib-only and < 150 ms; warm `recommend` < 2 s; wheel < 300 KB; clean install pulls no ML framework (CI `lightweight` job).

## TODO

- [x] Lazy top-level exports; placeholder `recommend` / `estimate` / `detect` raising `NotImplementedYet`
- [x] argparse CLI with all subcommands as placeholders, `--json`, `--version`
- [x] Budget tests and CI lightweight job
- [ ] Wire real functions as F3 / F4 land; device argument coercion (preset name, `"detect"`)
- [ ] Human-readable table output; exit codes
- [ ] `rightsize data update` and `--offline`
- [ ] MCP server with six tools; schema snapshot test
- [ ] `Plan.schema_json()` published as `schema/plan.schema.json` in releases
- [ ] README quickstart with three examples
