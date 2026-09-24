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

You do not write Python. Add a recipe under `data/recipes/<framework>/<stage>.yaml` following `data/schema/recipe.schema.json` (schema lands with F5), with typed inputs, the command or config template, the install extra, hardware constraints and the doc URL you took the flags from. CI renders every recipe; a nightly job dry-runs them against the installed toolkit.

Third-party packages can ship recipes too, via the `rightsize.recipes` entry point. See `docs/plans/F05-framework-registry.md`.

## Adding a rule

Rules live in `data/rules/*.yaml` and require `id`, `applies_to`, `condition`, `effect`, `source_url` and a `test`. A rule without a test fails schema validation. See `docs/plans/F04-rules-engine.md`.

## Feature work

Each feature has a plan with a TODO checklist in `docs/plans/`. Pick an unchecked item, open a PR that ticks it. Keep PRs to one feature.

## Commit style

Imperative subject, under 72 characters, body explains why. Reference the feature (`F3:`) when applicable.
