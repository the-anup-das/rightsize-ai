# Example notebooks

One walkthrough per feature. Numbers follow the plan files in `docs/plans/`. Notebooks are generated from `scripts/make_notebooks.py` so they stay in sync with the API; edit the script, not the `.ipynb`, then run `uv run python scripts/make_notebooks.py`.

| Notebook | Feature | Needs |
|---|---|---|
| [00-quickstart](00-quickstart.ipynb) | detect, estimate, quantize in one pass | network for Hub metadata; llama.cpp for the last cell |
| [01-model-catalog](01-model-catalog.ipynb) | F1 facts from Hub headers | network |
| [02-hardware](02-hardware.ipynb) | F2 presets and detection | nothing |
| [03-fit-engine](03-fit-engine.ipynb) | F3 memory breakdown, context sweep, golden check | network for facts |
| [05-recipes](05-recipes.ipynb) | F5 registry and rendering | nothing |
| [08-quantize-and-evaluate](08-quantize-and-evaluate.ipynb) | F8 end-to-end on Qwen3-0.6B with the KL-divergence gate | `.tools/llama.cpp`, `uv sync --extra llamacpp` |

Coming with their features: 04 rules and ranking, 06 MCP server, 07 cloud fallback, 09 calibration, 10 cloud connectors.

Run them with any Jupyter front end from the repo root, for example:

```bash
uv run --with jupyterlab jupyter lab notebooks/
```

`tests/test_notebooks.py` checks that every notebook is valid and that each code cell parses.
