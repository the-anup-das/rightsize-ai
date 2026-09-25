# Schemas

JSON Schemas for every file under `data/`, generated from the Pydantic models in
`src/rightsize/data_models.py` by `scripts/export_schemas.py`. The models are the source of
truth; these files exist so someone editing data in an editor that understands JSON Schema,
or from another language, can check their work without reading Python.

| Schema | Covers |
|---|---|
| `gguf_bpw.schema.json` | `quants/gguf_bpw.yaml` |
| `presets.schema.json` | `hardware/presets.yaml` |
| `gpu_catalog.schema.json` | `hardware/gpus.yaml` (generated) |
| `bandwidth.schema.json` | `hardware/bandwidth.yaml`, `hardware/bandwidth_wikipedia.yaml` (generated) |
| `gate_thresholds.schema.json` | `quality/gate_thresholds.yaml` |
| `kv_cache.schema.json` | `runtimes/*/kv_cache.yaml` |
| `recipe.schema.json` | `recipes/*/*.yaml` |

`tests/test_data_schemas.py` enforces three things: every data file validates; no YAML file
under `data/` is left without a schema; and these files match what the models export. The
ingest scripts also validate their output before writing it.

To check the whole tree by hand:

```bash
uv run python -c "from rightsize.data_models import validate_all; print(validate_all() or 'clean')"
```

Still to come with their features: `rule.schema.json` (F4), `offer.schema.json` (F7),
`measurement.schema.json` (F9).
