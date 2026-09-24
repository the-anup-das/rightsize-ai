# Schemas

JSON Schemas for the data files. Each lands with its feature:

- `device.schema.json`, `preset.schema.json` (F2)
- `quant.schema.json`, `kv_override.schema.json` (F1/F3)
- `rule.schema.json` (F4)
- `recipe.schema.json` (F5)
- `offer.schema.json` (F7)
- `measurement.schema.json` (F9)

Until then, the Pydantic models in `src/rightsize/types.py` are the reference; `Plan.schema_json()` exports the plan contract.
