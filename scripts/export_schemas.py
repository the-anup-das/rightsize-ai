"""Write data/schema/*.schema.json from the models in rightsize.data_models.

    uv run python scripts/export_schemas.py

The models are the source of truth; these files exist so someone editing data with an
editor that understands JSON Schema, or in another language, can check their work without
reading Python. tests/test_data_schemas.py fails if they fall behind the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rightsize.data_models import json_schemas

OUT = Path(__file__).resolve().parents[1] / "data" / "schema"


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, schema in json_schemas().items():
        path = OUT / f"{name}.schema.json"
        path.write_text(render(schema), encoding="utf-8")
        print(f"wrote {path.relative_to(OUT.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
