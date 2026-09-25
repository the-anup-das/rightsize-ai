"""Every data file validates, none dodges validation, and the exported schemas are current."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from rightsize.data_models import json_schemas, validate_all

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_schemas import render  # noqa: E402


def test_every_data_file_validates() -> None:
    problems = validate_all(ROOT / "data")
    assert problems == [], "\n".join(problems)


def test_the_validator_catches_what_it_should(tmp_path: Path) -> None:
    """A clean run proves nothing unless a broken file fails. These are the mistakes that
    have actually happened here: a field left under its old name after a migration, a
    source that is not a link, a record that yields no bandwidth, a file nobody covers."""
    data = tmp_path / "data"
    shutil.copytree(ROOT / "data", data)

    presets = data / "hardware" / "presets.yaml"
    doc = yaml.safe_load(presets.read_text(encoding="utf-8"))
    doc["presets"][0]["memory_gb"] = doc["presets"][0].pop("memory_gib")
    doc["presets"][1]["provenance"]["source_url"] = "techpowerup.com/gpu-specs"
    presets.write_text(yaml.safe_dump(doc), encoding="utf-8")

    bandwidth = data / "hardware" / "bandwidth.yaml"
    bw = yaml.safe_load(bandwidth.read_text(encoding="utf-8"))
    bw["devices"][0].pop("memory_speed_gbps")
    bandwidth.write_text(yaml.safe_dump(bw), encoding="utf-8")

    (data / "hardware" / "mystery.yaml").write_text("x: 1\n", encoding="utf-8")

    found = "\n".join(validate_all(data))
    assert "memory_gb: Extra inputs are not permitted" in found
    assert "must be an https:// link" in found
    assert "needs bandwidth_gbps, or bus_width_bits and memory_speed_gbps" in found
    assert "mystery.yaml: no schema covers this file" in found


def test_exported_schemas_are_current() -> None:
    """data/schema/ is generated; if a model changes and nobody re-exports, this says so."""
    for name, schema in json_schemas().items():
        path = ROOT / "data" / "schema" / f"{name}.schema.json"
        assert path.exists(), f"missing {path.name}: run scripts/export_schemas.py"
        assert path.read_text(encoding="utf-8") == render(schema), (
            f"{path.name} is stale: run scripts/export_schemas.py"
        )


def test_schemas_are_usable_json_schema() -> None:
    for name, schema in json_schemas().items():
        assert schema.get("type") == "object" and schema.get("properties"), name
        json.dumps(schema)  # serialisable without custom encoders
