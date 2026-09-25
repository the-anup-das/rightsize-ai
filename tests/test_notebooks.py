"""Notebooks stay valid and their code cells parse, without executing them."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))


def test_notebooks_exist() -> None:
    names = {p.name for p in NOTEBOOKS}
    assert {
        "00-quickstart.ipynb",
        "01-model-catalog.ipynb",
        "02-hardware.ipynb",
        "03-fit-engine.ipynb",
        "05-recipes.ipynb",
        "08-quantize-and-evaluate.ipynb",
    } <= names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_is_valid_and_code_parses(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"], "empty notebook"
    assert nb["cells"][0]["cell_type"] == "markdown", "start with a title cell"
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(src, filename=f"{path.name}#cell{i}")
            assert cell["outputs"] == [], "notebooks are committed without outputs"
