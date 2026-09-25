"""Load bundled recipes from data/recipes/<framework>/*.yaml (F5)."""

from __future__ import annotations

import functools
from pathlib import Path

from rightsize._data import data_dir
from rightsize.registry.render import validate_template
from rightsize.registry.schema import Recipe


@functools.lru_cache(maxsize=1)
def all_recipes() -> dict[str, Recipe]:
    import yaml

    out: dict[str, Recipe] = {}
    root = data_dir() / "recipes"
    for path in sorted(root.glob("*/*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        recipe = Recipe.model_validate(raw)
        validate_template(recipe.template)
        if recipe.id in out:
            raise ValueError(f"duplicate recipe id {recipe.id} in {path}")
        out[recipe.id] = recipe
    return out


def get(recipe_id: str) -> Recipe:
    try:
        return all_recipes()[recipe_id]
    except KeyError as exc:
        raise KeyError(f"unknown recipe {recipe_id!r}; known: {sorted(all_recipes())}") from exc


def frameworks() -> list[str]:
    return sorted({r.framework for r in all_recipes().values()})


def recipes_dir() -> Path:
    return data_dir() / "recipes"
