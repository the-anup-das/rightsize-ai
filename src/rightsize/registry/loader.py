"""Load bundled recipes from data/recipes/<framework>/*.yaml (F5)."""

from __future__ import annotations

import functools
from pathlib import Path

from rightsize._data import data_dir
from rightsize.registry.render import validate_template
from rightsize.registry.schema import Recipe

#: Third-party packages add recipes under this entry-point group. The entry point loads
#: either a directory of recipe YAML files or a callable returning recipe dicts:
#:
#:     [project.entry-points."rightsize.recipes"]
#:     mytool = "mytool.rightsize:RECIPES_DIR"
ENTRY_POINT_GROUP = "rightsize.recipes"


def _from_yaml_dir(root: Path) -> list[tuple[dict, str]]:
    import yaml

    found = []
    for path in sorted(root.glob("*/*.yaml")) + sorted(root.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as fh:
            found.append((yaml.safe_load(fh), str(path)))
    return found


def _plugins() -> list[tuple[dict, str]]:
    """Recipes from installed packages. A plugin that fails to load is skipped with a
    warning rather than taking the bundled recipes down with it."""
    import warnings
    from importlib.metadata import entry_points

    found: list[tuple[dict, str]] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            target = ep.load()
            if callable(target):
                found.extend((dict(r), f"plugin {ep.name}") for r in target())
            else:
                found.extend((raw, f"plugin {ep.name}: {where}")
                             for raw, where in _from_yaml_dir(Path(target)))
        except Exception as exc:  # a broken plugin must not break rightsize
            warnings.warn(f"recipe plugin {ep.name!r} failed to load: {exc}", stacklevel=2)
    return found


@functools.lru_cache(maxsize=1)
def all_recipes() -> dict[str, Recipe]:
    """Bundled recipes, then plugins'. A plugin may add recipes but not replace one: an id
    that is already taken is an error, so an installed package cannot silently change what
    a bundled recipe renders."""
    out: dict[str, Recipe] = {}
    for raw, where in _from_yaml_dir(data_dir() / "recipes") + _plugins():
        recipe = Recipe.model_validate(raw)
        validate_template(recipe.template)
        if recipe.id in out:
            raise ValueError(f"duplicate recipe id {recipe.id} from {where}")
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
