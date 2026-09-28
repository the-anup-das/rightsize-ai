"""Load recipes and framework descriptors from data/recipes/<framework>/ (F5).

Each framework is one folder: ``framework.yaml`` says what the toolkit is for, where it runs,
how it installs and how its steps join a plan; every other YAML file is a recipe. Plugins add
folders of the same shape, so a new SDK is data, not a change to rightsize.
"""

from __future__ import annotations

import functools
from pathlib import Path

from rightsize._data import data_dir
from rightsize.registry.render import validate_template
from rightsize.registry.schema import FrameworkInfo, InstallSpec, Recipe

#: Third-party packages add frameworks and recipes under this entry-point group. The entry
#: point loads either a directory laid out like data/recipes (a folder per framework, with a
#: framework.yaml and recipe files) or a callable returning dicts: a dict with a template
#: is a recipe, one without is a framework descriptor.
#:
#:     [project.entry-points."rightsize.recipes"]
#:     mytool = "mytool.rightsize:RECIPES_DIR"
ENTRY_POINT_GROUP = "rightsize.recipes"
DESCRIPTOR = "framework.yaml"


def _yaml_files(root: Path, descriptors: bool) -> list[tuple[dict, str]]:
    import yaml

    found = []
    for path in sorted(root.glob("*/*.yaml")) + sorted(root.glob("*.yaml")):
        if (path.name == DESCRIPTOR) is descriptors:
            with path.open("r", encoding="utf-8") as fh:
                found.append((yaml.safe_load(fh), str(path)))
    return found


def _from_yaml_dir(root: Path) -> list[tuple[dict, str]]:
    return _yaml_files(root, descriptors=False)


def _plugin_targets() -> list[tuple[str, object]]:
    """What each installed plugin's entry point loads. A plugin that fails to load is
    skipped with a warning rather than taking the bundled frameworks down with it."""
    import warnings
    from importlib.metadata import entry_points

    targets = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            target = ep.load()
            targets.append((ep.name, list(target()) if callable(target) else Path(target)))
        except Exception as exc:  # a broken plugin must not break rightsize
            warnings.warn(f"recipe plugin {ep.name!r} failed to load: {exc}", stacklevel=2)
    return targets


def _plugins() -> list[tuple[dict, str]]:
    """Recipes from installed packages."""
    found: list[tuple[dict, str]] = []
    for name, target in _plugin_targets():
        if isinstance(target, Path):
            found.extend((raw, f"plugin {name}: {where}") for raw, where in _from_yaml_dir(target))
        else:
            found.extend((dict(r), f"plugin {name}") for r in target if "template" in r)
    return found


def _plugin_frameworks() -> list[tuple[dict, str]]:
    """Framework descriptors from installed packages."""
    found: list[tuple[dict, str]] = []
    for name, target in _plugin_targets():
        if isinstance(target, Path):
            found.extend(
                (raw, f"plugin {name}: {where}")
                for raw, where in _yaml_files(target, descriptors=True)
            )
        else:
            found.extend((dict(r), f"plugin {name}") for r in target if "template" not in r)
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
    return sorted({r.framework for r in all_recipes().values()} | set(framework_infos()))


@functools.lru_cache(maxsize=1)
def framework_infos() -> dict[str, FrameworkInfo]:
    """Every framework's descriptor, bundled then plugins'. As with recipes, a plugin may add
    a framework but not replace one. A framework that has recipes and no descriptor (an
    older plugin) gets a minimal one made from its recipes, so it still lists."""
    out: dict[str, FrameworkInfo] = {}
    bundled = _yaml_files(data_dir() / "recipes", descriptors=True)
    for raw, where in bundled + _plugin_frameworks():
        info = FrameworkInfo.model_validate(raw)
        if info.name in out:
            raise ValueError(f"duplicate framework {info.name} from {where}")
        out[info.name] = info
    by_framework: dict[str, list[Recipe]] = {}
    for r in all_recipes().values():
        by_framework.setdefault(r.framework, []).append(r)
    for name, recipes in by_framework.items():
        if name not in out:
            first = recipes[0]
            out[name] = FrameworkInfo(
                name=name,
                title=name,
                summary=f"recipes for {name}",
                stages=sorted({r.stage for r in recipes}),
                families=sorted({f for r in recipes for f in r.families}),
                install=InstallSpec(kind="pip", line=first.install_line or first.source_doc_url),
                homepage=first.source_doc_url,
                source_doc_url=first.source_doc_url,
                version_tested=first.version_tested,
            )
    return out


def framework(name: str) -> FrameworkInfo:
    try:
        return framework_infos()[name]
    except KeyError as exc:
        raise KeyError(f"unknown framework {name!r}; known: {sorted(framework_infos())}") from exc


def trainer_for(vendor: str) -> FrameworkInfo | None:
    """The framework that fine-tunes on a device of this vendor when none is chosen: one
    whose descriptor names the vendor, else one that takes any vendor ("*"); the higher
    priority wins a tie. Adding a trainer, or changing the default on some hardware, is an
    edit to a framework.yaml."""
    trainers = [f for f in framework_infos().values() if f.finetune]
    for want in (vendor, "*"):
        named = [f for f in trainers if want in f.finetune.default_for]
        if named:
            return max(named, key=lambda f: (f.finetune.priority, f.name))
    return None


def clear_cache() -> None:
    """Forget loaded recipes and descriptors (after a data update, or in tests)."""
    all_recipes.cache_clear()
    framework_infos.cache_clear()


def recipes_dir() -> Path:
    return data_dir() / "recipes"
