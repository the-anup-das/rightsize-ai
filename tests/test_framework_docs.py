"""Framework pages are generated from the recipes and must not fall behind them (F5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rightsize.registry import all_recipes, loader
from rightsize.registry.docs import pages

ROOT = Path(__file__).resolve().parents[1]


def test_framework_pages_are_current() -> None:
    for name, text in pages().items():
        path = ROOT / "docs" / "frameworks" / name
        assert path.exists(), f"missing {name}: run python -m rightsize.registry.docs"
        assert path.read_text(encoding="utf-8") == text, (
            f"docs/frameworks/{name} is stale: run python -m rightsize.registry.docs"
        )


def test_every_recipe_appears_with_its_verification() -> None:
    text = "\n".join(pages().values())
    for recipe in all_recipes().values():
        assert f"`{recipe.id}`" in text
    assert "flags checked against the tool's --help" in text


@pytest.fixture
def fresh_registry():
    loader.all_recipes.cache_clear()
    yield
    loader.all_recipes.cache_clear()


def test_a_plugin_cannot_replace_a_bundled_recipe(monkeypatch, fresh_registry) -> None:
    """Plugins may add recipes. A clash would let an installed package change what a
    bundled recipe renders, so it is an error rather than a silent override."""
    bundled = next(iter(all_recipes().values())).model_dump()
    loader.all_recipes.cache_clear()
    monkeypatch.setattr(loader, "_plugins", lambda: [(bundled, "plugin evil")])
    with pytest.raises(ValueError, match="duplicate recipe id"):
        loader.all_recipes()


def test_a_plugin_can_add_a_recipe(monkeypatch, fresh_registry) -> None:
    extra = next(iter(all_recipes().values())).model_dump() | {"id": "plugin/example"}
    loader.all_recipes.cache_clear()
    monkeypatch.setattr(loader, "_plugins", lambda: [(extra, "plugin example")])
    assert "plugin/example" in loader.all_recipes()
