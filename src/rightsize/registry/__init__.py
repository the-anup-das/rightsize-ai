"""Framework registry and renderable recipes (F5).

Plan and TODO: docs/plans/F05-framework-registry.md
"""

from __future__ import annotations

from rightsize.registry.loader import all_recipes, frameworks, get
from rightsize.registry.render import TemplateError, render
from rightsize.registry.schema import Recipe, RecipeInput, RenderedStep

__all__ = [
    "Recipe",
    "RecipeInput",
    "RenderedStep",
    "TemplateError",
    "all_recipes",
    "frameworks",
    "get",
    "render",
]
