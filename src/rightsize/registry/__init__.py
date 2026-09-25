"""Framework registry and renderable recipes (F5).

Plan and TODO: docs/plans/F05-framework-registry.md
"""

from __future__ import annotations

from rightsize.registry.loader import (
    all_recipes,
    clear_cache,
    framework,
    framework_infos,
    frameworks,
    get,
    trainer_for,
)
from rightsize.registry.render import TemplateError, render
from rightsize.registry.schema import FrameworkInfo, Recipe, RecipeInput, RenderedStep

__all__ = [
    "FrameworkInfo",
    "Recipe",
    "RecipeInput",
    "RenderedStep",
    "TemplateError",
    "all_recipes",
    "clear_cache",
    "framework",
    "framework_infos",
    "frameworks",
    "get",
    "render",
    "trainer_for",
]
