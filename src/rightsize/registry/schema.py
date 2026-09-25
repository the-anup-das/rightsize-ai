"""Recipe schema (F5). A recipe renders a command or config; it never imports the toolkit."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RecipeInput(BaseModel):
    type: Literal["path", "str", "int", "float", "bool", "enum"] = "str"
    required: bool = False
    default: Any = None
    values: list[str] | None = None
    help: str | None = None


class Recipe(BaseModel):
    id: str
    framework: str
    stage: Literal["convert", "finetune", "quantize", "export", "serve", "calibrate", "evaluate"]
    families: list[str] = Field(default_factory=list)
    hardware: dict[str, Any] = Field(default_factory=dict)
    install_extra: str | None = None
    install_line: str | None = None
    inputs: dict[str, RecipeInput] = Field(default_factory=dict)
    kind: Literal["command", "config"] = "command"
    template: str
    notes: list[str] = Field(default_factory=list)
    source_doc_url: str
    version_tested: str | None = None


class RenderedStep(BaseModel):
    recipe_id: str
    framework: str
    stage: str
    kind: Literal["command", "config"]
    text: str
    argv: list[str] | None = Field(default=None, description="command split into arguments")
    install_line: str | None = None
    notes: list[str] = Field(default_factory=list)
    source_doc_url: str
