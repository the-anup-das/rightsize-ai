"""Recipe and framework schemas (F5). A recipe renders a command or config; it never imports
the toolkit. A framework descriptor says what the toolkit is for, where it runs, how it is
installed and how its steps join a plan, so adding one is data, not code."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal["convert", "finetune", "quantize", "export", "serve", "calibrate", "evaluate"]
TrainMode = Literal["lora", "qlora", "full"]


class RecipeInput(BaseModel):
    type: Literal["path", "str", "int", "float", "bool", "enum"] = "str"
    required: bool = False
    default: Any = None
    values: list[str] | None = None
    help: str | None = None


class Recipe(BaseModel):
    id: str
    framework: str
    stage: Stage
    families: list[str] = Field(default_factory=list)
    hardware: dict[str, Any] = Field(default_factory=dict)
    install_extra: str | None = None
    install_line: str | None = None
    inputs: dict[str, RecipeInput] = Field(default_factory=dict)
    kind: Literal["command", "config"] = "command"
    language: Literal["bash", "python", "yaml", "modelfile"] | None = Field(
        default=None,
        description="what a config renders to; tests parse python and yaml renders",
    )
    template: str
    notes: list[str] = Field(default_factory=list)
    source_doc_url: str
    version_tested: str | None = None
    verified: Literal["run", "help", "docs"] = Field(
        default="docs",
        description=(
            "How far this recipe has been checked against version_tested: 'run' means it was "
            "run end to end, 'help' that every flag was checked against the tool's own --help, "
            "'docs' that it was transcribed from the documentation only"
        ),
    )


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
    verified: Literal["run", "help", "docs"] = "docs"


class InstallSpec(BaseModel):
    """How a framework is installed, and how to tell that it is."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pip", "binary", "app"]
    packages: list[str] = Field(
        default_factory=list, description="pip requirements, pinned at version_tested"
    )
    needs_torch: bool = Field(default=False, description="needs a PyTorch build for the GPU")
    check: str | None = Field(
        default=None, description="a Python module to import, or a program to find on PATH"
    )
    line: str = Field(description="the install line a person would type")
    note: str | None = None


class QuantShape(BaseModel):
    """The QuantSpec a step records: bnb nf4 for QLoRA, MLX 4-bit, ..."""

    model_config = ConfigDict(extra="forbid")

    method: str
    variant: str | None = None
    bits_per_weight: float | None = None


class BeforeStep(BaseModel):
    """A step a trainer needs first, such as MLX quantizing the model it will train QLoRA on."""

    model_config = ConfigDict(extra="forbid")

    recipe: str
    stage: Stage
    quant: QuantShape | None = None
    model_from: str | None = Field(
        default=None, description="the plan value the training step reads as its model after this"
    )


class FinetuneRole(BaseModel):
    """What a framework does as a trainer, and on which hardware it is the default one."""

    model_config = ConfigDict(extra="forbid")

    modes: list[TrainMode] = Field(min_length=1, description="modes it has a recipe for")
    recipe: str
    qlora_quant: QuantShape | None = Field(
        default=None, description="how QLoRA stores the frozen base weights"
    )
    before: dict[TrainMode, BeforeStep] = Field(default_factory=dict)
    writes: Literal["merged", "adapter"] = Field(
        description="a merged 16-bit model, or an adapter the recipe merges into merged_dir"
    )
    default_for: list[str] = Field(
        default_factory=list,
        description="device vendors it is the default trainer on; '*' for a vendor no "
        "framework names",
    )
    priority: int = Field(default=0, description="breaks a tie between two defaults")


class FrameworkInfo(BaseModel):
    """One toolkit: what it is for, where it runs, how it installs, how its steps join a plan.

    Lives beside the toolkit's recipes as data/recipes/<name>/framework.yaml, or in a plugin
    package's recipe folder, so a new SDK needs no change to rightsize itself."""

    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    summary: str
    stages: list[Stage] = Field(min_length=1)
    families: list[str] = Field(default_factory=list)
    formats: list[str] = Field(
        default_factory=list, description="formats it writes: gguf, mlx, fp8, openvino, ..."
    )
    hardware: dict[str, Any] = Field(
        default_factory=dict, description="vendors, os, min_compute_capability, note"
    )
    install: InstallSpec
    finetune: FinetuneRole | None = None
    defaults: dict[str, str] = Field(
        default_factory=dict,
        description="values this framework's recipes read when a plan does not set them; "
        "{slug} is the model's repo id with / as __, {quant} the plan's quantization",
    )
    homepage: str
    source_doc_url: str
    version_tested: str | None = None
