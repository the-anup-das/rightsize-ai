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


class Gate(BaseModel):
    """How a target's output is checked: an evaluate-stage recipe, run in the environment
    that produced the output, since that is the one that can load it."""

    model_config = ConfigDict(extra="forbid")

    recipe: str = Field(description="an evaluate recipe, e.g. transformers/kld-eval")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="the recipe inputs that select this format's loader"
    )


class Target(BaseModel):
    """A format ``rightsize quantize --to NAME`` produces with this recipe."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="what --to takes: fp8, w4a16, nf4, openvino-int4, ...")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="the recipe inputs that select this format"
    )
    size_from: str | None = Field(
        default=None, description="the quants/formats.yaml entry that predicts the output size"
    )
    embedding_bits: float | None = Field(
        default=None,
        description="bits the input embedding keeps, e.g. 16 where only Linear layers are "
        "quantized; empty means the format's own",
    )
    head_bits: float | None = Field(
        default=None,
        description="bits an output head of its own keeps (a tied head is saved once, as the "
        "embedding); empty means the format's own",
    )
    weights: str | None = Field(
        default=None,
        description="glob, inside what the recipe writes, for the files the prediction "
        "covers; empty means every weight file, not the tokenizer and config beside them",
    )
    gate: Gate | None = Field(default=None, description="how to check the output; --eval runs it")
    method: str | None = Field(
        default=None,
        description="the family the rules key on (quant.method): fp8, awq, gptq, nvfp4, int8, "
        "bnb ...; empty means the size_from format",
    )
    serve_format: str | None = Field(
        default=None,
        description="what a server lists in serve.formats to load the output; empty means no "
        "plan serves it",
    )
    serve_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="inputs the serve step needs for this output, e.g. quantization: modelopt",
    )


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
    file: str | None = Field(
        default=None,
        description="for a config: the file it is written to in the run directory "
        "(default <framework>_<name>.<ext>)",
    )
    run: str | None = Field(
        default=None,
        description="for a config: the command that runs the written file, with {file} for "
        "its path; python configs default to 'python {file}', other configs are only written",
    )
    writes: list[str] = Field(
        default_factory=list,
        description="the inputs naming what this step produces (a file or a directory), "
        "checked and measured after a run",
    )
    targets: list[Target] = Field(
        default_factory=list,
        description="formats this recipe produces for rightsize quantize --to",
    )
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

    def file_name(self) -> str:
        """Where a config recipe is written in the run directory."""
        if self.file:
            return self.file
        ext = {"python": "py", "yaml": "yml", "bash": "sh"}.get(self.language or "", "txt")
        return f"{self.id.replace('/', '_').replace('.', '_').replace('-', '_')}.{ext}"

    def run_line(self) -> str | None:
        """The command that runs a config recipe, or None when it is only written."""
        if self.run:
            return self.run
        return "python {file}" if self.kind == "config" and self.language == "python" else None


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
        description="a merged 16-bit model, or an adapter that the merge recipe folds into "
        "merged_dir"
    )
    merge: str | None = Field(
        default=None,
        description="for a trainer that writes an adapter: the recipe that merges it into "
        "the 16-bit model the conversion reads next",
    )
    default_for: list[str] = Field(
        default_factory=list,
        description="device vendors it is the default trainer on; '*' for a vendor no "
        "framework names",
    )
    priority: int = Field(default=0, description="breaks a tie between two defaults")


class ServeRole(BaseModel):
    """What a framework does as a server: the steps that start it, the runtime name the fit
    engine and the rules know it by, and the weight formats it loads."""

    model_config = ConfigDict(extra="forbid")

    recipes: list[str] = Field(min_length=1, description="the serve steps, in order")
    runtime: str = Field(description="the name in runtimes/overheads.yaml and the rules")
    formats: list[str] = Field(min_length=1, description="gguf, mlx, compressed-tensors, ...")
    default: bool = Field(default=False, description="the server a plan uses unless one is pinned")


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
    serve: ServeRole | None = None
    defaults: dict[str, str] = Field(
        default_factory=dict,
        description="values this framework's recipes read when a plan does not set them; "
        "{slug} is the model's repo id with / as __, {name} the part after the slash in "
        "lower case, {quant} the plan's quantization",
    )
    homepage: str
    source_doc_url: str
    version_tested: str | None = None
