"""Core types shared by the SDK, CLI, MCP server and (later) the web platform.

One ``Plan`` model everywhere; its JSON Schema (``Plan.model_json_schema()``)
is the contract the platform generates TypeScript types from.

These shapes follow docs/plans/F03-fit-engine.md and F04-rules-engine.md and
will grow as those features land. Fields marked "F#" are placeholders owned by
that feature.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Family(StrEnum):
    """Model family. Each has its own estimator (F3)."""

    llm = "llm"  # includes VLMs
    diffusion = "diffusion"  # image and video
    audio = "audio"  # STT and TTS
    vision = "vision"  # detection, segmentation, classification
    embedding = "embedding"


class Mode(StrEnum):
    """What the hardware is asked to do."""

    infer = "infer"
    lora = "lora"
    qlora = "qlora"
    full = "full"


class Verdict(StrEnum):
    fits = "fits"
    tight = "tight"  # fits inside headroom but with little margin
    offload = "offload"  # fits only with CPU/system-RAM offload
    no_fit = "no_fit"


class Provenance(BaseModel):
    """Where a number came from. Required on every data record (lightweight rule 2)."""

    source_url: str
    fetched_at: str = Field(description="ISO-8601 date the value was taken from the source")
    note: str | None = None


class Device(BaseModel):
    """A fine-tuning box or a target device (F2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    vendor: Literal["nvidia", "amd", "intel", "apple", "qualcomm", "cpu", "other"] = "other"
    memory_gb: float = Field(gt=0, description="VRAM, or unified memory on Apple")
    system_ram_gb: float | None = None
    bandwidth_gbps: float | None = Field(default=None, description="Memory bandwidth, GB/s")
    compute_arch: str | None = Field(
        default=None, description="e.g. ada, hopper, blackwell, rdna3, m4, sm_89"
    )
    backends: list[str] = Field(default_factory=list, description="cuda, rocm, metal, vulkan, ...")
    os: Literal["linux", "windows", "macos", "ios", "android", "unknown"] = "unknown"
    usable_fraction: float = Field(default=1.0, gt=0, le=1.0)
    provenance: Provenance | None = None


class ModelRef(BaseModel):
    """A pointer to a model repo or file (F1)."""

    repo: str = Field(description="Hub id, e.g. Qwen/Qwen3-14B")
    revision: str | None = None
    file: str | None = Field(default=None, description="For GGUF/MLX: the specific file")


class ModelFacts(BaseModel):
    """What the catalog knows about a model without downloading it (F1)."""

    ref: ModelRef
    family: Family
    params_total: int | None = None
    params_active: int | None = Field(default=None, description="MoE active parameters")
    dtype: str | None = None
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    context_max: int | None = None
    license: str | None = None
    base_model: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class QuantSpec(BaseModel):
    """A quantization the fit engine and recipes agree on (F3, F5)."""

    method: str = Field(description="gguf, bnb, awq, gptq, fp8, mlx, torchao, ...")
    variant: str | None = Field(default=None, description="e.g. Q4_K_M, nf4, int8")
    bits_per_weight: float | None = Field(default=None, description="Real bpw, not nominal")


class RuntimeSpec(BaseModel):
    name: str = Field(description="llama.cpp, ollama, vllm, mlx_lm, diffusers, whisper.cpp, ...")
    version: str | None = None


class FitResult(BaseModel):
    """Output of one estimate (F3). Always explainable."""

    verdict: Verdict
    vram_gb: float
    ram_gb: float = 0.0
    breakdown: dict[str, float] = Field(
        default_factory=dict, description="weights, kv_cache, activations, overhead, ..."
    )
    speed: float | None = Field(default=None, description="tok/s for LLMs, s/image for diffusion")
    speed_unit: str | None = None
    confidence: float = Field(ge=0, le=1)
    formula_id: str = Field(description="Which estimator/formula produced this")
    notes: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    """One stage of a plan: finetune, quantize, export or serve (F4, F5)."""

    stage: Literal["finetune", "quantize", "export", "serve"]
    framework: str
    device: Device
    quant: QuantSpec | None = None
    runtime: RuntimeSpec | None = None
    fit: FitResult
    recipe_id: str | None = Field(default=None, description="Renderable recipe (F5)")


class Plan(BaseModel):
    """A ranked, explained answer (F4). JSON-serialisable and shareable."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    model: ModelFacts
    mode: Mode
    steps: list[PlanStep]
    score: float = Field(description="Ranking score; higher is better")
    quality_penalty: float | None = Field(
        default=None, description="Estimated quality loss from quantization, 0..1"
    )
    trace: list[str] = Field(
        default_factory=list, description="Rules that fired, each with its source URL"
    )
    cloud_fallback: dict[str, Any] | None = Field(default=None, description="F7 rent-this offer")

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def schema_json(cls) -> str:
        return json.dumps(cls.model_json_schema(), indent=2)


class Measurement(BaseModel):
    """One predicted-vs-measured pair recorded by an execution step (F8, F9)."""

    kind: Literal[
        "file_size_gb", "peak_vram_gb", "tok_per_s", "kld_mean", "top1_agreement", "ppl", "wall_s"
    ]
    value: float
    predicted: float | None = None
    note: str | None = None


class RunStep(BaseModel):
    recipe_id: str
    argv: list[str]
    started: str
    finished: str | None = None
    returncode: int | None = None
    log_path: str | None = None
    skipped: bool = False
    measurements: list[Measurement] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Everything about one execution: what was predicted, what ran, what was measured (F8)."""

    id: str
    created: str
    rightsize_version: str
    model: ModelRef
    facts: ModelFacts | None = None
    device: Device | None = None
    toolchain: dict[str, str] = Field(default_factory=dict)
    predicted: dict[str, FitResult] = Field(default_factory=dict, description="per quant")
    steps: list[RunStep] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict, description="name -> path")
    gate: dict[str, Any] = Field(
        default_factory=dict, description="per quant: pass|warn|fail + numbers"
    )
    status: Literal["running", "succeeded", "failed"] = "running"
