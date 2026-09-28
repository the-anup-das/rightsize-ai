"""Diffusion pipeline fit estimator: image and video (F3).

formula_id ``diffusion.components.v0``. A pipeline runs in phases, and its peak is the
worst phase: the weights resident during it plus that phase's activations.

  encode    text encoders                + text_encode_gb
  denoise   denoiser(s)                  + activations, per sample (CFG runs two):
              transformer  tokens x hidden x 2 B x dit_activation_factor
              UNet         latent pixels x unet_bytes_per_latent_pixel
  decode    VAE                          + output pixels x vae_decode_bytes_per_pixel
                                           x images decoded at once

Which weights are resident depends on the offload strategy:

  none        all of them, all the time             pipe.to("cuda")
  model       only the component that is running   pipe.enable_model_cpu_offload()
  sequential  about one layer of it                pipe.enable_sequential_cpu_offload()
              (group offloading gets near this memory at a fraction of the slowdown)

bitsandbytes and torchao quantize on the GPU as the pipeline loads, so every component in
one of their formats is on the GPU at once before any offloading starts: a load phase.
The diffusers blog measured it: FLUX.1-dev in nf4 with model offload peaked at 12.4 GiB,
all of it at load time.

vram = worst phase x (1 + allocator slack) + CUDA context. Offloaded weights live in
system RAM. The constants, and the published measurements they reproduce, are in
data/runtimes/diffusers/memory.yaml; tests/test_families.py replays every measurement.
There is no speed model yet: denoising is compute-bound, and the hardware table has
bandwidth, not FLOPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rightsize._data import load_yaml
from rightsize.fit.formats import lookup
from rightsize.types import Device, FitResult, ModelFacts, Verdict

FORMULA_ID = "diffusion.components.v0"
OFFLOAD = ("none", "model", "sequential")
#: Formats produced by quantizing on the GPU while loading (bitsandbytes, torchao).
_QUANTIZED_ON_LOAD = frozenset({"nf4", "nf4-dq", "int8", "int4", "fp8"})
_HEADROOM = 0.10
_OFFLOAD_CALL = {
    "model": "pipe.enable_model_cpu_offload()",
    "sequential": "pipe.enable_sequential_cpu_offload()",
}


@dataclass(frozen=True)
class Part:
    name: str
    role: str  # denoiser, text_encoder, vae, other
    params: int
    fmt: str
    bpw: float
    config: dict[str, Any]

    @property
    def gb(self) -> float:
        return self.params * self.bpw / 8 / 1e9


def constants() -> dict[str, float]:
    return load_yaml("runtimes/diffusers/memory.yaml")["constants"]


def parts(
    facts: ModelFacts,
    quant: str | dict[str, str] | None,
    text_encoder_quant: str | None,
    compute_dtype: str = "bf16",
) -> list[Part]:
    """Every component at the format it will run in.

    A string quant applies to the denoiser; text encoders take ``text_encoder_quant``;
    everything else stays at the compute dtype. A dict names the format per component,
    which is how diffusers' PipelineQuantizationConfig is written."""
    pipe = (facts.extra or {}).get("pipeline") or {}
    comps = pipe.get("components") or {}
    if not comps:  # a single-model repo (a transformer or UNet on its own)
        comps = {"model": {"role": "denoiser", "params": facts.params_total or 0, "config": {}}}
    out = []
    for name, c in comps.items():
        role = c.get("role", "other")
        if isinstance(quant, dict):
            fmt = quant.get(name, compute_dtype)
        elif role == "denoiser":
            fmt = quant or compute_dtype
        elif role == "text_encoder":
            fmt = text_encoder_quant or compute_dtype
        else:
            fmt = compute_dtype
        f = lookup(fmt, gguf="tensor")
        out.append(Part(name, role, int(c.get("params") or 0), f.id, f.bpw, c.get("config") or {}))
    return out


def _vae_factors(vae: dict[str, Any], video: bool) -> tuple[int, int, int]:
    """(spatial factor, temporal factor, latent channels) from the VAE's config."""
    spatial = vae.get("spatial_compression_ratio") or vae.get("scale_factor_spatial")
    if not spatial and vae.get("block_out_channels"):
        spatial = 2 ** (len(vae["block_out_channels"]) - 1)
    temporal = vae.get("temporal_compression_ratio") or vae.get("scale_factor_temporal")
    latent = vae.get("latent_channels") or vae.get("z_dim") or 16
    return int(spatial or 8), int(temporal or (4 if video else 1)), int(latent)


def _uses_cfg(facts: ModelFacts, denoiser: Part | None) -> bool:
    """Classifier-free guidance runs every step twice. FLUX is guidance-distilled: one run."""
    pipe_class = str(((facts.extra or {}).get("pipeline") or {}).get("class") or "")
    if pipe_class.startswith("Flux") or "flux" in facts.ref.repo.lower():
        return False
    return not (denoiser and denoiser.config.get("guidance_embeds"))


def activations(
    facts: ModelFacts,
    ps: list[Part],
    resolution: tuple[int, int],
    batch: int,
    frames: int | None,
    vae_slicing: bool,
) -> tuple[dict[str, float], list[str]]:
    """GB of activations per phase, and the assumptions behind them."""
    k = constants()
    width, height = resolution
    video = bool(frames and frames > 1)
    notes: list[str] = []
    vae = next((p for p in ps if p.role == "vae"), None)
    den = next((p for p in ps if p.role == "denoiser"), None)
    spatial, temporal, latent_ch = _vae_factors(vae.config if vae else {}, video)
    lat_h, lat_w = math.ceil(height / spatial), math.ceil(width / spatial)
    lat_frames = 1 + (frames - 1) // temporal if video else 1
    samples = batch * (2 if _uses_cfg(facts, den) else 1)

    cfg = den.config if den else {}
    cls = str(cfg.get("_class_name") or "")
    if "UNet" in cls or (den and den.name.startswith("unet")):
        width0 = (cfg.get("block_out_channels") or [320])[0]
        denoise = samples * lat_h * lat_w * k["unet_bytes_per_latent_pixel"] * width0 / 320 / 1e9
    else:
        patch = cfg.get("patch_size") or 2
        patch = int(patch[-1] if isinstance(patch, list) else patch)
        if cfg.get("in_channels") and cfg["in_channels"] >= 4 * latent_ch:
            patch = max(patch, 2)  # FLUX packs 2x2 latent patches before the transformer
        tokens = (lat_h // patch) * (lat_w // patch) * lat_frames + k["default_text_tokens"]
        hidden = (cfg.get("num_attention_heads") or 0) * (cfg.get("attention_head_dim") or 0)
        if not hidden:
            hidden = 3072
            notes.append("transformer width unknown (config not readable); assumed 3072")
        denoise = samples * tokens * hidden * 2 * k["dit_activation_factor"] / 1e9
    if samples > batch:
        notes.append("classifier-free guidance: each step runs twice")

    images = 1 if vae_slicing else batch
    frames_at_once = min(frames, 4) if video else 1
    decode = images * width * height * frames_at_once * k["vae_decode_bytes_per_pixel"] / 1e9
    if video:
        notes.append("video: the VAE is assumed to decode four frames at a time")
    return {"encode": k["text_encode_gb"], "denoise": denoise, "decode": decode}, notes


def _resident(ps: list[Part], role_set: set[str], offload: str) -> float:
    """Weights on the GPU during one phase, in GB."""
    k = constants()
    if offload == "none":
        return sum(p.gb for p in ps)
    running = [p for p in ps if p.role in role_set]
    if not running:
        return 0.0
    if offload == "model":
        return max(p.gb for p in running)  # one component at a time
    return min(k["sequential_resident_gb"], max(p.gb for p in running))


def peak(
    ps: list[Part],
    acts: dict[str, float],
    offload: str,
) -> tuple[float, str, dict[str, float]]:
    """(worst phase in GB before overheads, its name, every phase)."""
    phases = {
        "load": sum(p.gb for p in ps if p.fmt in _QUANTIZED_ON_LOAD),
        "encode": _resident(ps, {"text_encoder", "other"}, offload) + acts["encode"],
        "denoise": _resident(ps, {"denoiser"}, offload) + acts["denoise"],
        "decode": _resident(ps, {"vae"}, offload) + acts["decode"],
    }
    worst = max(phases, key=phases.get)
    return phases[worst], worst, phases


def _vram(worst: float) -> tuple[float, float]:
    k = constants()
    overhead = worst * k["allocator_slack"] + k["cuda_context_gb"]
    return worst + overhead, overhead


def estimate(
    facts: ModelFacts,
    quant: str | dict[str, str] | None,
    device: Device,
    *,
    offload: str = "none",
    resolution: tuple[int, int] = (1024, 1024),
    batch: int = 1,
    frames: int | None = None,
    text_encoder_quant: str | None = None,
    vae_slicing: bool = False,
) -> FitResult:
    if offload not in OFFLOAD:
        raise ValueError(f"offload must be one of {', '.join(OFFLOAD)}, not {offload!r}")
    ps = parts(facts, quant, text_encoder_quant)
    acts, notes = activations(facts, ps, resolution, batch, frames, vae_slicing)
    worst, phase, phases = peak(ps, acts, offload)
    vram, overhead = _vram(worst)
    usable = device.memory_gb * device.usable_fraction
    total_weights = sum(p.gb for p in ps)
    ram = total_weights if offload != "none" else 0.0

    notes[:0] = [
        "components: " + ", ".join(f"{p.name} {p.fmt} {p.gb:.2f} GB" for p in ps),
        f"peak in the {phase} phase; {resolution[0]}x{resolution[1]}, batch {batch}"
        + (f", {frames} frames" if frames else ""),
    ]
    if offload != "none":
        notes.append(
            f"offload={offload}: {_OFFLOAD_CALL[offload]}; the offloaded "
            f"weights need {ram:.1f} GB of system RAM"
        )
    if device.unified_memory and offload != "none":
        notes.append(
            "unified memory: offloading moves weights to the same memory pool, "
            "so it saves nothing here"
        )
        vram = max(vram, total_weights + max(acts.values()) + overhead)

    if vram <= usable * (1 - _HEADROOM):
        verdict = Verdict.fits
    elif vram <= usable:
        verdict = Verdict.tight
    else:
        verdict = Verdict.no_fit
        if not device.unified_memory:
            for alt in OFFLOAD[OFFLOAD.index(offload) + 1 :]:
                alt_vram, _ = _vram(peak(ps, acts, alt)[0])
                if alt_vram <= usable:
                    verdict = Verdict.offload
                    notes.append(
                        f"fits with {_OFFLOAD_CALL[alt]} ({alt_vram:.1f} GB on the "
                        f"GPU, {total_weights:.1f} GB of system RAM)"
                    )
                    break
    if batch > 1 and not vae_slicing and phase == "decode":
        notes.append("pipe.vae.enable_slicing() decodes one image at a time and lowers the peak")
    notes.append("no speed estimate for diffusion yet: denoising is compute-bound")
    if offload != "none" and any(p.fmt == "int8" for p in ps):
        notes.append(
            "bitsandbytes int8 weights may not leave the GPU: the diffusers blog "
            "measured FLUX.1-dev in bnb int8 with model offload at 23.4 GiB, about "
            "the same as without"
        )
    if any(p.fmt in _QUANTIZED_ON_LOAD for p in ps):
        notes.append(
            "bitsandbytes and torchao quantize on the GPU while loading; their "
            "published rows sit 1-3.3 GiB above this in reserved memory"
        )

    confidence = 0.3 if frames and frames > 1 else 0.5
    return FitResult(
        verdict=verdict,
        vram_gb=round(vram, 2),
        ram_gb=round(ram, 2),
        breakdown={
            "weights": round(total_weights, 3),
            **{f"weights.{p.name}": round(p.gb, 3) for p in ps},
            **{f"phase.{name}": round(v, 3) for name, v in phases.items()},
            "activations": round(acts.get(phase, 0.0), 3),
            "overhead": round(overhead, 3),
            "usable_memory": round(usable, 2),
        },
        speed=None,
        speed_unit=None,
        confidence=min(confidence, facts.confidence),
        formula_id=FORMULA_ID,
        notes=notes,
    )
