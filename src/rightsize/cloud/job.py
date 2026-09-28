"""How long a fine-tune takes on a rented GPU, and what it costs (F7).

formula_id ``cloud.job.flops.v0``:
  FLOPs  = 6 x active parameters x tokens x epochs
  hours  = FLOPs / (dense tensor TFLOPS x efficiency) / 3600
  cost   = hours x the offer's price

Six FLOPs per parameter per token is the usual training count (Kaplan et al. 2020): two for
the forward pass, four for the backward. LoRA skips the frozen weights' gradients (two) but
gradient checkpointing, which every recipe here turns on, recomputes the forward (two), so
LoRA and QLoRA land on six as well. The efficiency is an assumption, not a measurement:
0.35 of datasheet throughput, where single-GPU fine-tunes usually sit. QLoRA's dequantizing
makes it slower than 16-bit LoRA, which this does not model. Confidence 0.3 until F9 has
measured runs to fit against.
"""

from __future__ import annotations

from rightsize._data import load_yaml
from rightsize.types import JobEstimate, Offer

FORMULA_ID = "cloud.job.flops.v0"
FLOPS_PER_PARAM_TOKEN = 6
EFFICIENCY = 0.35


def tensor_tflops(gpu: str) -> tuple[float, str] | None:
    """(dense tensor TFLOPS, datasheet URL) for a catalog accelerator name, if known."""
    for rec in load_yaml("cloud/gpu_tflops.yaml")["gpus"]:
        if gpu in rec["names"]:
            return float(rec["tflops"]), rec["source_url"]
    return None


def estimate_job(
    params: int,
    tokens: int,
    offer: Offer,
    *,
    epochs: int = 1,
    efficiency: float = EFFICIENCY,
) -> JobEstimate:
    """Time and cost to train ``params`` active parameters on ``tokens`` tokens per epoch."""
    known = tensor_tflops(offer.gpu)
    if known is None:
        return JobEstimate(
            tokens=tokens,
            epochs=epochs,
            efficiency=efficiency,
            confidence=0.0,
            formula_id=FORMULA_ID,
            notes=[f"no datasheet throughput for {offer.gpu}; price only"],
        )
    tflops, source = known
    per_gpu = tflops * 1e12 * efficiency
    seconds = FLOPS_PER_PARAM_TOKEN * params * tokens * epochs / (per_gpu * offer.gpu_count)
    hours = seconds / 3600
    return JobEstimate(
        tokens=tokens,
        epochs=epochs,
        hours=round(hours, 2),
        usd=round(hours * offer.usd_per_hour, 2),
        tflops=tflops,
        efficiency=efficiency,
        confidence=0.3,
        formula_id=FORMULA_ID,
        notes=[
            f"{offer.gpu}: {tflops:g} dense tensor TFLOPS ({source}) at {efficiency:.0%}",
            "6 FLOPs per parameter per token; data loading, evaluation and the first "
            "download are not counted",
        ],
    )
