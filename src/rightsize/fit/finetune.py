"""Fine-tuning memory for LoRA, QLoRA and full fine-tuning (F3).

formula_id ``llm.finetune.components.v1``, every term in ``breakdown``:

  weights      base model at the mode's precision. QLoRA keeps linear layers in NF4 (4.5 bits:
               4 plus an fp32 scale per 64 weights) and embeddings in 16-bit, which bitsandbytes
               does not quantize; LoRA and full keep everything in 16-bit. Counted as
               transformers loads the model, without a tied head the checkpoint stores twice.
  trainable    LoRA adapters on every linear layer (rank 16 by default) at 6 bytes a parameter -
               weight, gradient and 8-bit AdamW's two states; or, for full fine-tuning, every
               parameter at 16 bytes - bf16 weight and gradient, fp32 master copy and AdamW's
               two fp32 states.
  activations  with gradient checkpointing, one hidden-state tensor per layer boundary.
  logits       0.7 x batch x sequence x vocabulary x 2 bytes: what the loss holds at once,
               measured on Unsloth's chunked loss (v0 had no term for it, and a 152k-vocabulary
               0.6B model allocated twice the tensors it predicted).
  overhead     1 GB: CUDA context, cuBLAS workspace, allocator slack. PyTorch's caching
               allocator reserves more when the card has room - 1.6 GB beyond what those
               Qwen3 runs allocated - and gives it back when memory runs short.

For LoRA and QLoRA the result never goes below Unsloth's published minimum, interpolated
between the rows either side (data/finetune/unsloth_vram.yaml). Those are measurements of a
real trainer; the components are an explanation. Confidence is lower than for inference
because batch size, sequence length and the trainer's own tricks (Unsloth offloads
checkpoints to system RAM and chunks the loss) move the real figure more than any single
constant here.
"""

from __future__ import annotations

from functools import lru_cache

from rightsize._data import load_yaml
from rightsize.types import GB, Device, FitResult, Mode, ModelFacts, Verdict

FORMULA_ID = "llm.finetune.components.v1"
CONFIDENCE = 0.45
_QLORA_BPW = 4.5
_ADAPTER_BYTES = 6  # bf16 weight + bf16 grad + 8-bit AdamW m and v
_FULL_BYTES = 16  # bf16 weight + bf16 grad + fp32 master + fp32 m + fp32 v
_OVERHEAD_GB = 1.0
# The loss needs the logits: batch x sequence x vocabulary. Unsloth computes it in chunks, so
# less than a whole bf16 batch is live at once. Its QLoRA runs on Qwen3-0.6B and 1.7B (batch 2,
# 2048 tokens, 152k vocabulary) allocated 0.94 and 0.77 GB beyond the other parts, 0.76 and
# 0.62 of that batch (2026-09-28, RTX 4070 Ti SUPER)
_LOGITS_SHARE = 0.7
_HEADROOM = 0.10


@lru_cache(maxsize=1)
def _floors() -> list[dict]:
    return sorted(load_yaml("finetune/unsloth_vram.yaml")["rows"], key=lambda r: r["params_b"])


def unsloth_floor(params: int, mode: Mode) -> tuple[float | None, str | None]:
    """(minimum GB, how it was read) from Unsloth's published table.

    Interpolated linearly between the two rows either side. Rounding up to the next row was
    tried first and made a step function out of sparse data: a 4B model took the 7B row's
    19 GB for 16-bit LoRA and came back no_fit on a 16 GB card, where the neighbouring rows
    put it near 11. Outside the table there is no floor, and the components stand alone."""
    if mode not in (Mode.lora, Mode.qlora):
        return None, None
    key = "qlora_gb" if mode is Mode.qlora else "lora_gb"
    size_b = params / 1e9
    rows = _floors()
    for lo, hi in zip(rows, rows[1:], strict=False):
        if lo["params_b"] <= size_b <= hi["params_b"]:
            span = hi["params_b"] - lo["params_b"]
            t = (size_b - lo["params_b"]) / span if span else 0.0
            gb = lo[key] + t * (hi[key] - lo[key])
            return round(gb, 2), f"between its {lo['params_b']:g}B and {hi['params_b']:g}B rows"
    return None, None


def merged(facts: ModelFacts) -> ModelFacts:
    """The model a fine-tune writes, as the steps after it see it. Trainers save through
    transformers or MLX, which store a tied output head once, so a second copy the base
    checkpoint kept (F1's tied_head_stored) is gone and llama.cpp's converter writes no
    output.weight: Unsloth's merge of Qwen3-0.6B has 596M parameters, not 751.6M, and its
    Q8_0 came out at 0.639 GB against the 0.799 GB sized from the base checkpoint."""
    stored = int((facts.extra or {}).get("tied_head_stored") or 0)
    if not stored or not facts.params_total:
        return facts
    return facts.model_copy(
        update={
            "params_total": facts.params_total - stored,
            "params_active": facts.params_active - stored if facts.params_active else None,
            "extra": {**facts.extra, "tied_head_stored": None},
        }
    )


def embedding_params(facts: ModelFacts) -> int:
    """Input embeddings plus the output head when it is not tied to them."""
    extra = facts.extra or {}
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if not (vocab and hidden):
        return 0
    return int(vocab * hidden * (1 if extra.get("tie_word_embeddings") else 2))


def lora_params(facts: ModelFacts, rank: int = 16) -> int:
    """Adapter parameters for rank-r LoRA on all seven linear layers of each block.

    A rank-r adapter on a d_in x d_out layer adds r * (d_in + d_out) parameters. MoE expert
    layers are counted once, as if dense: adapters on every expert would be larger, and most
    MoE recipes do not put them there."""
    extra = facts.extra or {}
    h = extra.get("hidden_size")
    layers = facts.num_layers
    if not (h and layers):
        return 0
    heads = extra.get("num_attention_heads") or 0
    hd = facts.head_dim or (h // heads if heads else 0)
    q = (heads * hd) or h
    kv = (facts.num_kv_heads or heads) * hd or h
    inter = extra.get("intermediate_size") or int(h * 3.5)
    per_layer = (
        (h + q)  # q_proj
        + 2 * (h + kv)  # k_proj, v_proj
        + (q + h)  # o_proj
        + 2 * (h + inter)  # gate_proj, up_proj
        + (inter + h)  # down_proj
    )
    return int(rank * per_layer * layers)


def estimate_finetune(
    facts: ModelFacts,
    device: Device,
    mode: Mode | str = Mode.qlora,
    *,
    seq_len: int = 2048,
    batch: int = 1,
    lora_rank: int = 16,
) -> FitResult:
    mode = Mode(mode)
    if mode is Mode.infer:
        raise ValueError("estimate_finetune is for lora, qlora and full; use estimate() to infer")
    if not facts.params_total:
        raise ValueError("params_total is required")
    facts = merged(facts)  # transformers ties a stored output head away when it loads the model
    params = facts.params_total
    embed = min(embedding_params(facts), params)
    notes: list[str] = []

    if mode is Mode.qlora:
        weights = ((params - embed) * _QLORA_BPW / 8 + embed * 2) / GB
        notes.append("QLoRA: linear layers in NF4, embeddings kept in 16-bit")
    else:
        weights = params * 2 / GB

    if mode is Mode.full:
        trainable = params * (_FULL_BYTES - 2) / GB  # the 2-byte weights are counted above
        notes.append("full fine-tuning: AdamW with fp32 master weights, 16 bytes a parameter")
    else:
        adapters = lora_params(facts, lora_rank)
        trainable = adapters * _ADAPTER_BYTES / GB
        notes.append(f"rank-{lora_rank} LoRA on every linear layer: {adapters / 1e6:.0f}M params")
    if (facts.extra or {}).get("num_experts"):
        notes.append("MoE: every expert stays resident; adapters counted as if dense")

    hidden = (facts.extra or {}).get("hidden_size") or 0
    activations = batch * seq_len * hidden * (facts.num_layers or 0) * 2 / GB
    vocab = (facts.extra or {}).get("vocab_size") or 0
    logits = batch * seq_len * vocab * 2 * _LOGITS_SHARE / GB
    parts = weights + trainable + activations + logits + _OVERHEAD_GB

    floor, where = unsloth_floor(params, mode)
    vram = parts
    if floor is not None:
        notes.append(f"Unsloth's published minimum ({mode.value}) is {floor:g} GB, {where}")
        if floor > parts:
            vram = floor
            notes.append("the published minimum is above the component estimate; using it")

    usable = device.memory_gb * device.usable_fraction
    if vram <= usable * (1 - _HEADROOM):
        verdict = Verdict.fits
    elif vram <= usable:
        verdict = Verdict.tight
    else:
        verdict = Verdict.no_fit
        if mode is not Mode.qlora:
            notes.append("does not fit; QLoRA needs roughly a quarter of the weights' memory")

    notes.append(f"batch {batch}, sequence {seq_len}, gradient checkpointing")
    breakdown = {
        "weights": round(weights, 3),
        "trainable_state": round(trainable, 3),
        "activations": round(activations, 3),
        "logits": round(logits, 3),
        "overhead": _OVERHEAD_GB,
        "usable_memory": round(usable, 2),
    }
    if floor is not None:
        breakdown["published_minimum"] = floor
    return FitResult(
        verdict=verdict,
        vram_gb=round(vram, 2),
        breakdown=breakdown,
        confidence=CONFIDENCE,
        formula_id=FORMULA_ID,
        notes=notes,
    )
