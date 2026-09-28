"""Where llama.cpp puts the input embedding, and what serving costs beyond the buffers (F3).

The measurements here are llama.cpp b11177 on an RTX 4070 Ti SUPER under Windows
(2026-09-28): its own log of buffer sizes on load, and the VRAM rise it caused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightsize.fit import predicted_file_gb
from rightsize.fit.finetune import merged
from rightsize.fit.llm import _routed_experts, _type_sizes, embedding_type, estimate
from rightsize.hardware import resolve
from rightsize.types import ModelFacts

FIX = Path(__file__).parent / "fixtures" / "facts"
MIB = 1024 * 1024 / 1e9


def _facts(name: str) -> ModelFacts:
    return ModelFacts.model_validate(json.loads((FIX / f"{name}.json").read_text(encoding="utf-8")))


def _embedding_bytes(file_type: str, vocab: int, hidden: int, *, tied: bool,
                     architecture: str | None = None) -> int:
    block, size = _type_sizes()[embedding_type(file_type, hidden, tied=tied,
                                               architecture=architecture)]
    return vocab * hidden * size // block


@pytest.mark.parametrize("name", sorted(p.stem for p in FIX.glob("*GGUF*.json")))
def test_the_rules_give_every_fixture_its_embedding_bytes(name: str) -> None:
    """Tied, a file type's own type, a hidden size the block does not divide (gemma-3-270m's
    640 against Q6_K's 256), MXFP4_MOE: each header's token_embd bytes, exactly."""
    f = _facts(name)
    g = f.extra["gguf"]
    assert _embedding_bytes(g["file_type"], f.extra["vocab_size"], f.extra["hidden_size"],
                            tied=g["output_bytes"] is None,
                            architecture=f.extra["model_type"]) == g["input_embedding_bytes"]


@pytest.mark.parametrize("file_type, vocab, hidden, tied, cpu_mb", [
    ("Q4_K_M", 151936, 2048, False, 175.03),  # Qwen3-1.7B, converted with its stored head
    ("Q5_K_M", 151936, 2048, False, 213.93),
    ("Q8_0", 151936, 2048, False, 330.61),
    ("Q8_0", 151936, 1024, True, 165.31),  # Unsloth's merge of Qwen3-0.6B
    ("Q4_0", 65536, 1024, True, 55.05),  # LFM2-350M
    ("Q2_K", 100352, 768, True, 63.22),  # granite-4.0-h-350m
])
def test_llama_cpp_keeps_that_much_on_the_cpu(file_type, vocab, hidden, tied, cpu_mb) -> None:
    """llama.cpp's CPU_Mapped model buffer on load held the embedding and nothing else."""
    assert _embedding_bytes(file_type, vocab, hidden, tied=tied) / 1e6 == pytest.approx(
        cpu_mb, abs=0.01)


def test_the_embedding_leaves_vram_only_when_the_head_is_its_own() -> None:
    dev = resolve("RTX 4070 Ti SUPER")
    own = _facts("Qwen__Qwen3-1.7B")  # tied in its config, but the checkpoint stores the head
    gpu = estimate(own, "Q4_K_M", dev, ctx=4096)
    kept = estimate(own, "Q4_K_M", dev, ctx=4096, runtime="vllm")
    assert gpu.breakdown["input_embedding_ram"] == pytest.approx(0.175, abs=0.001)
    assert gpu.ram_gb == 0.18
    assert kept.ram_gb == 0 and "input_embedding_ram" not in kept.breakdown
    assert kept.breakdown["weights"] - gpu.breakdown["weights"] == pytest.approx(0.175, abs=0.001)
    # tied and stored once: the output head is a copy of the embedding and stays on the GPU
    tied = _facts("Qwen__Qwen3-4B")
    a = estimate(tied, "Q4_K_M", dev, ctx=4096)
    b = estimate(tied, "Q4_K_M", dev, ctx=4096, runtime="vllm")
    assert a.breakdown["weights"] == b.breakdown["weights"] and a.ram_gb > 0 == b.ram_gb
    # one memory for both: nothing to move
    mac = estimate(own, "Q4_K_M", resolve("M4 Max"), ctx=4096)
    assert mac.ram_gb == 0


@pytest.mark.parametrize("quant, ctx, all_logits, measured_mib", [
    ("Q4_K_M", 4096, False, 1765),  # llama-server
    ("Q8_0", 4096, False, 2461),
    ("Q4_K_M", 512, True, 1797),  # llama-perplexity: four sequences, a micro-batch of logits
    ("Q8_0", 512, True, 2495),
])
def test_qwen3_1_7b_as_llama_cpp_used_it(quant, ctx, all_logits, measured_mib) -> None:
    """The whole estimate, from the Hub facts, against the VRAM rise: 0.75 GB + 2% said
    2.31 GB for the first case, and counted the embedding in VRAM."""
    r = estimate(_facts("Qwen__Qwen3-1.7B"), quant, resolve("RTX 4070 Ti SUPER"), ctx=ctx,
                 all_logits=all_logits)
    assert r.vram_gb == pytest.approx(measured_mib * MIB, rel=0.03)


def test_a_perplexity_pass_holds_four_contexts_and_a_batch_of_logits() -> None:
    f = _facts("Qwen__Qwen3-1.7B")
    dev = resolve("RTX 4070 Ti SUPER")
    serve = estimate(f, "Q8_0", dev, ctx=512)
    ppl = estimate(f, "Q8_0", dev, ctx=512, all_logits=True)
    assert ppl.breakdown["kv_cache"] == pytest.approx(4 * serve.breakdown["kv_cache"], rel=0.01)
    logits = 512 * 151936 * 4 / 1e9  # 297 MiB; llama-perplexity's compute buffer was 301
    extra = ppl.breakdown["overhead"] - serve.breakdown["overhead"]
    assert extra == pytest.approx(logits, abs=0.001)


def test_mxfp4_is_mxfp4_for_the_experts_only() -> None:
    """llama-quantize's MXFP4_MOE writes every tensor but the routed experts as Q8_0; params x
    4.25 bits said 11.11 GB for gpt-oss-20b, whose GGUF holds 12.10 GB of tensors."""
    f = _facts("openai__gpt-oss-20b")
    g = _facts("ggml-org__gpt-oss-20b-GGUF").extra["gguf"]
    assert _routed_experts(f) == 19_110_297_600  # the GGUF's MXFP4 parameters, exactly
    assert predicted_file_gb(f, "MXFP4") == pytest.approx(g["tensor_bytes"] / 1e9, rel=0.005)
    r = estimate(f, "MXFP4", resolve("RTX 4070 Ti SUPER"), ctx=8192, runtime="lm studio")
    assert r.breakdown["input_embedding_ram"] == pytest.approx(g["input_embedding_bytes"] / 1e9,
                                                               abs=0.001)


def test_after_a_fine_tune_the_head_is_saved_once() -> None:
    """Unsloth's merge of Qwen3-0.6B converted to 596M parameters; its Q8_0 was 0.639 GB."""
    stored = 151936 * 1024
    base = _facts("Qwen__Qwen3-1.7B").model_copy(update={
        "params_total": 751_632_384,
        "extra": {**_facts("Qwen__Qwen3-1.7B").extra, "hidden_size": 1024,
                  "tied_head_stored": stored}})
    after = merged(base)
    assert after.params_total == 751_632_384 - stored and not after.extra["tied_head_stored"]
    assert predicted_file_gb(base, "Q8_0") == pytest.approx(0.799, abs=0.001)
    assert predicted_file_gb(after, "Q8_0") == pytest.approx(0.639, rel=0.02)
    assert merged(_facts("Qwen__Qwen3-4B")) == _facts("Qwen__Qwen3-4B"), "nothing stored twice"
