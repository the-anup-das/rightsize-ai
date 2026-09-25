# F3. Fit engine

Package: `rightsize.fit`. Phase 1 (LLM, audio), phase 1 late (diffusion tables), later (vision, embeddings beyond weight-only). Depends on: F1, F2, data seed. Feeds: F4.

## Goal

For `(model, quant, runtime, device, ctx | resolution | batch, mode)` return a `FitResult`: predicted VRAM and system RAM with a component breakdown, a fit verdict, speed (tok/s or s/image), a confidence, and the `formula_id` that produced it.

## Why it beats what exists

One interface over five families; fine-tuning and inference in the same engine; speed is part of the verdict; the memory breakdown is as explicit as Red Hat's estimator and Will It Run AI's diffusion calculator; the llama.cpp path uses oobabooga's regression fitted on 19,517 measurements instead of a nominal-bits formula.

## Public API

```python
from rightsize.fit import estimate, Estimator

estimate(model: ModelFacts, quant: QuantSpec, runtime: RuntimeSpec, device: Device,
         mode: Mode = Mode.infer, ctx: int = 8192, batch: int = 1,
         resolution: tuple[int, int] | None = None, frames: int | None = None) -> FitResult

class Estimator(Protocol):      # one per Family, registered by family
    formula_id: str
    def estimate(...) -> FitResult
```

## Data and formulas

**LLM (analytic, `llm.analytic.v1`)**

- Weights = `params x real_bpw / 8`. Real bits-per-weight from the GGUF table, not nominal bits (Q4_K_M ≈ 4.85 bpw, Q2_K 2.625, IQ2_XXS 2.06) — `data/quants/gguf_bpw.yaml` from [`quant-descriptions.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/gguf/src/quant-descriptions.ts). bnb nf4 ≈ 4.5 bpw incl. absmax; AWQ / GPTQ int4 g128 ≈ 4.25.
- KV cache = `2 x layers x kv_heads x head_dim x ctx x batch x bytes_per_elem`, GQA-aware; per-architecture overrides from `kv_overrides.yaml` (MLA compressed KV for DeepSeek, sliding-window layers for Gemma, hybrid SSM layers with no KV).
- Runtime overhead constants (calibration data, `data/quality/overheads.yaml`): llama.cpp ≈ 0.75 GB + 2%, Ollama ≈ 0.9 GB + 3%, vLLM CUDA-graph footprint per GPU class; headroom 15–20%.
- Apple: usable = 70% of unified memory unless raised.

**LLM llama.cpp path (`llm.gguf.oobabooga.v1`)**: [oobabooga's regression](https://oobabooga.github.io/blog/posts/gguf-vram-formula/) (median error 365 MiB, models partial offload). Use when the file is GGUF and the runtime is llama.cpp / Ollama / LM Studio.

**Speed (`speed.bandwidth.v1`)**: decode tok/s ≈ `bandwidth / bytes_read_per_token`, where bytes per token = active weights + KV read at current ctx. MoE uses `params_active`. Verdict flags below 5 tok/s.

**Fine-tuning (`ft.components.v1`)**: weights + gradients + optimizer states + activations, following Red Hat's component model. Full fine-tune ≈ 16–20 bytes/param with AdamW mixed precision. LoRA = 16-bit base + adapters + activations. QLoRA = NF4 base + bf16 adapters. Floors from [Unsloth's requirements table](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) (QLoRA / LoRA GB: 7B 5/19, 8B 6/22, 14B 8.5/33, 27B 22/64, 70B 41/164) as a **lookup**, never interpolated.

**Diffusion (`diffusion.table.v1`)**: peak = component weights (transformer + text encoders + VAE) + activations(resolution x batch x frames) + VAE decode spike. No closed form; tables keyed on `(model, quant, offload_strategy)` with system-RAM cost. Seed from the [HF diffusers quantization blog](https://huggingface.co/blog/diffusers-quantization) FLUX table (BnB4 17.3 GB peak, GGUF Q2_K 17.8, fp8 layerwise + group offload 9.3) and SDXL / SD3.5 published numbers. Nunchaku INT4 / NVFP4 rows.

**Audio (`audio.table.v1`)**: Whisper large-v3 2.9 GB fp16 -> 547 MB at whisper.cpp q5_0 (+0.1–0.3 pp WER); faster-whisper int8 ≈ 4 GB working set; TTS (Kokoro < 2 GB).

**Vision, embeddings (`weights_only.v1`)**: weights x bpw + fixed runtime overhead. Embeddings additionally report index shrink for int8 / binary output vectors (32x) — this is a note, not memory.

## Design

- `Estimator` implementations: `llm.py`, `diffusion.py`, `audio.py`, `vision.py`, `embedding.py`; registry keyed by `Family`; a router picks the formula (`gguf.oobabooga` vs `analytic`) and records it in `formula_id`.
- Every result has `breakdown` (weights, kv_cache, activations, optimizer, overhead, offloaded), `confidence` (1.0 measured table row, 0.8 fitted regression, 0.6 analytic, 0.4 heuristic), and `notes` (assumptions).
- Multi-GPU: sum memory across devices for weights, KV on each device proportional to layers; flag interconnect assumptions.
- Offload: when weights exceed VRAM, compute the layer split and the resulting speed penalty (CPU bandwidth), like gguf-parser's `--gpu-layers`.

## MVP scope

LLM analytic + oobabooga path + speed; fine-tune components with Unsloth floors; audio table. Diffusion FLUX / SDXL / SD3.5 table rows. Vision and embeddings weight-only.

## Follow-up research

- DiT activation memory as a function of resolution and frames (needed for video).
- MLA and hybrid KV rules per architecture; validate against DeepSeek and Qwen3-Next.
- Measured overheads per runtime version (feeds F9).
- MoE speed model with expert routing and offloaded experts.

## Tests (golden, `tests/golden/fit/`)

- Llama 3.1 70B KV at 128K ctx, bf16, GQA 8 heads ≈ 42.9 GB.
- Unsloth floors reproduced exactly for the table rows.
- oobabooga sample points within its stated median error.
- FLUX table rows reproduced from the diffusers blog.
- Whisper large-v3 q5_0 = 547 MB.
- Speed: RTX 4090 (1008 GB/s) with an 8B Q4_K_M model ≈ 100+ tok/s order of magnitude; M2 Max (400 GB/s) ≈ 40% of that.

## Reuse

- Notebook repo `data_pipeline/metrics.py`: measured tok/s per model / host is the first calibration set for `speed.bandwidth.v1`.

## TODO

- [ ] `FitResult`, `QuantSpec`, `RuntimeSpec` finalised; `Estimator` protocol; family registry
- [ ] `data/quants/gguf_bpw.yaml` ingested from `quant-descriptions.ts` with provenance; bnb / AWQ / GPTQ / MLX bpw rows
- [ ] LLM weights + KV (GQA, overrides) + runtime overhead constants
- [ ] oobabooga regression port + router
- [ ] Speed model (bandwidth / bytes per token; MoE active params; offload penalty)
- [ ] Fine-tune component model + Unsloth floor lookup
- [ ] Diffusion table schema + FLUX / SDXL / SD3.5 rows + estimator
- [ ] Audio table + estimator
- [ ] Vision / embedding weight-only estimator
- [ ] Multi-GPU and offload split
- [ ] `bits_that_fit(model, device, ctx, runtime)` helper: the effective bits-per-weight a budget allows after KV and overhead; feeds ModelOpt AutoQuantize and GGUF mix candidates (F4, F5)
- [ ] Golden tests listed above
- [ ] `confidence` and `formula_id` on every result; docs page explaining each formula
