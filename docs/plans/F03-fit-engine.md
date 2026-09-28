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

**Diffusion (`diffusion.components.v0`, built)**: the peak is the worst phase of load, encode, denoise and decode: the weights resident in it (which depends on the offload strategy: none, model, sequential) plus its activations. The transformer's activations scale with tokens x hidden, a UNet's with latent pixels (x2 under classifier-free guidance), and the VAE decode with output pixels. bitsandbytes and torchao quantize on the GPU while loading, so they add a load phase with every quantized component resident. Constants in `data/runtimes/diffusers/memory.yaml`, fitted by hand to 18 published rows: the [diffusers quantization blog](https://huggingface.co/blog/diffusers-quantization) (FLUX.1-dev, H100, `max_memory_reserved` in GiB: BF16 peak 36.2, bnb nf4 17.3, GGUF Q2_K 17.8; the "9.3" first copied here for FP8 layerwise + group offload is its memory after loading, the peak is 14.2) and the [SDXL optimisation post](https://huggingface.co/blog/simple_sdxl_optimizations) (A100, batch 4). The GGUF rows and the torchao / quanto int8 and FP8 rows land within 5%; torchao int4 and FP8 layerwise casting about 10% low and bitsandbytes 15-26% low, the allocator cache that quantizing or casting at load time leaves behind; the SDXL rows within 18%. Each row records the error it admits to.

**Audio (`audio.whisper.v0`, built)**: weights at the runtime's type plus the runtime's overhead measured in [faster-whisper's benchmark](https://github.com/SYSTRAN/faster-whisper#benchmark) (faster-whisper, whisper.cpp, transformers, openai-whisper on an RTX 3070 Ti and an i7-12700K, batch 1 and 8), scaled to other sizes by params^0.36, the slope of whisper.cpp's own memory table. Weights match every ggml file checked within 5%. The earlier note "large-v3 2.9 GB -> 547 MB at q5_0" mixed two models: large-v3 q5_0 is 1.08 GB, 547 MiB (574 MB) is large-v3-turbo q5_0. Other audio models (TTS, other recognizers): weights plus a flat allowance, `audio.weights.v0`, confidence 0.3.

**Vision, embeddings (`encoder.weights.v0`, built)**: weights x bpw + one batch of activations (tokens x hidden x 2 B x 16; tokens from the sequence length or a ViT's patches) + the CUDA context. Embeddings additionally report index shrink for int8 / binary output vectors (32x) — this is a note, not memory. Nothing measured backs the activation factor yet: confidence 0.4.

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
- FLUX and SDXL rows reproduced from the diffusers posts, each within the error it records.
- Whisper: faster-whisper's benchmark rows within 5%; whisper.cpp ggml file sizes within 5%
  (large-v3 q5_0 1.08 GB, large-v3-turbo q5_0 574 MB).
- Speed: RTX 4090 (1008 GB/s) with an 8B Q4_K_M model ≈ 100+ tok/s order of magnitude; M2 Max (400 GB/s) ≈ 40% of that.

## Reuse

- Notebook repo `data_pipeline/metrics.py`: measured tok/s per model / host is the first calibration set for `speed.bandwidth.v1`.

## TODO

- [x] `FitResult`, `QuantSpec`, `RuntimeSpec` finalised; family registry: `fit.estimate`
      routes by `ModelFacts.family`, and `FAMILY_ARGS` says which arguments each family
      takes, so one call shape (CLI, MCP) serves all five. A plain function per family
      turned out simpler than an `Estimator` protocol
- [x] `data/quants/gguf_bpw.yaml` ingested from `quant-descriptions.ts` with provenance; bnb / AWQ / GPTQ / MLX bpw rows:
      GGUF rows derived from llama.cpp's block formulas and checked against
      `llama-quantize --help`; `data/quants/formats.yaml` has bnb nf4 (4.5, 4.127 with
      double quant), int8, torchao int4 (4.25), AWQ / GPTQ (4.156), MLX (4.5 / 8.5), FP8,
      NVFP4, MXFP4, each derived from its block layout and cited
- [x] LLM weights + KV (GQA, sliding window, MLA, hybrid) + runtime overhead constants.
      KV checked against llama.cpp's own dry-run allocation (`llama-fit-params`) for plain GQA,
      sliding window, a conv hybrid and a Mamba2 hybrid: 0.0% worst disagreement.
- [ ] oobabooga regression port + router
- [x] Speed model (bandwidth / bytes per token; MoE active params; offload penalty).
      Validated on an RTX 4070 Ti SUPER: `rightsize bench` measured 70% of peak, the
      efficiency the model assumes.
- [x] Fine-tune component model + Unsloth floor, interpolated between published rows
      (a step lookup made 16-bit LoRA on a 4B model a 19 GB no_fit on a 16 GB card).
      `rightsize estimate --mode qlora|lora|full`.
- [x] Diffusion estimator + FLUX / SDXL rows (SD3.5 has no published peak we could find;
      the quanto post's PixArt and SD3 rows do not say what they measured, so they are
      left out). Video: tokens scale with frames; the decode assumes four frames at a
      time; confidence 0.3 because nothing measured backs it
- [x] Audio estimator: Whisper per runtime, anything else weights-only
- [x] Vision / embedding estimator: weights + one batch of activations
- [x] Fine-tuning memory for small models with large vocabularies ran low: Unsloth QLoRA on
      Qwen3-0.6B (batch 2, 2048 tokens) raised the card 3.72 GB against 1.82 GB predicted.
      The training step alone (2026-09-28): PyTorch allocated at most 1.80 GB for Qwen3-0.6B
      and 2.76 GB for 1.7B, where the parts came to 0.86 and 1.99 GB; the loss's logits had
      no term. With 0.7 x batch x sequence x vocabulary x 2 bytes (Unsloth chunks its loss)
      and the weights counted as transformers loads them, the parts are -4.0% and +3.7% off
      what was allocated. PyTorch's caching allocator reserved 1.6 GB beyond that and the
      card rose 4.2 and 4.6 GB, room a full card would not have given it: the estimate is
      the need, the rise is what a roomy card shows. Two models of one vocabulary fit the
      0.7; other vocabularies and trainers should be measured before it is trusted there
- [x] After a fine-tune the converter reads the merged model, which stores a tied output head
      once: the Q8_0 of Unsloth's Qwen3-0.6B merge was 0.639 GB against 0.799 GB predicted
      from the checkpoint's count (`tied_head_stored`, F1). `fit.finetune.merged()` drops the
      second copy for every step after a fine-tune, in `recommend` and in the runner's size
      check: 0.633 GB predicted
- [x] llama.cpp keeps the input embedding in system RAM (`data/runtimes/llama.cpp/embedding.yaml`,
      from `src/llama-model.cpp` and `src/llama-quant.cpp` at b11177). With an output head of
      its own the embedding leaves VRAM; tied, llama.cpp puts a copy on the GPU as the head, so
      VRAM keeps it and RAM holds it as well. Its type follows llama-quantize's rules (Q6_K
      when tied, the file type's own otherwise, Q8_0 when the hidden size does not divide the
      block), which give every GGUF fixture's token_embd bytes exactly. Loading seven GGUFs,
      llama.cpp's CPU buffer and CUDA buffer matched the rule to 0.02 MB on all six made with
      llama-quantize's defaults; Unsloth's UD mixes choose their own types, so only a header
      read sizes theirs. Applies to llama.cpp, Ollama and LM Studio, not on unified memory
- [x] Overhead refit for llama.cpp: 0.27 GB + 1% of the weights on the GPU, fitted to
      llama-server serving six GGUFs at ctx 4096 and 16384, VRAM rise less llama.cpp's own
      buffers. About 0.23 GB is the CUDA context in every run; a server's compute buffer is
      small (23-64 MiB) because it keeps logits only for sampled tokens. Error of the total:
      mean 0.0%, worst -7.5%; 0.75 GB + 2% was +54% on average. Perplexity and imatrix passes
      keep every token's logits and run n_batch / ctx sequences at once: `all_logits=True`
      adds 512 x vocab x 4 bytes and the extra KV (Qwen3-1.7B at -c 512: 1.89 GB predicted,
      1.88 measured). Ollama's and LM Studio's constants are still first estimates
- [x] MXFP4_MOE (gpt-oss): llama-quantize writes only the routed experts as MXFP4 and every
      other tensor as Q8_0. params x 4.25 bits said 11.11 GB for gpt-oss-20b; the split, with
      the expert count recovered from the active parameters, says 12.07 against the GGUF's
      12.10 GB of tensors
- [ ] Multi-GPU and offload split
- [x] `bits_that_fit(facts, device, ctx, runtime)` (fit/llm.py): the fit solved for the
      weights. Usable memory less the headroom the verdict keeps, the KV cache and the
      runtime's overhead is what the weights may take; the embedding and an untied head
      stay 16-bit and the rest is shared over the quantized parameters. Qwen3-4B on a
      12 GB card at 8k context: 15.3 bits; at 32k, under 16 and falling with the KV cache.
      `rightsize quantize --to modelopt-auto` hands it to Model Optimizer's AutoQuantize as
      `effective_bits` (F5); GGUF mix candidates from it are still to do
- [ ] Golden tests listed above
      (partial: Llama 3.1 70B KV at 128K, Qwen3-4B Q4_K_M file size, measured Qwen3-1.7B
      file sizes within 3%, the diffusion and Whisper rows (tests/test_families.py); the
      oobabooga points wait on that port)
- [ ] `confidence` and `formula_id` on every result; docs page explaining each formula
      (partial: on every result, and each estimator's module docstring states its formula;
      no docs page yet)
- [ ] Speed for diffusion (s/image), audio (real-time factor) and encoders: compute-bound,
      so it needs FLOPS per device, which the hardware table does not have yet
- [ ] Fine-tuning memory for diffusion (LoRA on FLUX / SDXL); `mode` other than infer raises
      NotImplementedYet for non-LLM families today
- [ ] `recommend` for the other families (which image model fits this card); needs a
      candidate list per family and a quality signal to rank by
