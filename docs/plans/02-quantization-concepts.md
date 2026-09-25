# 02. Quantization concepts and method selection (reference)

Working notes distilled from three references, kept here so the fit engine (F3), the rules (F4) and the recipes (F5) cite the same vocabulary. Read 2026-09-25.

- Hugging Face Optimum, [Quantization concept guide](https://huggingface.co/docs/optimum/en/concept_guides/quantization)
- NVIDIA, [Model quantization: concepts, methods, and why it matters](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/)
- NVIDIA, [Optimizing LLMs for performance and accuracy with post-training quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)

## 1. Vocabulary the SDK uses

| Term | Meaning | Where it shows up |
|---|---|---|
| Affine scheme | `x = S * (x_q - Z)`; scale `S` (float), zero-point `Z` (int); `x_q = clip(round(x / S + Z), qmin, qmax)` | `QuantSpec.scheme` |
| Symmetric scheme | range `[-a, a]`, `Z = 0`, int8 range `[-127, 127]`; cheaper kernels | `QuantSpec.scheme` |
| Granularity | per-tensor (one `S, Z`), per-channel (one per output channel), per-block / per-group (e.g. group size 32, 64, 128) | `QuantSpec.group_size`; real bits-per-weight must include the scales (F3) |
| Weight-only (W4A16) | weights quantized, activations stay 16-bit; no calibration data strictly needed; memory-bound wins | GGUF, AWQ, GPTQ, bnb nf4, EXL3 |
| Weight-and-activation (W8A8, W4A4) | activations quantized too; needs calibration; compute-bound wins on hardware with the matching tensor cores | FP8, INT8 SmoothQuant, NVFP4 |
| Dynamic quantization | activation ranges computed at runtime; easy, slightly slower, unavailable on some hardware | ONNX Runtime dynamic, vLLM `fp8` dynamic |
| Static quantization | ranges fixed at quantization time by running calibration data through observers | INT8 SmoothQuant, static FP8, OpenVINO static |
| QAT | fake-quant ops during training; straight-through estimator | Axolotl QAT, torchao QAT, Brevitas |
| Calibration technique | min-max (weights), moving-average min-max (activations), histogram: entropy, MSE, percentile | recipe input `calibration.method` |
| Calibration set | ~200 examples (Optimum guide); 128–512 samples (TensorRT Model Optimizer) | recipe input `calibration.samples` |
| KV-cache quantization | separate precision for the KV cache (FP8, int8, q4/q8 in llama.cpp); several GB at long context | `estimate(kv_dtype=...)` in F3 |
| Accumulation dtype | int8 accumulates in int32, fp16 in fp16, bf16 in fp32 | explains why some backends need fp32 accumulate |

## 2. PTQ methods the recipes name

| Method | What it does | Typical format | Tools |
|---|---|---|---|
| Round-to-nearest with min-max | baseline | any | everything |
| SmoothQuant | per-channel scaling migrates activation outliers into weights so W8A8 works | INT8 W8A8, FP8 | llm-compressor, TensorRT Model Optimizer, Quark |
| AWQ | protects salient weight channels found from activations | INT4 W4A16, group-wise | llm-compressor, AutoAWQ (in transformers), LMDeploy, mlc `q4f16_awq` |
| GPTQ | Hessian-guided row-wise error compensation | INT2–8 W4A16 | GPTQModel, llm-compressor, AutoRound (improves on it) |
| imatrix (llama.cpp) | importance matrix from calibration text steers K- and I-quants; required for i-quants | GGUF | `llama-imatrix`, Unsloth Dynamic (public imatrix) |
| AutoQuantize (Model Optimizer) | per-layer gradient-based format selection under a constraint | mixed | TensorRT Model Optimizer |
| Layer-selective mixing | keep sensitive layers at higher bits | UD-Q4_K_XL, Dynamic NVFP4 (FP8 / BF16 for important layers) | Unsloth, Pruna auto |

## 3. Formats and hardware (rules data for F4)

| Format | Scope | Hardware | Runtimes | When |
|---|---|---|---|---|
| FP8 E4M3 | W8A8, KV | Ada, Hopper, Blackwell; AMD MI300 | vLLM, SGLang, TensorRT-LLM, TGI | throughput serving on data-center or 40/50-series; near-lossless |
| INT8 SmoothQuant | W8A8 (per-channel W, per-tensor A) | Turing+ (vLLM), CPUs | vLLM, TensorRT-LLM, OpenVINO | older GPUs and CPUs where FP8 is unavailable |
| INT4 AWQ / GPTQ | W4A16 group-wise | Turing+ (AWQ), Volta+ (GPTQ); Marlin kernels Ampere+ | vLLM, SGLang, TGI, LMDeploy, transformers | memory-bound, small batch, consumer GPUs |
| NVFP4 | W4A4 (E2M1 with two-level scaling) | Blackwell only (RTX 50, B200, DGX Spark) | vLLM, SGLang, TensorRT-LLM | 2–3x decode throughput, near-original accuracy per NVIDIA |
| MXFP4 | W4 block-scaled | Blackwell (Marlin MXFP4 not on Turing) | vLLM, transformers | gpt-oss native format |
| GGUF K-/I-quants | weight-only, block-wise (32) with scale mixes | any CPU / GPU / Apple | llama.cpp family | the local default; i-quants need imatrix |
| bnb nf4 / int8 | weight-only, block 64 (nf4) | NVIDIA all; ROCm / Metal partial | transformers, vLLM, diffusers | QLoRA base; on-the-fly loading |
| MLX 4/8-bit, AFQ | weight-only, group 32/64 | Apple | mlx_lm, mistral.rs | Apple default |
| EXL3 | weight-only trellis 1–8 bpw | NVIDIA Ampere+ | ExLlamaV3 / TabbyAPI | best quality per bit on NVIDIA consumer |

## 4. Selection heuristics that become rules or notes

1. Start with weight-only (GGUF / AWQ / nf4) for single-user local inference: the workload is memory-bound and weight-only gives the memory win without calibration risk.
2. Use W8A8 (FP8 on Ada+, INT8 SmoothQuant elsewhere) when serving many users: compute-bound, needs the matching tensor cores.
3. NVFP4 only on Blackwell; Dynamic NVFP4 keeps sensitive layers at FP8 / BF16.
4. Try dynamic before static quantization (Optimum's practical order); go to QAT only when PTQ accuracy fails the gate (F8).
5. Calibration set: 128–512 samples for LLM PTQ; ~200 for classic static int8; the set should resemble the deployment domain (agentic / coding vs chat).
6. Quantize the KV cache separately at long context; it is often the larger share of memory past 32K.
7. **Small models are the exception.** Under ~3B parameters, nf4 can raise energy use 25–56% because dequantization overhead outweighs bandwidth savings, and int8 mixed precision adds 17–33% energy over FP16 (EcoCompute benchmarks cited in the Optimum guide). If a 16-bit or int8 variant fits, prefer it; rank the 4-bit variant lower with a note. Batch size dominates per-token energy (84–96% reduction from batch 1 to 8–64).
8. Weight-only formats still report real bits-per-weight including scales (Q4_K_M ≈ 4.85, nf4 ≈ 4.5 with absmax), never nominal bits (F3).

## 5. Where this lands

- F3: `QuantSpec` gains `scheme`, `group_size`, `activations_bits`, `kv_dtype`; formula ids reference the table in section 3.
- F4: rules `fp8_needs_ada_or_newer`, `nvfp4_needs_blackwell`, `marlin_mxfp4_not_turing`, `awq_turing_plus`, `gptq_volta_plus`, `w8a8_for_multiuser_serving`, `weight_only_for_single_user`, `small_model_prefer_8_or_16_bit`, `iquant_requires_imatrix`, `kv_quant_at_long_context` with these sources.
- F5: recipe inputs `calibration.method`, `calibration.samples`, `calibration.dataset` on every calibration-based recipe; defaults from section 1.
