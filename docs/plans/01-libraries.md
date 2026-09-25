# 01. Library and toolkit inventory

The registry (F5) renders recipes for these. This is the inventory of what exists per stage and family, verified 2026-09-25 unless marked **(verify)**, with the status Rightsize gives each one: **MVP** recipe in phase 1, **P2** phase 2, **later**, or **runtime** (a target the recommender knows about, no recipe needed beyond serve). Competitors are in [00-competitors.md](00-competitors.md); this file is about toolkits we integrate, not products we compete with.

## 1. What the two big ecosystems offer

### vLLM

- **llm-compressor** (vLLM project): the recommended tool for producing checkpoints for vLLM. Produces `compressed-tensors` format: FP8 W8A8, INT8 W8A8, INT4 W4A16, INT8 W4A8, NVFP4, SmoothQuant, GPTQ, AWQ and 2:4 sparsity. **MVP**.
- **On-load quantization inside vLLM**: `--quantization fp8` dynamic on Ada/Hopper/Blackwell, bitsandbytes on-the-fly, torchao; plus quantized KV cache (FP8). These are serve-recipe flags, not separate tools. **MVP** (in the vLLM serve recipe).
- **Checkpoints vLLM loads**: AWQ (Turing+), GPTQ / GPTQModel (Volta+), Marlin (Ampere+; Turing except MXFP4), compressed-tensors INT8 W8A8 (Turing+, x86 and Arm CPU), W4A8 (Arm CPU only), FP8 W8A8 (Ada, Hopper, AMD MI), bitsandbytes (all NVIDIA), NVIDIA Model Optimizer, AMD Quark, Intel Neural Compressor, GGUF (experimental). This matrix is rules data for F4, taken from [vLLM's quantization docs](https://docs.vllm.ai/en/latest/features/quantization/index.html).

### Hugging Face

- **transformers quantization configs** ([overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)), 22 methods as of 2026-09-25. On-the-fly (no calibration): bitsandbytes 4/8, HQQ 1–8, optimum-quanto 2/4/8, torchao 4/8, EETQ 8, FBGEMM FP8, FineGrained FP8, NVFP4 (Blackwell kernels), HIGGS 2/4, FP-Quant 4, Four Over Six 4, SINQ 2–8, Metal 2/4/8 (Apple, Hub kernels). Calibration-based: GPTQModel 2/3/4/8, AWQ 4, AutoRound 2/3/4/8 (Intel), AQLM 1/2, VPTQ 1–8, SpQR 3, compressed-tensors, Quark (AMD), GGUF (load only). PEFT fine-tuning on top of the quantized model: bitsandbytes, AWQ, GPTQModel, HQQ, AQLM, compressed-tensors, EETQ. **MVP**: bitsandbytes, GPTQModel, AWQ, HQQ, torchao, quanto, FP8 variants; **P2**: AutoRound, NVFP4, SINQ, Quark, AQLM / VPTQ / SpQR (research-grade).
- **One-click Hub Spaces**: [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) (any Hub model -> GGUF with chosen quant, optional imatrix and split), [MLX-my-repo](https://huggingface.co/spaces/mlx-community/mlx-my-repo), [bnb-my-repo](https://huggingface.co/spaces/bnb-community/bnb-my-repo). These are the "no local GPU" path: a recipe can output "open this Space with these settings" instead of a command. **MVP** (zero-install recipe variant).
- **optimum** family: `optimum-onnx` (export Transformers, Diffusers, Sentence Transformers, timm to ONNX; graph optimization; ONNX Runtime quantization) **MVP**; `optimum-intel` (OpenVINO IR via `optimum-cli export openvino`, NNCF weight compression, Intel Neural Compressor static / dynamic / QAT) **MVP**; `optimum-executorch` (mobile, `.pte`) **later**; `optimum-quanto` (in transformers table) **MVP**; optimum-amd, optimum-neuron, optimum-nvidia **later**.
- **PEFT + TRL + bitsandbytes**: the reference QLoRA path. **MVP**.
- **Serving**: Text Generation Inference (bnb, GPTQ, AWQ, Marlin, EETQ, EXL2, FP8) **runtime**; Text Embeddings Inference (BERT / XLM-R / Nomic / GTE / Qwen3 / ModernBERT / Gemma3 embeddings) **runtime**.
- **Hub metadata we ingest**: safetensors header via Range requests, `config.json`, model card `base_model` and the model tree's "Quantizations" section (tag `base_model:quantized`), hardware compatibility panel inputs (`hardwareItems`), `huggingface.js` GGUF quant descriptions. See F1, F2.
- **accelerate estimate-memory**: load-memory estimate on the meta device; needs transformers. We do not depend on it (see 00-competitors).

## 2. Inventory by stage

Status column: MVP / P2 / later / runtime. Families: L = LLM and VLM, D = diffusion (image, video), A = audio (STT, TTS), V = vision, E = embeddings.

### 2.1 Fine-tuning

| Library | Families | Hardware | Notes | Status |
|---|---|---|---|---|
| Unsloth | L, VLM, TTS, STT, D, E | NVIDIA 7.0+, AMD, Intel, Apple | fastest single-GPU QLoRA; also exports (see 2.2) | MVP |
| TRL + PEFT + bitsandbytes | L, VLM | NVIDIA, Intel | reference QLoRA; GRPO / DPO | MVP |
| Axolotl | L; QAT int8 / int4 / FP8 / NVFP4 / MXFP4 | NVIDIA Ampere+, AMD | YAML-config driven; multi-GPU | MVP |
| MLX `mlx_lm.lora` | L | Apple | LoRA / QLoRA / DoRA on unified memory | MVP |
| diffusers training scripts | D | NVIDIA | DreamBooth, LoRA, FLUX QLoRA (~9 GB) | MVP |
| LLaMA-Factory | L, VLM | NVIDIA, AMD, Ascend | web UI, 100+ models | P2 |
| ms-swift | L, VLM, A | NVIDIA, Ascend | broad Chinese-ecosystem coverage | P2 |
| kohya sd-scripts, SimpleTuner, ai-toolkit, OneTrainer, FluxGym, musubi-tuner (video) | D | NVIDIA (SimpleTuner: no Apple) | LoRA for SD / SDXL / FLUX / Wan | P2 |
| NVIDIA NeMo / NeMo AutoModel | L, A (Parakeet, Canary) | NVIDIA | the ASR fine-tune path for Parakeet | P2 |
| Ultralytics `train` | V | NVIDIA, Apple, CPU | YOLO | P2 |
| sentence-transformers trainer; FlagEmbedding | E | any | contrastive fine-tune | P2 |
| LitGPT, torchtune (wound down 2025), DeepSpeed / Megatron (multi-node), Oumi, Kiln, Transformer Lab, H2O LLM Studio | L | mixed | platforms or large-scale; recipes later or never | later |

### 2.2 Quantization and export

| Toolkit | Output | Families | Targets | Notes | Status |
|---|---|---|---|---|---|
| llama.cpp `llama-quantize` + `llama-imatrix` | GGUF (Q2–Q8 K-quants, IQ, UD-style mixes) | L, VLM (mmproj), E | CPU, any GPU, Apple | the distribution standard | MVP |
| Unsloth `save_pretrained_gguf` / `push_to_hub_gguf` | GGUF, 25 quant types (f32 … iq2_xxs), llama.cpp under the hood | L | as GGUF | `save_pretrained_merged` 16-bit, 8-bit, `merged_4bit` (bnb); Studio also exports NVFP4, FP8 and imatrix GGUF; Dynamic quants (UD-*, Dynamic NVFP4) are Unsloth-produced, imatrix file public | MVP |
| transformers quantization configs | safetensors | L, VLM | see section 1 | 22 methods | MVP / P2 |
| llm-compressor | compressed-tensors | L | vLLM, SGLang, TGI | FP8, INT8, W4A16, NVFP4, SmoothQuant, 2:4 | MVP |
| GGUF-my-repo, MLX-my-repo, bnb-my-repo Spaces | GGUF, MLX, bnb | L | any | zero-install path on the Hub | MVP |
| MLX `mlx_lm.convert -q`; mflux | MLX (2/3/4/6/8-bit, AFQ) | L, D | Apple | `--q-bits`, `--q-group-size`; mixed / AWQ **(verify)** | MVP |
| diffusers `PipelineQuantizationConfig` | safetensors / GGUF | D | NVIDIA | bnb, torchao, quanto, GGUF, layerwise fp8, group offload | MVP |
| `optimum-cli export openvino` (NNCF) | OpenVINO IR int8 / int4 | L, V, E | Intel CPU / iGPU / NPU | weight compression, AWQ-style data-aware | MVP |
| `optimum-cli export onnx` + ONNX Runtime quantization | ONNX int8 / fp16 | V, E, small L | CPU, DirectML, mobile | dynamic and static quant | MVP |
| whisper.cpp `quantize`; faster-whisper / CTranslate2 `ct2-transformers-converter --quantization int8` | GGML; CT2 int8 / int8_float16 | A (STT) | CPU, NVIDIA, Apple | | MVP |
| sentence-transformers `export_dynamic_quantized_onnx_model`, `export_static_quantized_openvino_model`, `quantize_embeddings` (int8 / binary); model2vec | ONNX / OpenVINO int8; static vectors | E | CPU | output-vector quant shrinks the index 4–32x | MVP |
| ExLlamaV3 (EXL3, 1–8 bpw trellis / QTIP; v1.5.1 2026-09-20) | EXL3 | L | NVIDIA Ampere+ only, CUDA 12.4+; served by TabbyAPI | ExLlamaV2 archived; not loadable by vLLM / SGLang | P2 |
| mistral.rs ISQ (`--isq 2..8`, MoQE for MoE experts, AFQ on Metal) | in-memory | L, VLM | CUDA, Metal, CPU | quantizes at load, model larger than RAM can stream | runtime + P2 recipe |
| MLC LLM (`mlc_llm convert_weight`, `compile`) | q4f16_1, q4f32_1, q3f16_1, q4f16_awq | L | CUDA, ROCm, Metal, Vulkan, OpenCL, WebGPU, iOS, Android | one model to every backend incl. browser | later (mobile phase) |
| LMDeploy `lmdeploy lite auto_awq` **(verify)** | AWQ W4A16 | L | NVIDIA | TurboMind runtime | P2 |
| NVIDIA TensorRT Model Optimizer | FP8 / NVFP4 / INT4 checkpoints, TRT-LLM engines | L, D | NVIDIA | | P2 |
| Intel AutoRound; Intel Neural Compressor | gptq / awq / GGUF; int8 | L, V | Intel, CUDA | AutoRound tops Intel's low-bit leaderboard | P2 |
| AMD Quark; Brevitas (QAT) | ONNX / safetensors / GGUF | L, V | AMD | | P2 |
| Nunchaku (SVDQuant INT4 / NVFP4) | Nunchaku | D | NVIDIA Turing+ / Blackwell | 4-bit FLUX at ~1/4 memory | P2 |
| stable-diffusion.cpp; ComfyUI-GGUF (city96) | GGUF | D | CPU, any GPU | | P2 |
| torchao `quantize_` / QAT; Ultralytics `export` | int8 / int4 / fp8; ONNX / TensorRT / OpenVINO / CoreML / TFLite | V, L | NVIDIA, Intel, Apple, mobile | | P2 |
| ik_llama.cpp (IQ*_K, KT quants), KTransformers (MoE experts on CPU, attention on GPU) | GGUF variants | L (MoE) | consumer GPU + big RAM | the DeepSeek-at-home path | P2 |
| BitNet.cpp (1.58-bit) | ternary | L (BitNet models only) | CPU | | later |
| AQLM, VPTQ, SpQR, QuIP#, HIGGS | extreme low-bit | L | NVIDIA | research-grade, slow to produce | later |
| coremltools; Microsoft Olive; ExecuTorch (`optimum-executorch`); Qualcomm AI Hub; LiteRT / AI Edge Torch; AIMET; MNN, NCNN | mobile formats | all | Apple ANE, Windows NPU, Android / iOS | mobile phase | later |
| Pruna `smash` | many | all | NVIDIA | auto-search over methods; partly paid | later |

### 2.3 Runtimes the recommender knows about

| Runtime | Formats | Hardware | Notes |
|---|---|---|---|
| llama.cpp, Ollama, LM Studio, Jan, KoboldCpp, text-generation-webui | GGUF | everything | Ollama 0.19+ uses MLX on Apple |
| vLLM, SGLang, TGI, LMDeploy, Aphrodite | safetensors quants (see section 1), FP8 | NVIDIA, AMD, Intel, TPU | multi-user serving |
| TensorRT-LLM | TRT engines | NVIDIA | |
| mlx_lm, mlx-vlm, mflux, WhisperKit, DiffusionKit | MLX / Core ML | Apple | |
| mistral.rs, candle | safetensors, GGUF, ISQ | CUDA, Metal, CPU | Rust |
| ExLlamaV3 / TabbyAPI | EXL3 | NVIDIA | |
| MLC LLM, WebLLM | MLC | all incl. browser and phones | |
| OpenVINO GenAI, ONNX Runtime GenAI, Foundry Local, Lemonade, Nexa, Intel AI Playground | IR / ONNX / GGUF | Intel, AMD, Qualcomm NPUs, Windows | vendor stacks |
| ComfyUI, stable-diffusion.cpp, Draw Things | safetensors / GGUF / Nunchaku | NVIDIA, Apple, CPU | diffusion |
| whisper.cpp, faster-whisper, WhisperKit, sherpa-onnx, NeMo (Parakeet), Moonshine; Kokoro, Piper, F5-TTS, Chatterbox | GGML / CT2 / ONNX / MLX | CPU, NVIDIA, Apple, mobile | audio |
| TEI, fastembed (ONNX int8), Infinity, llama.cpp embeddings | ONNX / safetensors / GGUF | CPU, GPU | embeddings |
| ExecuTorch, Core ML, LiteRT-LM, MediaPipe, MNN-LLM | mobile | iOS, Android | mobile phase |

### 2.4 Evaluation (for the F8 gate)

`llama-perplexity --kl-divergence` (llama.cpp), lm-evaluation-harness, lighteval, EvalPlus (code), MTEB (embeddings), jiwer / WER for STT, Ultralytics `val` (mAP). LocalBench and Intel's low-bit leaderboard as external references (see 00-competitors section 3.9).

### 2.5 Small libraries the SDK itself may use

| Need | Candidate | Decision |
|---|---|---|
| GGUF header reading | `gguf` (official, llama.cpp) needs numpy; `@huggingface/gguf` is JS | write our own ~200-line reader; core stays numpy-free (F1) |
| safetensors header | `safetensors` package not needed; the header is 8 bytes + JSON | own reader over httpx Range (F1) |
| GPU detection | `nvidia-ml-py` (NVML) is the accurate path; `nvidia-smi` subprocess needs no dependency | subprocess in core, NVML under `[detect]` (F2) |
| CPU / RAM | `psutil`, `py-cpuinfo` | stdlib `os` / `/proc` / `sysctl` / `wmic` in core; psutil under `[detect]` (F2) |
| Templates | jinja2 | own minimal renderer in core; jinja2 in `[dev]` for parity tests (F5) |

## 3. Gaps this inventory closes in the plans

- F05 table gains: Unsloth as an export tool, ExLlamaV3, mistral.rs ISQ, MLC LLM, LMDeploy, ik_llama.cpp / KTransformers, GGUF-my-repo family, optimum-onnx, TGI / TEI, NeMo, OneTrainer / musubi, WhisperKit / Parakeet / Kokoro / Piper, fastembed.
- F04 rules gain the verified vLLM hardware matrix and the transformers on-the-fly vs calibration split.
- The **(verify)** on "Unsloth NVFP4 / FP8 export" is resolved for Unsloth Studio (yes, after training). The library-level API for NVFP4 export remains **(verify)**.
