# F5. Framework registry and recipe adapters

Package: `rightsize.registry`. Phase 1. Depends on: F4 (`Plan`). Feeds: F6, F8, generated docs.

## Goal

One registry of the open-source fine-tuning, quantization, export and serving toolkits, each with when-to-use guidance, hardware constraints, an install line, and **renderable recipes** that turn a `PlanStep` into the actual command or config file. Generate one docs page per framework from the registry so the docs never drift. Let third parties add frameworks via an entry point.

## Why it beats what exists

Calculators end at a number. Rightsize ends at `llama-quantize ... Q4_K_M` or an Axolotl YAML you can run. Users get the benefit of 20+ toolkits without learning each one's flags.

## Public API

```python
from rightsize.registry import frameworks, framework, render

frameworks(stage="quantize", family="llm") -> list[FrameworkInfo]
framework("unsloth") -> FrameworkInfo          # description, when to use, hardware, install, recipes
render(step: PlanStep, framework: str | None = None) -> RenderedStep  # command | config_text, install_line, notes
Plan.render(framework=None) -> list[RenderedStep]
```

## Data

Recipe schema (`data/schema/recipe.schema.json`), one YAML per `(framework, stage)` in `data/recipes/<framework>/<stage>.yaml`:

```yaml
framework: llama.cpp
stage: quantize
families: [llm]
hardware: { vendors: [nvidia, amd, apple, cpu] }
install_extra: llamacpp
install_line: "brew install llama.cpp   # or build from source"
inputs:
  input_gguf: { type: path, required: true }
  quant: { type: enum, values: [Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ4_XS], required: true }
  imatrix: { type: path, required: false }
template: |
  llama-quantize {% if imatrix %}--imatrix {{ imatrix }} {% endif %}{{ input_gguf }} {{ output_gguf }} {{ quant }}
source_doc_url: https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize
version_tested: b6000
```

The full inventory, including what vLLM and Hugging Face offer and the libraries verified on 2026-09-25, is in [01-libraries.md](01-libraries.md). The tables below are the registry rows (status: **MVP** = recipe in phase 1, P2 = phase 2):

**Fine-tuning**

| Framework | Families | Hardware | Status |
|---|---|---|---|
| Unsloth | LLM, VLM, TTS, STT | NVIDIA 7.0+, AMD, Intel, Apple | MVP |
| HF TRL v1 + PEFT + bitsandbytes | LLM, VLM | NVIDIA, Intel | MVP |
| Axolotl | LLM, QAT (int8 / int4 / FP8 / NVFP4 / MXFP4) | NVIDIA Ampere+, AMD | MVP |
| MLX `mlx_lm.lora` | LLM | Apple | MVP |
| diffusers training scripts (DreamBooth / LoRA, FLUX QLoRA) | diffusion | NVIDIA | MVP |
| LLaMA-Factory | LLM, VLM; web UI | NVIDIA, AMD, Ascend | P2 |
| kohya sd-scripts, SimpleTuner, ai-toolkit, FluxGym | diffusion | NVIDIA | P2 |
| Ultralytics train | detection | NVIDIA, Apple, CPU | P2 |
| sentence-transformers trainer | embeddings | any | P2 |

**Quantization and export**

| Toolkit | Output | Families | Targets | Status |
|---|---|---|---|---|
| llama.cpp `llama-quantize` (+ imatrix) | GGUF | LLM, VLM (mmproj), embeddings | CPU, any GPU, Apple | MVP |
| Unsloth `save_pretrained_gguf` (25 GGUF types via llama.cpp), `save_pretrained_merged` (16-bit, 8-bit, bnb `merged_4bit`); Studio exports NVFP4 / FP8 / imatrix GGUF | GGUF, safetensors | LLM, VLM | as GGUF / vLLM / SGLang | MVP |
| GGUF-my-repo, MLX-my-repo, bnb-my-repo Hub Spaces (no local GPU needed) | GGUF, MLX, bnb | LLM | any | MVP |
| transformers quantization configs (22 methods: bnb, GPTQModel, AWQ, HQQ, torchao, quanto, FP8 variants, NVFP4, AutoRound, SINQ, ...) | safetensors | LLM, VLM | NVIDIA, Intel, AMD partial, Apple (Metal kernels) | MVP core, P2 rest |
| llm-compressor (vLLM project) | compressed-tensors (FP8, INT8, W4A16, W4A8, NVFP4, SmoothQuant, 2:4) | LLM | vLLM, SGLang, TGI | MVP |
| vLLM on-load quantization (`--quantization fp8`, bitsandbytes, torchao, FP8 KV cache) | in-memory | LLM | Ada / Hopper / Blackwell, AMD MI | MVP (serve recipe flags) |
| `optimum-cli export onnx` + ONNX Runtime quantization | ONNX int8 | vision, embeddings, small LLM | CPU, DirectML, mobile | MVP |
| ExLlamaV3 (EXL3 trellis quant, 1–8 bpw) + TabbyAPI | EXL3 | LLM | NVIDIA Ampere+ only | P2 |
| mistral.rs ISQ (`--isq`, MoQE for MoE); LMDeploy `lite auto_awq` **(verify)**; ik_llama.cpp and KTransformers (MoE on CPU + GPU) | in-memory; AWQ; GGUF variants | LLM | CUDA, Metal, CPU | P2 |
| MLC LLM (`convert_weight`, `compile`: q4f16_1, q3f16_1, q4f16_awq) | MLC | LLM | CUDA, ROCm, Metal, Vulkan, WebGPU, iOS, Android | later (mobile phase) |
| `optimum-cli export openvino` (NNCF) | OpenVINO IR | LLM, vision, embeddings | Intel CPU / iGPU / NPU | MVP |
| MLX `mlx_lm.convert -q`; mflux | MLX | LLM, diffusion | Apple | MVP |
| diffusers `PipelineQuantizationConfig` (bnb, torchao, quanto, GGUF, layerwise fp8, group offload) | safetensors / GGUF | diffusion | NVIDIA | MVP |
| whisper.cpp quantize; faster-whisper / CTranslate2; WhisperKit (Apple); NeMo Parakeet export | GGML; CT2 int8; Core ML; ONNX | audio (STT) | CPU, NVIDIA, Apple | MVP (whisper.cpp, CT2), P2 (WhisperKit, Parakeet) |
| Kokoro, Piper, F5-TTS export paths | ONNX / GGML | audio (TTS) | CPU, Apple, NVIDIA | P2 |
| sentence-transformers `export_*_quantized_*`, `quantize_embeddings`, model2vec; fastembed; Text Embeddings Inference | ONNX / OpenVINO int8; static vectors | embeddings | CPU, GPU | MVP (sentence-transformers), P2 (fastembed, TEI) |
| NVIDIA TensorRT Model Optimizer (`nvidia-modelopt`, Apache-2.0): `mtq.quantize` with `FP8_DEFAULT_CFG`, `INT8_SMOOTHQUANT_CFG`, `INT4_AWQ_CFG`, `NVFP4_DEFAULT_CFG`, MXFP4; `mtq.auto_quantize(constraints={"effective_bits": ...})` per-layer mixed precision; `export_hf_checkpoint` | HF checkpoints for vLLM, SGLang, TensorRT-LLM, Dynamo; TRT-LLM engines | LLM, VLM, diffusion | NVIDIA (GPU needed for calibration) | MVP (PTQ recipes + AutoQuantize); QAT, pruning, distillation, speculative decoding later |
| Intel AutoRound; AMD Quark | gptq / awq / GGUF; ONNX | LLM | Intel; AMD | P2 |
| Nunchaku (SVDQuant); stable-diffusion.cpp; ComfyUI-GGUF | INT4 / NVFP4; GGUF | diffusion | NVIDIA Turing+ / Blackwell; any | P2 |
| torchao `quantize_` / QAT; ONNX Runtime quantization; Ultralytics export | int8 / int4 / fp8; ONNX / TensorRT / OpenVINO / CoreML / TFLite | vision | many | P2 |
| coremltools; Olive; ExecuTorch; Qualcomm AI Hub; LiteRT | mobile formats | all | Apple ANE, Windows NPU, Android / iOS | mobile phase |

**Serve recipes (MVP):** llama.cpp `llama-server`, Ollama `Modelfile` + `ollama create`, vLLM `vllm serve` with quant flags, `mlx_lm.server`.

## Design

- Bundled recipes load from `data/recipes/`; third-party recipe packs register via entry point `rightsize.recipes` pointing at a folder.
- Template renderer: a minimal `{{ var }}` / `{% if %}` subset implemented in-house (no jinja2 in core); jinja2 optional under `[dev]` for validating parity.
- `render()` validates inputs against the recipe's `inputs`, fills defaults from the `PlanStep` (quant, paths, device), and returns `RenderedStep{kind: command|config, text, install_line, notes, source_doc_url}`.
- **Budget-driven mixed precision.** When the target is NVIDIA and the memory budget falls between two uniform formats, the plan offers a `modelopt.auto_quantize` recipe whose `effective_bits` input comes from F3 (`bits_that_fit = (usable_vram - kv - overhead) x 8 / params`, rounded down to 0.1). The same idea maps to llama.cpp as a UD-style mix and to Unsloth Dynamic where a matching upload exists.
- Every calibration-based recipe (AWQ, GPTQ, SmoothQuant, FP8 static, NVFP4, imatrix, OpenVINO static, ModelOpt) declares `calibration.method`, `calibration.samples` (default 128–512 for LLMs, ~200 for classic int8) and `calibration.dataset`; see [02-quantization-concepts.md](02-quantization-concepts.md).
- `docs.py` writes `docs/frameworks/<framework>.md` from the registry: what it is, when to pick it, hardware, install, recipes with example renders. Run in CI; diff must be clean.

## MVP scope

The 14 MVP recipes above plus 4 serve recipes; docs generation; entry point discovery.

## Follow-up research

- Pin and verify exact CLI flags per toolkit version; record `version_tested`.
- Unsloth NVFP4 / FP8 export: confirmed for Unsloth Studio after training (2026-09-25); the library-level API for NVFP4 export is still **(verify)**. `mlx_lm` mixed-precision / AWQ recipes **(verify)**.
- Which recipes need a preceding `require` step (imatrix, calibration data for AWQ / GPTQ).

## Tests

- Every recipe renders with sample inputs; rendered text snapshot-tested.
- Schema validation of every recipe file; `source_doc_url` and `version_tested` required.
- Nightly workflow (`.github/workflows/recipes-nightly.yml`) installs each toolkit in a matrix and runs the rendered command with `--help` or a dry-run flag.

## Reuse

- Notebook repo `data_pipeline/scripts/serve_llama.sh` and `serve_vllm.sh`: first serve recipes; docker-compose `local-ai*` profiles inform the vLLM flags.

## TODO

- [x] `data/schema/recipe.schema.json`; `FrameworkInfo`, `RenderedStep` types. Each framework's
      `framework.yaml` beside its recipes (`data/schema/framework.schema.json`): summary,
      stages, formats, hardware, install, a `finetune` role (modes, recipe, what it writes,
      QLoRA's storage, a step it needs first, the vendors it is the default trainer on) and
      the defaults its recipes read. Trainer choice and every framework-specific default left
      `recommend.py` and `render_plan.py`; a plugin shipping a descriptor and a recipe becomes
      a default trainer with no code change (`tests/test_frameworks.py`). MLX QLoRA plans now
      record MLX's own 4-bit rather than bitsandbytes NF4
- [x] Minimal template renderer with tests (token-by-token, so Windows paths survive)
- [x] Loader for bundled recipes + entry-point discovery (`rightsize.recipes`; a plugin may
      add recipes but not replace a bundled one)
- [x] 14 MVP quantize / fine-tune recipes + 4 serve recipes, each with `source_doc_url` and `version_tested`:
      24 recipes over 14 frameworks. llama.cpp's six were run (`verified: run`, `help` for the
      server). The other 18 (Unsloth, TRL, Axolotl, MLX-LM x3, vLLM, Ollama x2, Optimum Intel,
      llm-compressor x2, transformers + bnb, diffusers, whisper.cpp x2, CTranslate2,
      sentence-transformers) were checked on 2026-09-25 against each tool's current docs and,
      where the docs were stale, its source: `verified: docs`, nothing run yet. Traps the check
      found, each now a recipe note: Unsloth pins trl<=0.24 and datasets<4.4 (its own
      environment); vLLM 0.30 moved bitsandbytes and GGUF into plugins; whisper.cpp's quantize
      binary is whisper-quantize; transformers 5 dropped `load_in_4bit=` and `torch_dtype=`;
      the diffusers docs' torchao example fails on torchao 0.18. Config recipes declare a
      `language`, and CI parses every Python and YAML render. Since run end to end as well:
      `unsloth/sft`, and the six recipes behind `rightsize quantize --to` (F8, 2026-09-28)
- [x] Recipes that make a format declare it as a target: the name `--to` takes, the inputs
      that select it, the `quants/formats.yaml` entry that sizes it, the bits the embedding
      and an untied head keep, which files are the weights, and the `gate` recipe that
      checks the output (`quantize --to X --eval`). Fourteen targets over seven
      toolkits: fp8, w4a16, awq (llm-compressor), nf4 (transformers), openvino-int4/int8
      (Optimum Intel), modelopt-fp8/int8-sq/int4-awq/nvfp4 (Model Optimizer),
      mlx-4bit/8bit (MLX LM), onnx-int8 (sentence-transformers), ct2-int8
      (CTranslate2); the
      framework pages name each recipe's formats
- [x] Model Optimizer recipes: FP8, INT8 SmoothQuant, INT4 AWQ, NVFP4 (`mtq.quantize` +
      `export_hf_checkpoint`) as `modelopt/ptq`, run and gated on Qwen3-0.6B (F8); the
      export serves through vLLM (`--quantization modelopt`), SGLang and TensorRT-LLM
- [x] AutoQuantize recipe (`modelopt/autoquant`, `--to modelopt-auto`) with `effective_bits`
      from F3's `bits_that_fit` unless `--set` names one. Qwen3-1.7B for a 12 GB card at 57k
      context: 11.7 bits, 66 layers at FP8, 46 kept 16-bit, one at INT4 AWQ, 2.516 GB as
      predicted, gate pass at KLD 0.010 where every uniform 4-bit format of that model
      failed. The target says what its bits average over (`bits_over: linear`, Model
      Optimizer's definition), since the export lands on budget only when the two agree
- [x] `render()` and `Plan.render()`. Two-stage plans render the fine-tune too: Unsloth on
      NVIDIA and Intel, Axolotl on AMD, MLX-LM on Apple (with an `mlx_lm.convert -q` step first
      for QLoRA); llama.cpp then converts the merged model. Full fine-tunes have no recipe yet
- [x] `docs.py` generator; CI diff check (tests/test_framework_docs.py)
- [ ] Nightly dry-run workflow
- [ ] Verify the two **(verify)** items
