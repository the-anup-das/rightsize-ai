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

Registry rows (status: **MVP** = recipe in phase 1, P2 = phase 2):

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
| transformers quantization configs (bnb, GPTQModel, AWQ, HQQ, torchao, quanto, FP8) | safetensors | LLM, VLM | NVIDIA, Intel, AMD partial | MVP |
| llm-compressor | compressed-tensors (FP8, INT8, W4A16, NVFP4) | LLM | vLLM, SGLang | MVP |
| `optimum-cli export openvino` (NNCF) | OpenVINO IR | LLM, vision, embeddings | Intel CPU / iGPU / NPU | MVP |
| MLX `mlx_lm.convert -q`; mflux | MLX | LLM, diffusion | Apple | MVP |
| diffusers `PipelineQuantizationConfig` (bnb, torchao, quanto, GGUF, layerwise fp8, group offload) | safetensors / GGUF | diffusion | NVIDIA | MVP |
| whisper.cpp quantize; faster-whisper / CTranslate2 | GGML; CT2 int8 | audio | CPU, NVIDIA, Apple | MVP |
| sentence-transformers `export_*_quantized_*`, `quantize_embeddings`, model2vec | ONNX / OpenVINO int8; static vectors | embeddings | CPU | MVP |
| NVIDIA TensorRT Model Optimizer | FP8 / NVFP4, TRT-LLM engines | LLM, diffusion | NVIDIA | P2 |
| Intel AutoRound; AMD Quark | gptq / awq / GGUF; ONNX | LLM | Intel; AMD | P2 |
| Nunchaku (SVDQuant); stable-diffusion.cpp; ComfyUI-GGUF | INT4 / NVFP4; GGUF | diffusion | NVIDIA Turing+ / Blackwell; any | P2 |
| torchao `quantize_` / QAT; ONNX Runtime quantization; Ultralytics export | int8 / int4 / fp8; ONNX / TensorRT / OpenVINO / CoreML / TFLite | vision | many | P2 |
| coremltools; Olive; ExecuTorch; Qualcomm AI Hub; LiteRT | mobile formats | all | Apple ANE, Windows NPU, Android / iOS | mobile phase |

**Serve recipes (MVP):** llama.cpp `llama-server`, Ollama `Modelfile` + `ollama create`, vLLM `vllm serve` with quant flags, `mlx_lm.server`.

## Design

- Bundled recipes load from `data/recipes/`; third-party recipe packs register via entry point `rightsize.recipes` pointing at a folder.
- Template renderer: a minimal `{{ var }}` / `{% if %}` subset implemented in-house (no jinja2 in core); jinja2 optional under `[dev]` for validating parity.
- `render()` validates inputs against the recipe's `inputs`, fills defaults from the `PlanStep` (quant, paths, device), and returns `RenderedStep{kind: command|config, text, install_line, notes, source_doc_url}`.
- `docs.py` writes `docs/frameworks/<framework>.md` from the registry: what it is, when to pick it, hardware, install, recipes with example renders. Run in CI; diff must be clean.

## MVP scope

The 14 MVP recipes above plus 4 serve recipes; docs generation; entry point discovery.

## Follow-up research

- Pin and verify exact CLI flags per toolkit version; record `version_tested`.
- Unsloth NVFP4 / FP8 export **(verify)**; `mlx_lm` mixed-precision / AWQ recipes **(verify)**.
- Which recipes need a preceding `require` step (imatrix, calibration data for AWQ / GPTQ).

## Tests

- Every recipe renders with sample inputs; rendered text snapshot-tested.
- Schema validation of every recipe file; `source_doc_url` and `version_tested` required.
- Nightly workflow (`.github/workflows/recipes-nightly.yml`) installs each toolkit in a matrix and runs the rendered command with `--help` or a dry-run flag.

## Reuse

- Notebook repo `data_pipeline/scripts/serve_llama.sh` and `serve_vllm.sh`: first serve recipes; docker-compose `local-ai*` profiles inform the vLLM flags.

## TODO

- [ ] `data/schema/recipe.schema.json`; `FrameworkInfo`, `RenderedStep` types
- [ ] Minimal template renderer with tests
- [ ] Loader for bundled recipes + entry-point discovery
- [ ] 14 MVP quantize / fine-tune recipes + 4 serve recipes, each with `source_doc_url` and `version_tested`
- [ ] `render()` and `Plan.render()`
- [ ] `docs.py` generator; CI diff check
- [ ] Nightly dry-run workflow
- [ ] Verify the two **(verify)** items
