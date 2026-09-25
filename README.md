# Rightsize

**Quantize and fit any model to your hardware.** Pick a model, pick your hardware, get a ranked plan with the quantization, runtime and framework that fit, plus the commands to make it happen.

> **Status: placeholder release 0.0.1.** The package name is reserved; the engine is being built feature by feature. See [docs/plans](docs/plans/README.md) for the roadmap and each feature's plan. Nothing here estimates anything yet.

## The problem

Running a model on your own hardware today means choosing a fine-tuning framework, a quantization toolkit, a format and a runtime by hand, then finding out by out-of-memory error whether the combination fits. Calculators tell you the VRAM for one LLM; fine-tuning UIs run a job that then OOMs; Hub badges compare file sizes. Nothing joins *fine-tune box + target device + task* into a ranked, runnable answer.

## What Rightsize will do

Two flows, one engine, five model families (LLM/VLM, diffusion, audio, vision, embeddings):

- **Hardware-first.** "I have an RTX 4070 to train on and want to run on a MacBook Air 16 GB for coding help." Rightsize returns ranked models with the quant and framework for each stage.
- **Model-first.** "I want Qwen3 14B." Rightsize returns what it needs to fine-tune (LoRA/QLoRA/full) and to run, the quant per runtime, and the cheapest cloud GPU if your box is too small.

Every answer carries a memory breakdown, a speed estimate, a confidence, the formula used, and the rules that fired with their source URLs. Every plan renders the actual commands for the chosen toolkit (llama.cpp, Unsloth, Axolotl, MLX, optimum, diffusers, whisper.cpp and more).

Surfaces: Python SDK, CLI with `--json`, MCP server so agents can call it.

## How it stays light

- The core depends on `httpx`, `pydantic` and `pyyaml` only. No torch, no transformers. Enforced in CI.
- Hardware tables, quant tables, rules and recipes are versioned data, not code.
- Frameworks are declarative recipes that render commands. Running them is an opt-in extra (`rightsize[unsloth]`, `rightsize[llamacpp]`, ...), loaded lazily.
- Model metadata is fetched on demand from safetensors and GGUF headers (a few KB) and cached.

## New to model formats?

Read [Choosing a model format](docs/guide/choosing-a-model-format.md): what GGUF is, how to read a quant name like Q4_K_M, what the alternatives are (safetensors, bnb, AWQ, GPTQ, FP8, NVFP4, EXL3, MLX, OpenVINO, ONNX, MLC, Core ML) and their pros and cons, and which hardware each one reaches.

## Install

```bash
pip install rightsize        # or: uv add rightsize
rightsize --version
```

## Roadmap

| Feature | Plan | Status |
|---|---|---|
| Competitor landscape | [00-competitors](docs/plans/00-competitors.md) | research done |
| F1 Model catalog | [F01](docs/plans/F01-model-catalog.md) | planned |
| F2 Hardware DB + detection | [F02](docs/plans/F02-hardware.md) | planned |
| F3 Fit engine | [F03](docs/plans/F03-fit-engine.md) | planned |
| F4 Rules + ranking | [F04](docs/plans/F04-rules-engine.md) | planned |
| F5 Framework registry + recipes | [F05](docs/plans/F05-framework-registry.md) | planned |
| F6 SDK / CLI / MCP | [F06](docs/plans/F06-surfaces.md) | placeholder CLI |
| F7 Cloud fallback | [F07](docs/plans/F07-cloud-fallback.md) | planned |
| F8 Execution + eval gate | [F08](docs/plans/F08-execution-eval.md) | phase 2 |
| F9 Calibration loop | [F09](docs/plans/F09-calibration.md) | phase 2 |
| F10 Cloud provider connectors | [F10](docs/plans/F10-cloud-connectors.md) | phase 3 |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Adding a framework means adding a recipe file, not code.

## License

Apache-2.0. See [LICENSE](LICENSE).
