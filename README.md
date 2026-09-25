# Rightsize

**Quantize and fit any model to your hardware.** Pick a model, pick your hardware, get a ranked plan with the quantization, runtime and framework that fit, plus the commands to make it happen.

> **Status: early.** `rightsize` 0.0.1 on PyPI is a placeholder that reserves the name. The `main` branch has the first working slice: read a model's facts from the Hub without downloading it, predict memory and speed for GGUF quants on your GPU, then convert, quantize and gate the result with llama.cpp. See [docs/plans](docs/plans/README.md) for the roadmap and each feature's plan.

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

## Try it

```bash
git clone https://github.com/the-anup-das/rightsize-ai && cd rightsize-ai
uv sync --group dev                       # core only: detect, estimate, recipes
rightsize detect                          # what machine is this?
rightsize estimate Qwen/Qwen3-4B --quant Q4_K_M --quant Q8_0   # no download, reads Hub headers

# To actually produce files: llama.cpp binaries in .tools/llama.cpp (see the guide) and
uv sync --group dev --extra llamacpp     # torch CPU + transformers for the conversion step
rightsize quantize Qwen/Qwen3-1.7B --quant Q4_K_M --imatrix --eval
```

`pip install rightsize` works too, but until the next release it installs the 0.0.1 placeholder.

## Roadmap

| Feature | Plan | Status |
|---|---|---|
| Competitor landscape | [00-competitors](docs/plans/00-competitors.md) | research done |
| F1 Model catalog | [F01](docs/plans/F01-model-catalog.md) | first slice: facts from Hub headers |
| F2 Hardware DB + detection | [F02](docs/plans/F02-hardware.md) | first slice: 16 presets, NVIDIA / Apple / CPU detection |
| F3 Fit engine | [F03](docs/plans/F03-fit-engine.md) | first slice: GGUF inference memory and speed |
| F4 Rules + ranking | [F04](docs/plans/F04-rules-engine.md) | planned |
| F5 Framework registry + recipes | [F05](docs/plans/F05-framework-registry.md) | first slice: five llama.cpp recipes |
| F6 SDK / CLI / MCP | [F06](docs/plans/F06-surfaces.md) | CLI: detect, estimate, frameworks, quantize |
| F7 Cloud fallback | [F07](docs/plans/F07-cloud-fallback.md) | planned |
| F8 Execution + eval gate | [F08](docs/plans/F08-execution-eval.md) | first slice: llama.cpp adapter with KL-divergence gate |
| F9 Calibration loop | [F09](docs/plans/F09-calibration.md) | phase 2 |
| F10 Cloud provider connectors | [F10](docs/plans/F10-cloud-connectors.md) | phase 3 |

## Where this project is honestly at

I'm one developer building this in my spare time, and I want to be upfront about what that means before you rely on a number it gives you.

**The hardware I have.** Day to day I work on a Windows 11 desktop with an RTX 4070 Ti SUPER (16 GB) and 64 GB of RAM. I also have a Linux machine, an OpenMediaVault 8 server that is CPU-only, and a Mac. There is no cloud budget. So the paths that get exercised most are GGUF quantization and evaluation of models up to roughly 8B on a single NVIDIA card, plus whatever I can run on CPU. The Linux, NAS and Apple Silicon paths get real runs as I get to them, not on every change.

**What that means for you.**

- Estimates for NVIDIA consumer cards and CPU-only boxes are checked against real runs on my machines. The `runs/*/manifest.json` files record predicted versus measured, and I keep the constants honest from those.
- Estimates for AMD, Intel, Apple, multi-GPU and data-center GPUs are built from documentation and vendor spec sheets. Every preset carries the URL it came from so you can check it, and the estimate carries a lower confidence on purpose. Treat them as a starting point until someone with that hardware confirms them.
- Fine-tuning estimates, the rules engine, the MCP server, the cloud fallback and the calibration loop are not built yet. Each has a plan with a TODO list in `docs/plans/`, and the roadmap table above says which is which.

**How you can help, in five minutes.** If a number is off on your hardware, that is the single most useful thing you can send me.

> **[Open an issue](https://github.com/the-anup-das/rightsize-ai/issues/new)** with the output of `rightsize detect` and `rightsize estimate <model>`, or attach a `runs/<run>/manifest.json` from a real run. If you know the right bandwidth or memory figure for a device, the presets are a YAML file; a one-line PR with the source URL is perfect.

When better hardware or cloud access becomes affordable, the "from documentation" list shrinks and the "checked" list grows. Until then I'd rather tell you exactly what has been tested than let a clean-looking table imply more than it should.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Adding a framework means adding a recipe file, not code.

## License

Apache-2.0. See [LICENSE](LICENSE).
