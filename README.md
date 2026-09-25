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
rightsize estimate Qwen/Qwen3-4B --device "RTX 5080"           # or any of 259 catalogued devices
rightsize estimate Qwen/Qwen3-4B --device @your-hf-username    # or the hardware on your HF profile

# To actually produce files:
uv sync --group dev --extra llamacpp     # torch CPU + transformers for the conversion step
rightsize tools install llama.cpp        # pinned binaries + converter into .tools/llama.cpp (auto-picks CUDA/CPU/Metal)
rightsize quantize Qwen/Qwen3-1.7B --quant Q4_K_M --imatrix --eval
rightsize bench Qwen/Qwen3-1.7B           # measure this machine's real memory bandwidth
```

`pip install rightsize` works too, but until the next release it installs the 0.0.1 placeholder.


## Where the hardware numbers come from

A speed estimate is only worth as much as the bandwidth behind it, so this part is built to
be checkable rather than convenient.

The device catalogue is **ingested, not typed**: `scripts/ingest_hf_hardware.py` reads
Hugging Face's MIT-licensed SKU table (`huggingface.js`), pinned to a commit, for 259
accelerators across NVIDIA, AMD, Intel, Apple and Qualcomm, with their memory options,
compute capability and TFLOPS. Every record keeps the URL it came from.

That table has no memory bandwidth, and nothing redistributable does: Wikidata has no such
property, TechPowerUp serves a captcha and marks its pages `noindex,nofollow`, and the GPU
datasets on GitHub are all scrapes of it. So `scripts/ingest_bandwidth.py` reads the
**primary facts** from Wikipedia's GPU lists - bus width, memory type, memory clock - and
does the arithmetic itself:

```
bandwidth_gbps = bus_width_bits * memory_speed_gbps / 8
```

That is exact for GDDR and LPDDR parts, and it carries its own derivation: your card reads
`672.0 GB/s  256-bit GDDR6X at 21 Gbps`, not a number copied from a table. It also catches
errors - Wikipedia's own bandwidth column says 336 GB/s for the RTX 3060 12 GB where its
bus width and clock give the 360 NVIDIA publishes.

Where a vendor publishes the figure directly and the arithmetic cannot reach it (HBM parts,
Apple's unified memory, Intel Arc), it is curated from the vendor's own page with a link.
The ingest refuses to write if it disagrees by more than 2% with anything hand-checked.

**Roughly 200 of 259 devices have a bandwidth.** The rest say so instead of guessing, and
there are two ways to fill one in:

```bash
rightsize bench Qwen/Qwen3-1.7B        # measure it: runs llama-bench and solves the
                                       # speed model backwards for this machine
rightsize estimate ... --bandwidth 102 # or supply a figure you trust
```

Measuring is the better answer for laptop GPUs in particular, where NVIDIA publishes a bus
width but no bandwidth, because the memory speed is the laptop maker's choice - there is no
single correct number to look up. On this project's RTX 4070 Ti SUPER, a measured 362.7
tok/s derives 675 GB/s against a published 672, which says the decode reached the 70% of
peak the speed model assumes.

**Two units, on purpose.** Device memory is **GiB**, because that is what the vendor, the
box and `nvidia-smi` all say - a "16GB" card holds 16 GiB. Model and file sizes are
**decimal GB**, matching how Hugging Face lists them. `Device.memory_gb` converts, so the
fit engine only ever compares like with like: 16 GiB is 17.18 GB, and treating it as 16 made
every verdict on that card 7% pessimistic.

## Roadmap

| Feature | Plan | Status |
|---|---|---|
| Competitor landscape | [00-competitors](docs/plans/00-competitors.md) | research done |
| F1 Model catalog | [F01](docs/plans/F01-model-catalog.md) | first slice: facts from Hub headers |
| F2 Hardware DB + detection | [F02](docs/plans/F02-hardware.md) | 259 devices ingested, bandwidth for ~200, detection, `bench`, HF profile import |
| F3 Fit engine | [F03](docs/plans/F03-fit-engine.md) | first slice: GGUF inference memory and speed |
| F4 Rules + ranking | [F04](docs/plans/F04-rules-engine.md) | planned |
| F5 Framework registry + recipes | [F05](docs/plans/F05-framework-registry.md) | first slice: five llama.cpp recipes |
| F6 SDK / CLI / MCP | [F06](docs/plans/F06-surfaces.md) | CLI: detect, estimate, frameworks, quantize, bench, tools install. MCP not started |
| F7 Cloud fallback | [F07](docs/plans/F07-cloud-fallback.md) | planned |
| F8 Execution + eval gate | [F08](docs/plans/F08-execution-eval.md) | first slice: llama.cpp adapter with KL-divergence gate |
| F9 Calibration loop | [F09](docs/plans/F09-calibration.md) | phase 2; runs already record predicted vs measured, and `bench` checks the speed constant |
| F10 Cloud provider connectors | [F10](docs/plans/F10-cloud-connectors.md) | phase 3 |

## Where this project is at, and where it is going

I'm one developer building this in my spare time, and I'd rather tell you exactly what has been exercised than let a clean table imply more than it should.

**What is verified today.** My daily machine is a Windows 11 desktop with an RTX 4070 Ti SUPER (16 GB) and 64 GB of RAM, and I also have a Linux box, an OpenMediaVault 8 server (CPU only) and a Mac. GGUF quantization and evaluation of models up to roughly 8B on a single NVIDIA card is the path that runs on every change, with CPU-only runs alongside. Every real run writes a `runs/<run>/manifest.json` with predicted next to measured, and those numbers are what keep the estimator honest.

**What is next on the hardware side.** Linux and CPU-only runs on the NAS, then Apple Silicon on the Mac, are the next real-run targets. AMD, Intel, multi-GPU and data-center GPUs are modelled from vendor spec sheets for now: every preset links the page it came from, and the estimate carries a lower confidence on purpose, so you know which numbers to double-check. As people with that hardware send back measurements, those move into the verified column.

**What is next on the feature side.** The roadmap table above is the plan: fine-tuning estimators, the rules and ranking engine, the MCP server, the cloud fallback and the calibration loop each have a plan file with a TODO list in `docs/plans/`, in build order. The pieces that exist today are the foundation the rest is built on, and each new feature lands with tests and a notebook.

**How you can help, in five minutes.** If a number is off on your hardware, that is the single most useful thing you can send.

> **[Open an issue](https://github.com/the-anup-das/rightsize-ai/issues/new)** with the output of `rightsize detect` and `rightsize estimate <model>`, or attach a `runs/<run>/manifest.json` from a real run. If you know the right bandwidth or memory figure for a device, the presets are a YAML file; a one-line PR with the source URL is perfect.

There is no cloud budget behind this yet. When better hardware or cloud access becomes affordable, the verified column grows faster; until then, measurements from the community are how it grows.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Adding a framework means adding a recipe file, not code.

## License

Apache-2.0. See [LICENSE](LICENSE).
