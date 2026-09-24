# 00. Competitor landscape and how Rightsize is better

Researched 2026-09-24. Re-check quarterly; this space moves fast.

## The gap in one sentence

Nobody joins `(fine-tune hardware, target device, task) -> ranked (model, quant, framework) + runnable recipe`. Every existing tool does one slice: a VRAM number for one LLM, a fit badge on one file, a training job you already configured, or compression of a model you already chose.

## Landscape

| Category | Tools | What they do well | What they lack |
|---|---|---|---|
| Web VRAM calculators | [vram-calc.com](https://vram-calc.com/) (will-it-fit, quant comparison, KV and max-context, tok/s from bandwidth, Mac calculator, MLA / sliding-window / hybrid attention math), [APXML](https://apxml.com/tools/vram-calculator) (graph-capture vs eager, chunked prefill budgeting), [NyxKrage HF Space](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator), [LLMHardware.io](https://llmhardware.io/) (7.4k-model catalog, GPU tier guides), convly, bytecalculators, devtk, rapidtoolset, promptquorum | Fast inference-memory answers for one LLM; some model speed | LLM-only; no fine-tuning; no framework or recipe output; no API/CLI/SDK/MCP; no ranking across models; nothing for diffusion, audio, vision, embeddings; formulas and tables closed |
| Diffusion calculators | [Will It Run AI](https://willitrunai.com/calculator/diffusion) (image and video models, FP16/FP8/NF4, ControlNet and LoRA add-ons, offload flag, generation time, max resolution), DataZier | Component breakdown: weights, VAE, text encoder, activations, overhead | No fine-tuning; no toolkit recipe (which tool produces the FP8 or GGUF?); closed formulas; no API |
| Fine-tune memory estimators | [Red Hat Training Hub `memory_estimator`](https://developers.redhat.com/articles/2026/03/04/estimate-gpu-memory-llm-fine-tuning-red-hat-ai) (SFT, LoRA, QLoRA, OSFT; weights + grads + optimizer + activations + LoRA matrices; 1.3x overhead multiplier), [gpu_poor](https://github.com/RahulSChand/gpu_poor), [AndreaPi/llm-memory-calculator](https://github.com/AndreaPi/llm-memory-calculator), [mshojaei77/vram-calculator](https://github.com/mshojaei77/vram-calculator), [accelerate estimate-memory](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/model_size_estimator.md) (meta-device load, "within a few %" for load, +20% for inference) | Correct component model for training memory | Single model; no target-device reasoning; no quant or framework choice; no ranking; Red Hat's is tied to its training stack; accelerate needs transformers installed |
| GGUF-specific | [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) (remote header parse, memory and max tok/s, layer/row split across devices, offload count), [GPUStack compatibility check](https://docs.gpustack.ai/latest/user-guide/compatibility-check/) (backend x OS x GPU x architecture; schedulability; `weights x 1.2 + framework footprint` for non-GGUF) | Best-in-class GGUF estimation and cluster placement | Checks the model you chose; no alternative suggested when it does not fit; no fine-tuning; Go, not importable from Python |
| Hub and runtime built-ins | [HF hardware compatibility panel](https://huggingface.co/docs/hub/en/hardware) (saved hardware -> per-GGUF / MLX-file fit badge), [LM Studio fit badges](https://localllm.in/blog/lm-studio-vram-requirements-for-local-llms) (green / yellow per quant at full offload), Ollama (nothing built in; community VRAM tables) | Zero-effort hint at download time | File-size heuristics only; ignore context length and KV cache; one repo at a time; nothing for fine-tuning or non-GGUF formats; no cross-model ranking |
| Fine-tuning UIs | [Unsloth Studio / Desktop](https://unsloth.ai/docs/new/studio) (4-step wizard: modality, method, dataset, hyperparams; live VRAM; GGUF / merged / LoRA export; NVIDIA 7.0+, AMD, Apple), [Kiln](https://docs.kiln.tech/) (zero-code, exports datasets to Unsloth / Axolotl), [Transformer Lab](https://lab.cloud/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (GUI, 100+ models) | Run the job with a good UI | Assume you already chose model and framework; no target-device planning; no cross-framework comparison; OOM discovered at runtime; Unsloth publishes a lookup table, not a calculator |
| Compression platforms | [Pruna](https://docs.pruna.ai/en/stable/compression.html) (quant, prune, distill, cache, compile, combinable), [Nexa SDK](https://sourceforge.net/projects/nexa-sdk.mirror/) (ONNX / GGML runtime, text, vision, ASR, TTS), [NetsPresso](https://netspresso.ai/) (edge NPUs, expert-led) | Compress and run a model you already chose | Model-first only; vendor runtime or paid service; no "what should I pick" |
| MCP servers | HF MCP, ollama-mcp-server, vram-mcp | Hub search, local model control | None answers "will this fit, at which quant, with which framework" |

## Where Rightsize wins

Each item is a product requirement tracked in a feature TODO.

| # | Requirement | Nobody else | Feature |
|---|---|---|---|
| 1 | Two-stage, two-direction planning: fine-tune box AND target device; hardware-first AND model-first | joins these | F3, F4 |
| 2 | Ranked alternatives, not a verdict: when a model does not fit, return the next best model / quant / runtime that does, with the quality penalty stated | GPUStack, HF panel, LM Studio stop at yes/no | F4 |
| 3 | Recipe output: every plan renders the commands or configs for the chosen toolkit | calculators end at a number | F5 |
| 4 | Five families behind one API: LLM/VLM, diffusion, audio, vision, embeddings | each competitor is LLM-only or diffusion-only | F3 |
| 5 | Programmable surface: Python SDK, CLI with `--json`, MCP server | web calculators are UI-only; gguf-parser is Go | F6 |
| 6 | Honest numbers: `confidence`, `formula_id`, source URL on every estimate; KV and context always modelled | HF / LM Studio use file size | F3, F4 |
| 7 | Speed is a verdict: tok/s or s/image from bandwidth; "fits at 3 tok/s" is flagged | only vram-calc and gguf-parser do speed, LLM only | F3 |
| 8 | Cloud fallback: cheapest GPU that fits and job cost when the box is too small | no calculator does this | F7 |
| 9 | Open, versioned data with provenance anyone can correct | competitors' tables are closed or scattered | F2, F4 |
| 10 | Self-correcting: opt-in predicted-vs-actual telemetry refits constants and publishes the dataset | nobody closes the loop | F9 |

## Adopt, do not re-invent

- oobabooga's fitted GGUF formula and gguf-parser's offload model -> F3 llama.cpp path.
- Red Hat's component breakdown for training memory -> F3 fine-tune estimator.
- vram-calc's MLA / sliding-window / hybrid attention handling -> F1 KV overrides, F3.
- Will It Run AI's diffusion component breakdown -> F3 diffusion estimator.
- HF `hardwareItems` as the zero-entry hardware source -> F2.
- GPUStack's backend x OS x GPU compatibility matrix -> F4 rules.
- accelerate's meta-device approach is what we avoid: we read headers instead so the core never needs transformers.

## Sources

vram-calc.com, apxml.com, llmhardware.io, willitrunai.com, developers.redhat.com (2026-03-04), github.com/gpustack/gguf-parser-go, docs.gpustack.ai, huggingface.co/docs/hub/en/hardware, unsloth.ai/docs/new/studio, docs.kiln.tech, github.com/huggingface/accelerate, github.com/RahulSChand/gpu_poor, docs.pruna.ai, netspresso.ai, localllm.in (LM Studio fit badges).
