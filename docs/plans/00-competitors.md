# 00. Competitor landscape and how Rightsize is better

Researched 2026-09-24, revised the same day after a second, step-by-step pass (discovery per category, then verification of each close competitor on its primary page). Re-check quarterly; this space moves fast. Items marked **(verify)** could not be confirmed from a primary source.

## 1. The gap in one sentence

Nobody joins `(fine-tune hardware, target device, task) -> ranked (model, quant, runtime, framework) + runnable recipe` across model families. Several tools now do the **LLM inference fit** slice well, including weights plus KV cache plus overhead, quant suggestions and speed estimates. Nobody does fine-tuning and target together, nobody covers diffusion / audio / vision / embeddings behind the same engine, and nobody ends at the command that produces the artifact.

## 2. Corrections to the first pass

The first version of this document made three claims that verification did not support:

| Claim | Reality |
|---|---|
| "No MCP server answers will this fit" | [local-ai-mcp](https://github.com/TMHSDigital/local-ai-mcp) has `fit_check` (weights + KV vs free VRAM / RAM) and `suggest_model` (ranked by task and fit) across Ollama / LM Studio / llama.cpp; [gpu-container](https://github.com/mcp-tool-shop-org/gpu-container) plans VRAM / RAM / NVMe placement across llama.cpp, vLLM, TensorRT-LLM and returns a measured tok/s receipt. Both are LLM-inference only. |
| "LM Studio uses file size only" | LM Studio's estimator (`lms load --estimate-only`, Discover-tab badge) reports estimated GPU and total memory and accounts for flash attention and vision models. The Hugging Face hub panel is still file-size based. |
| "Every competitor is a single yes/no" | [CanIRun.ai](https://www.canirun.ai/), [canirunllm](https://pypi.org/project/canirunllm/), [modelfit.io](https://modelfit.io/), [localaimaster model recommender](https://localaimaster.com/tools/model-recommender) and [LLM Configurator](https://llmconfigurator.com/en/analyzer) all rank models or quants by fit for a detected or chosen machine, some with tok/s. |

## 3. Landscape by category

### 3.1 Local "can I run it" checkers and recommenders (closest to hardware-first, inference only)

| Tool | Form | Detection | What it models | Output | Families | Missing |
|---|---|---|---|---|---|---|
| [CanIRun.ai](https://www.canirun.ai/) | web, free | browser (GPU / Mac, VRAM, bandwidth, RAM) | catalog of published requirements | grades S–F, quant options Q2–F16, tok/s, ctx, MoE active params, tier lists | LLM, VLM, image gen, video gen | fine-tuning, audio, API, recipes |
| [canirunllm](https://pypi.org/project/canirunllm/) 0.1.4 (2026-09-21, MIT) | Python CLI / dashboard | real CPU, RAM, NVIDIA VRAM | weights x quant + KV by ctx + overhead + margin; memory strategy (single GPU / multi / offload); runtime compatibility | verdict with confidence, quant + runtime suggestion (llama.cpp, Ollama), labelled speed estimates; sources documented | LLM (curated registry) | fine-tuning, non-LLM, AMD / Apple detection, recipes, MCP |
| [modelfit.io](https://modelfit.io/) | web, free | manual (memory, chip) | ~0.6 GB / B at Q4 + overhead; tok/s from bandwidth | one recommended model + count of fits; 143 models, 106 via Ollama | LLM | fine-tuning, quant ladder, API |
| [localaimaster model recommender](https://localaimaster.com/tools/model-recommender) / [AI model finder](https://localaimaster.com/tools/ai-model-finder) | web, free | preset VRAM tiers | Q4_K_M only; quality score 0–100 from benchmarks | ranked models, tok/s, Ollama commands | LLM (chat, coding, reasoning, RAG, vision, audio tags) | fine-tuning, quant ladder, API |
| [LLM Configurator analyzer](https://llmconfigurator.com/en/analyzer) | web, free | browser or dropdown | tiers "runs great / runs / won't fit"; Q4 / Q5 / Q8 / FP16 | tok/s, electricity cost, Ollama commands; separate fine-tuning calculator | LLM | API, non-LLM, recipes |
| [LLMHardware.io](https://llmhardware.io/) | web, free | manual | per-model VRAM at Q4 / Q8 | 7.4k-model catalog, GPU tier guides, tok/s | LLM mainly | fine-tuning, API |
| LocalClaw **(verify)** | desktop | scans GPU / RAM | unknown | recommends a model | LLM | unverified |

### 3.2 VRAM and memory calculators (one model at a time)

| Tool | Scope | Notable | Missing |
|---|---|---|---|
| [vram-calc.com](https://vram-calc.com/) | inference | will-it-fit, quant comparison, KV and max-context, tok/s from bandwidth, Mac calculator, MLA / sliding-window / hybrid attention | fine-tuning, multi-GPU, non-LLM, API |
| [APXML](https://apxml.com/tools/vram-calculator) **(verify, 403)** | inference and fine-tune modes reported | graph-capture vs eager, chunked prefill budgeting | API, non-LLM |
| [NyxKrage HF Space](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator) | inference | Hub model id in, GPU pick, ctx, quant | fine-tuning, speed |
| [gekro LoRA / QLoRA memory calculator](https://gekro.com/apps/lora-memory-calculator/) | fine-tuning | weights / optimizer / gradients / activations breakdown, GPU compatibility table | inference, target device, recipes |
| [accelerate estimate-memory](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/model_size_estimator.md) | load memory | meta-device load; "within a few %" for load, +20% for inference | needs transformers; no KV; no fit |
| [Red Hat Training Hub memory_estimator](https://developers.redhat.com/articles/2026/03/04/estimate-gpu-memory-llm-fine-tuning-red-hat-ai) | fine-tuning | SFT / LoRA / QLoRA / OSFT; weights + grads + optimizer + activations + LoRA matrices; 1.3x overhead | tied to Training Hub stack; no target device |
| [gpu_poor](https://github.com/RahulSChand/gpu_poor), [llm-memory-calculator](https://github.com/AndreaPi/llm-memory-calculator), [mshojaei77/vram-calculator](https://github.com/mshojaei77/vram-calculator) | inference + training | early open formulas; gpu_poor has GGML / bnb / QLoRA | stale, no hardware DB, no ranking |
| Diffusion: [Will It Run AI](https://willitrunai.com/calculator/diffusion), DataZier | image + video | weights / VAE / text-encoder / activation breakdown, FP16 / FP8 / NF4, ControlNet, LoRA, offload flag, generation time, max resolution | fine-tuning, toolkit recipe, API |

### 3.3 Analytic performance estimators (research-grade, Python)

| Tool | Estimates | Hardware | Quant | Notes |
|---|---|---|---|---|
| [llm-analysis](https://github.com/cli99/llm-analysis) (Apache-2.0, 494 stars) | memory + latency, training and inference, TP / PP / SP / EP / DP, ZeRO, recompute | JSON GPU configs (A100 etc.) | 32 / 16 / 8 / 4 bit, FP8 planned | lower-bound estimates; needs transformers |
| [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) (MIT, 681 stars) | per-layer memory and roofline latency, inference; HF models and DiT | small hardware list | bitwidth setting | web UI + CLI |
| [GenZ-LLM-Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer) (MIT) | inference latency + memory; TP / PP; validated ~5.8% geomean error (paper) | GPU / CPU / accelerators | INT8 / FP8 / INT4 / FP4 / INT2 | Streamlit UI |
| [LLM-X](https://github.com/Sheikyon/LLM-X) (MIT) | inference memory from tensor-level analysis; claims 98.2% accuracy vs hf-mem 113% and accelerate 130% error | psutil + NVML on the local box | KV precision / compression settings | CLI-first |
| [LLM-para](https://github.com/dengls24/LLM-para) | FLOPs, memory, roofline; GQA, MoE, MLA; 19 models x 20+ platforms | table | yes | research |

### 3.4 GGUF-specific and placement planners

| Tool | What it does | Missing |
|---|---|---|
| [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) | remote header parse, memory and max tok/s, layer / row split across devices, offload count | Go; checks a chosen file; no fine-tuning |
| [GPUStack compatibility check](https://docs.gpustack.ai/latest/user-guide/compatibility-check/) | backend x OS x GPU x architecture; schedulability; `weights x 1.2 + footprint` for non-GGUF | no alternative when it fails |
| [gpu-container](https://github.com/mcp-tool-shop-org/gpu-container) (MIT, Python + MCP) | profiles VRAM / RAM / NVMe / CUDA; dense vs MoE, largest layer, KV by ctx; placement plan for llama.cpp, vLLM, Accelerate, TensorRT-LLM, DeepSpeed offload; measured receipt | no alternative models or quants; inference only; single GPU |

### 3.5 Built into hubs and local runners (zero-effort fit hints)

| Where | Feature | Method | Limits |
|---|---|---|---|
| [Hugging Face hub](https://huggingface.co/docs/hub/en/hardware) | hardware compatibility panel per GGUF / MLX file (GGUF since 2025, MLX 2026-01-22); "filter models by hardware" (2026-06-30); `hardwareItems` in profile; Local Apps hand-off | file size vs declared RAM / VRAM | ignores ctx and KV; one repo at a time; no fine-tuning |
| LM Studio | "Likely Fit" badge in Discover; `lms load --estimate-only` prints estimated GPU / total memory; accounts for flash attention and vision models; partial offload control | estimator | one model at a time; no alternatives; no fine-tuning |
| [Jan](https://www.jan.ai/docs/desktop/manage-models) | "Fits / May be slow / Won't fit" pill per model, no download needed | own signal (undocumented) | LLM only |
| GPT4All | curated 10–15 models labelled by RAM need | static | tiny catalog |
| Ollama | none built in; community VRAM tables | n/a | n/a |
| [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/what-is-foundry-local) | detects CPU / GPU / NPU and downloads the best pre-compiled ONNX variant of a catalog model; Windows, macOS, Linux; chat and Whisper | curated catalog, variant per execution provider | no memory estimate, no "which size fits", catalog only |
| Intel AI Playground, AMD Lemonade / GAIA, Nexa | vendor local runners with model pickers for their silicon | curated lists | single-vendor |

### 3.6 Fine-tuning UIs and frameworks

| Tool | Hardware awareness | Missing |
|---|---|---|
| [Unsloth Studio / Desktop](https://unsloth.ai/docs/new/studio) (Apache-2.0 core, AGPL-3.0 UI) | wizard (modality, method, dataset, hyperparams); live loss / VRAM; text, diffusion, TTS, embeddings, multimodal; GGUF / safetensors export; NVIDIA 7.0+, AMD, Apple | no pre-run fit estimate found in docs **(verify)**; no target-device planning; publishes a [requirements table](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements), not a calculator |
| [Kiln](https://docs.kiln.tech/), [Transformer Lab](https://lab.cloud/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), Axolotl, Oumi, H2O LLM Studio | run the job; no pre-flight VRAM check found | model and framework already chosen; OOM discovered at runtime |

### 3.7 Compression platforms (model-first, you already chose the model)

| Tool | Notable | Missing |
|---|---|---|
| [Pruna](https://docs.pruna.ai/en/stable/compression.html) | `SmashConfig` automatic compression search with target metrics (memory on disk, memory at inference, first-load) across quant / prune / factorize / compile; hqq, gptq, llm_int8, higgs (pro) | model-first only; no hardware-to-model ranking; parts paid |
| Nexa SDK, NetsPresso, Edge Impulse (Qualcomm), Latent AI, Embedl | compress and deploy to edge; NetsPresso expert-led; Edge Impulse tied to Qualcomm AI Hub | vendor runtimes or services |

### 3.8 Edge and mobile: measured numbers per device (data, not competitors)

| Source | What it gives |
|---|---|
| [Qualcomm AI Hub](https://aihub.qualcomm.com/) | pre-optimized models; profiling on 50+ real Snapdragon devices in the cloud: latency, memory, power, compute-unit utilization |
| [Google AI Edge Portal](https://ai.google.dev/edge/ai-edge-portal) | benchmark GenAI models on 120+ Android device types (30+ with NPU): init time, prefill, decode, peak memory |
| Ultralytics [benchmarks](https://www.ultralytics.com/benchmarks) and `benchmark` mode | YOLO latency / mAP per export format and device |
| [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | embedding models with parameters and memory usage (FP32) columns |

### 3.9 Quality-of-quantization data (feeds the ranking, F4)

| Source | What it gives | Access |
|---|---|---|
| [Intel Low-bit Quantized Open LLM Leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard) | quantized LLMs searchable by algorithm (AutoRound, GPTQ, AWQ, bnb, GGUF), compute / weight dtype, size; accuracy per config | HF Space |
| [LocalBench GGUF quality benchmark](https://localbench.substack.com/p/gguf-benchmark-methodology) | KL divergence + top-1 agreement vs BF16 over ~250k tokens (coding, chat, tool calling, science, non-Latin, long docs); compares uploaders (unsloth, bartowski, lmstudio-community, ggml-org, AesSedai) | blog posts; raw data not published **(verify)** |
| Unsloth Dynamic 3.0 KL / Divergence-300 tables; [smcleod KL method](https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/); [Artefact2 gist](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) (2024) | per-quant quality curves | self-published |

### 3.10 Speed data and cloud prices (feeds F3, F7, F9)

| Source | What it gives | Access |
|---|---|---|
| [LocalScore](https://www.localscore.ai/about) (Mozilla Builders, on llamafile) | prompt-processing tok/s, generation tok/s, TTFT across 8 scenarios; public database of submitted results by CPU / GPU / Apple | website; API / dump not documented **(verify)** |
| [GetDeploying](https://getdeploying.com/) | 116 providers, 5,002 GPU prices, refresh every 14 minutes; "GPU Prices API" | API listed, terms **(verify)** |
| [ComputePrices](https://computeprices.com/) | 64 providers, hourly GPU and LLM API prices, JSON API | API |
| Shadeform, RunPod GraphQL `gpuTypes`, Apify trackers | multi-cloud launch API; live prices | API |

## 4. Feature matrix: Rightsize vs the closest eight

Legend: Y yes, P partial, N no.

| Capability | Rightsize (target) | canirunllm | CanIRun.ai | LM Studio | HF hub panel | gpu-container | local-ai-mcp | Unsloth Studio | Pruna |
|---|---|---|---|---|---|---|---|---|---|
| Hardware-first ranking of models | Y | Y | Y | N | P (filter) | N | Y | N | N |
| Model-first: what hardware do I need | Y | P | P | Y | P | Y | P | N | N |
| Fine-tune box AND target device in one plan | **Y** | N | N | N | N | N | N | N | N |
| Fine-tuning memory (full / LoRA / QLoRA) | Y | N | N | N | N | N | N | P (live only) | N |
| KV cache and context modelled | Y | Y | P | Y | N | Y | Y | n/a | n/a |
| Speed verdict (tok/s, s/image) | Y | Y (labelled) | Y | N | N | Y (measured) | N | N | N |
| Diffusion | Y | N | Y (image, video) | N | N | N | N | Y (train) | Y |
| Audio (STT / TTS) | Y | N | N | N | N | N | N | Y (TTS) | Y |
| Vision, embeddings | Y | N | N | N | N | N | N | Y (embed) | Y |
| Quant ladder with quality penalty | Y | P | P | N | N | N | N | N | P |
| Renders toolkit commands / configs | **Y** | P (Ollama pull) | N | N | N | P (placement) | N | runs it | runs it |
| Python SDK | Y | Y | N | N | N | Y | N | Y | Y |
| CLI with JSON | Y | Y | N | Y | N | Y | N | N | N |
| MCP server | Y | N | N | N | N | Y | Y | N | N |
| Open data with provenance | Y | Y (SOURCES.md) | N | N | N | N | N | N | N |
| Cloud rental fallback | **Y** | N | N | N | N | N | N | N | N |
| Predicted-vs-actual calibration loop | **Y** | N | N | N | N | P (receipt) | N | N | N |
| Detects local hardware | Y | Y (NVIDIA) | Y (browser) | Y | Y (profile) | Y | Y | Y | N |

Bold Y = nobody else has it.

## 5. Where Rightsize wins (revised)

| # | Requirement | Status of the field | Feature |
|---|---|---|---|
| 1 | Fine-tune box AND target device, hardware-first AND model-first, in one plan | nobody joins the two stages | F3, F4 |
| 2 | Five families behind one engine and one `Plan` type | CanIRun.ai covers LLM + image + video; Unsloth Studio trains several; nobody plans across all five | F3 |
| 3 | Recipe output: the plan ends at the command or config for the chosen toolkit | placement planners and runners exist; no advisor renders recipes across 20+ toolkits | F5 |
| 4 | Ranked alternatives with a stated quality penalty when the first choice does not fit | recommenders rank by fit; none carry a per-quant quality penalty with sources | F4 |
| 5 | Honest numbers: `confidence`, `formula_id`, source URL on every estimate and every rule | canirunllm labels confidence and cites sources for its registry; nobody cites rules | F3, F4 |
| 6 | Cloud rental fallback with job cost | nobody | F7 |
| 7 | Open, versioned data package anyone can correct (hardware, quants, rules, quality) | canirunllm documents sources; nobody ships data as a separately versioned package | F2, F4 |
| 8 | Self-correcting: opt-in predicted-vs-measured telemetry refits constants; public dataset | gpu-container measures its own plan; LocalScore collects speed only; nobody closes the loop on predictions | F9 |
| 9 | Agent-callable planner with fine-tuning and non-LLM scope | local-ai-mcp and gpu-container are MCP, inference-only, LLM-only | F6 |
| 10 | Core with no ML dependencies, importable in under 150 ms | llm-analysis needs transformers; LLM-X needs NVML; most others are web-only | all |

## 6. Threats to watch

- **canirunllm** is the same shape as our LLM inference slice, in Python, MIT, actively released. If it adds fine-tuning and non-LLM families it becomes the direct competitor. Consider collaboration or ingesting its registry with attribution.
- **CanIRun.ai** owns the browser-detection UX for the platform phase; the platform must offer something it does not (fine-tune planning, recipes, shareable plans).
- **LM Studio and the Hugging Face hub** ship fit hints where users already are. Rightsize must win on the questions they cannot answer, not on the badge.
- **Unsloth Studio** could add a pre-flight estimator and target-device export in one release. Our fine-tune floors come from their table; keep the relationship constructive (recipes for Unsloth first).
- **Foundry Local** shows the "auto-pick the variant for the hardware" idea shipping from a platform vendor, for a curated catalog. Rightsize's answer is open catalog, open data, any runtime.

## 7. Adopt, do not re-invent

- oobabooga's fitted GGUF formula and gguf-parser's offload split -> F3 llama.cpp path.
- LLM-X's tensor-level accuracy approach and its published error comparison -> F3 validation method.
- Red Hat's component breakdown and gekro's per-component view for training memory -> F3 fine-tune estimator.
- vram-calc's MLA / sliding-window / hybrid attention handling -> F1 KV overrides.
- Will It Run AI's diffusion component breakdown -> F3 diffusion estimator.
- canirunllm's memory-strategy classification (single GPU / multi / offload) and runtime compatibility check -> F3 / F4.
- GPUStack's backend x OS x GPU matrix -> F4 rules.
- Intel's low-bit leaderboard and LocalBench KL results -> F4 quant penalty data (check licences).
- LocalScore's public speed database -> F3 speed calibration and F9 seed.
- Qualcomm AI Hub and Google AI Edge Portal measured numbers -> mobile phase tables.
- HF `hardwareItems` -> F2 zero-entry hardware; GetDeploying and ComputePrices -> F7.
- accelerate's meta-device approach is what we avoid: read headers so the core never needs transformers.

## 8. Research log

| Step | What | Result |
|---|---|---|
| 1 | Discovery: 8 searches across calculators, estimators, hubs, edge, quant toolkits, cloud prices, MCP | found 3.1, 3.3, 3.9, 3.10 categories missing from the first pass |
| 2 | Discovery: runners, Foundry Local, LocalScore, MTEB / Ultralytics, Pruna auto, edge portals, vendor apps, MCP servers | found Foundry Local, Jan labels, LocalScore, local-ai-mcp, gpu-container |
| 3 | Verification on primary pages: local-ai-mcp, gpu-container, canirunllm, CanIRun.ai, LLM-X, LocalScore, modelfit.io, localaimaster, LLM Configurator, Foundry Local, GetDeploying, LM Studio docs, Jan docs, HF hub changelog, GGUF quality sources | corrections in section 2 |
| 4 | Verification: llm-analysis, LLM-Viewer, GenZ, LocalBench methodology, Unsloth Studio docs; fine-tune and audio calculators; NVIDIA apps | gekro calculator found; no audio chooser exists; ChatRTX deprecated 2026-01-21 |
| blocked | APXML (HTTP 403), LLM Explorer (TLS certificate mismatch), llm-analysis PyPI page (JS only, used GitHub instead) | marked **(verify)** |
