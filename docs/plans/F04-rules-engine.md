# F4. Rules and recommendation engine

Package: `rightsize.rules`. Phase 1. Depends on: F1, F2, F3, data seed. Feeds: F5, F6.

## Goal

Turn candidates plus fit results into a ranked, explained list of `Plan`s. Enforce hard gates (what cannot run where). Support a pinned framework and a quality floor. Serve both flows: hardware-first (`recommend`) and model-first (`recommend_for_model`).

## Why it beats what exists

Hub badges, LM Studio and GPUStack answer yes / no for the model you picked; CanIRun.ai and canirunllm rank alternatives by fit but carry no per-quant quality penalty and cite no rules. We return the **next best thing that fits** with the quality penalty stated, and every rule that fired cites its source URL. GPUStack's backend x OS x GPU matrix and the HF / vLLM quantization matrices become open, testable rules.

## Public API

```python
from rightsize.rules import recommend, recommend_for_model, load_rules

recommend(task: str, family: Family, finetune_device: Device | None, target_device: Device,
          mode: Mode, pinned_framework: str | None = None, quality_floor: float | None = None,
          ctx: int = 8192, top_k: int = 5) -> list[Plan]

recommend_for_model(model: ModelRef | str, finetune_device: Device | None, target_device: Device,
                    mode: Mode, pinned_framework: str | None = None, ctx: int = 8192) -> list[Plan]
```

## Data

Rules are YAML in `data/rules/*.yaml`:

```yaml
- id: fp8_needs_ada_or_newer
  applies_to: { quant.method: fp8, stage: [quantize, serve] }
  condition: "device.vendor == 'nvidia' and device.compute_capability < 8.9"
  effect: block
  message: "FP8 W8A8 needs Ada, Hopper or Blackwell (or AMD MI300)."
  source_url: https://docs.vllm.ai/en/latest/features/quantization/index.html
  test:
    fires:   { device: { vendor: nvidia, compute_capability: 8.6 }, quant: { method: fp8 } }
    silent:  { device: { vendor: nvidia, compute_capability: 8.9 }, quant: { method: fp8 } }
```

Seed rules (from the research, each with its URL):

- QLoRA fine-tuning on NVIDIA / Intel = bitsandbytes (the established path). Apple = MLX `mlx_lm.lora` or Unsloth. AMD = Axolotl / LLaMA-Factory ROCm.
- FP8 W8A8 needs Ada / Hopper / Blackwell or AMD MI. AWQ does not run on AMD in vLLM. GPTQ runs Volta+.
- Sub-2-bit GGUF collapses on agentic / tool-use tasks (Unsloth Dynamic 3.0 docs): floor agentic tasks at Q4.
- Nunchaku INT4 = Turing / Ampere / Ada; NVFP4 = Blackwell; Volta and Hopper unsupported. torchao NVFP4 / MXFP8 = Blackwell. Core ML palettization = iOS 17+ / macOS 14+.
- i-quants need an imatrix and are slow on weak CPUs; k-quants win at 4-bit and above.
- Deprecated: torchtune (wound down 2025), HF AutoTrain (unmaintained), Unsloth Dynamic 2.0 (superseded by 3.0).
- Unsloth: Python 3.11–3.13, CUDA capability 7.0+, CUDA 12.4+ (12.8+ for Blackwell).
- GPUStack-style backend compatibility: which runtime runs on which OS x vendor (vLLM Linux-only; MLX macOS-only; llama.cpp everywhere; SGLang Linux NVIDIA / AMD).
- Speed floor: block plans under 5 tok/s for chat tasks unless the user opts in.
- Small models: under ~3B parameters, penalize 4-bit variants when an 8- or 16-bit variant fits (nf4 raises energy 25–56% at that scale; Optimum guide / EcoCompute). Weight-only for single-user local inference; W8A8 for multi-user serving. NVFP4 Blackwell-only; Marlin MXFP4 not on Turing. See [02-quantization-concepts.md](02-quantization-concepts.md) section 4 for the full list.

Quality data (`data/quality/`): base-model quality from Artificial Analysis / Arena / llm-stats (**not** the archived Open LLM Leaderboard); quant penalty curves from Unsloth Dynamic 3.0 KL / Divergence-300 tables; non-LLM families sizing-only in MVP.

## Design

- **Condition evaluator**: a small safe expression language over `device.*`, `model.*`, `quant.*`, `runtime.*`, `task`, `stage` (comparison, boolean ops, `in`, attribute access). No `eval`. Parsed once at load.
- **Effects**: `block` removes the candidate; `penalize: 0.x` multiplies the score; `require` adds a step (e.g. imatrix before i-quant); `note` adds to `Plan.trace` only.
- **Candidate generation**: hardware-first enumerates curated models for the family and task x quant ladder x runtimes supported on the target; model-first enumerates the quant ladder x runtimes for one model. Both call F3 for each `(model, quant, runtime, device, stage)`.
- **Ranking**: `score = base_quality x (1 - quant_penalty)`, tie-break by speed, then simplicity (fewer install steps). Quality floor filters before ranking. Pinned framework restricts recipes, not candidates.
- **Explainability**: `Plan.trace` lists every fired rule as `"{id}: {message} ({source_url})"`, plus the formula ids used.
- **Budget-to-bits candidates**: besides the uniform quant ladder, the generator adds a mixed-precision candidate when the budget lands between two formats: ModelOpt AutoQuantize (`effective_bits`) on NVIDIA, UD-style GGUF mixes on llama.cpp. Ranked like any other candidate; its quality penalty is interpolated between the neighbouring uniform formats and marked lower confidence.

## MVP scope

About 40 rules; LLM quant penalty from Unsloth KL tables; non-LLM families ranked by size only with a note that quality is unranked.

## Follow-up research

- Run `llama-perplexity --kl-divergence` on a fixed model set to build our own penalty curve per quant type and size class.
- Diffusion quality under quantization (essentially unquantified publicly).
- Licence of quality sources for redistribution.

## Tests

- Every rule ships its own `test.fires` / `test.silent` cases; a schema check fails any rule without them.
- Golden ranking cases in `tests/golden/rules/`: RTX 3060 12 GB coding chat; M4 Max 64 GB agentic; T4 Colab QLoRA of a 7B; RTX 4090 FLUX; CPU-only 32 GB STT.

## Reuse

- Notebook repo `data_pipeline/endpoints.py`: `default_extras` and thinking-switch families become `note` rules per family.

## TODO

- [ ] `data/schema/rule.schema.json`; loader; test-case requirement enforced
- [ ] Safe condition evaluator with parser tests
- [ ] Seed 40 rules with sources and tests
- [ ] Candidate generation for both flows
- [ ] Quant penalty table from Unsloth KL data; base quality table with sources
- [ ] Ranking, quality floor, pinned framework
- [ ] `Plan.trace` population; formula ids included
- [ ] Golden ranking cases
- [ ] Wire `rightsize.recommend` / `recommend_for_model` and the CLI
