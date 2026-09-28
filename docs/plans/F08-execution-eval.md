# F8. Execution adapters and evaluation gate

Package: `rightsize.execution`. Phase 2. Depends on: F5 (rendered recipes), F3 (predictions). Feeds: F9.

## Goal

Run a plan's Unsloth and llama.cpp steps in-process, then evaluate the quantized output against the base model (KL divergence, perplexity, task metrics) before declaring success. Record predicted vs actual memory and speed in a run manifest.

## Why it beats what exists

Unsloth Studio runs the job but does not gate on degradation. Rightsize measures the damage a quant did and refuses to call it done if the threshold is exceeded, and every run feeds the calibration loop.

## Public API

```python
from rightsize.execution import run, evaluate

run(step: RenderedStep, workdir: Path, adapter: str | None = None) -> RunManifest
evaluate(base: ModelRef, candidate: Path, suite: str = "default", thresholds: dict | None = None) -> EvalReport
```

Runners register via the `rightsize.runners` entry point. Toolkits are not extras of rightsize: each gets its own uv environment under `.tools/<framework>` when someone picks it, since they pin conflicting library versions. A missing toolkit raises `ToolkitMissing` with the install line.

## Design

- `Adapter` protocol: `supports(step) -> bool`, `run(step, workdir) -> RunManifest`. Imports of torch / unsloth happen inside `run`, never at module import.
- `RunManifest{plan_id, step, command, started, finished, predicted: FitResult, measured: {peak_vram_gb, ram_gb, tok_per_s}, artifacts, logs_path}`. Measurement via `nvidia-smi` polling, `torch.cuda.max_memory_allocated()` when available, Ollama `/api/ps` after load.
- Evaluation gate: KL divergence vs base on a fixed prompt set (`llama-perplexity --kl-divergence` for GGUF), perplexity delta, and task metrics from a small suite; thresholds per family in `data/quality/thresholds.yaml`.
- Streaming logs to the console and to `workdir/logs/`.

## MVP scope (phase 2)

Unsloth QLoRA adapter (fine-tune -> merged or GGUF), llama.cpp quantize adapter, evaluation gate for LLMs. Diffusion and audio execution later. **Local machine only** by decision (2026-09-25); the same `Adapter` protocol is reused by the cloud connectors in F10 (phase 3), so design it with provision / execute / teardown hooks even though the local adapter only implements execute.

## Follow-up research

- Reliable peak-memory measurement across runtimes without instrumenting them.
- Task-metric suites per family that are cheap enough to run after every quantization.

## Tests

- Adapters mocked in unit tests (command capture, manifest shape).
- One real-GPU integration test marked `slow`, skipped in CI.

## Reuse (notebook repo `C:\Users\awesome\Downloads\notebook`)

- `data_pipeline/scripts/finetune.py`: Unsloth QLoRA -> GGUF / merged with a per-family catalog and a run manifest. Basis for the Unsloth adapter.
- `data_pipeline/scripts/evaluate.py` and `data_pipeline/eval/thresholds.json`: the degradation gate and its thresholds.
- `data_pipeline/scripts/serve_llama.sh`, `serve_vllm.sh`: serve recipes (F5) used to measure the candidate.

## TODO

- [x] `Runner` protocol (the adapter), `RunManifest`, `RunStep`: `execution/runner.py` runs
      any plan with `rightsize run PLAN.json`. A generic runner covers every toolkit installed
      with pip: commands run with the toolkit's environment first on PATH, configs are written
      into the run directory and run with the recipe's `run` line, and what each recipe
      `writes` is checked and measured. llama.cpp has its own runner (its binaries, the model
      download before the converter, the calibration text). Checked end to end on a
      recommend plan for Qwen3-0.6B (convert 11 s, Q8_0 in 3 s, serve printed, not started).
      The gate still reports a dict rather than an `EvalReport` type
- [x] Entry-point registration and lazy loading: a plugin's runner registers under
      `rightsize.runners`; toolkits install on demand, each into its own uv environment under
      `.tools/<framework>` (`rightsize tools install | list | remove`), and a missing one
      raises `ToolkitMissing` with the install command rather than an ImportError
- [x] Unsloth runs end to end without an adapter of its own: the generic runner writes the
      recipe's training script into the run directory and runs it in Unsloth's environment.
      A QLoRA plan for Qwen3-0.6B on an RTX 4070 Ti SUPER (Windows): Unsloth 2026.9.11, 30
      steps on 300 FineTome examples in 37 s (loss 1.23), merge to 16-bit (1.20 GB), llama.cpp
      convert (20 s) and Q8_0 (3 s, 0.64 GB); `runs/unsloth-qwen06/manifest.json`. The
      notebook repo's `finetune.py` was not needed
- [x] llama.cpp quantize adapter: convert, imatrix, quantize, and `rightsize tools install`
      for pinned binaries plus the matching converter
- [x] VRAM is measured as the rise over what the card held before the step started; the raw
      `memory.used` peak had counted the desktop's 1.5 GB into every figure
- [x] Memory and speed measurement helpers: VRAM sampling, file sizes, a GPU preflight
      that names what else holds the card, and `rightsize bench`
- [x] Evaluation gate: the gate mechanics of `evaluate.py` (a thresholds file, pass / warn /
      fail per candidate), measuring what applies to any model rather than that project's
      task fields: mean KL divergence and top-1 agreement against the 16-bit reference, from
      `llama-perplexity --kl-divergence` over 100 chunks of wikitext-2, thresholds in
      `data/quality/gate_thresholds.yaml`. Completed on Qwen3-1.7B on an RTX 4070 Ti SUPER
      (2026-09-25):

      | Quant | File, predicted / measured | Mean KLD | Top-1 agreement | Gate |
      |---|---|---|---|---|
      | Q4_K_M | 1.243 / 1.282 GB (+3.1%) | 0.059 | 0.900 | warn |
      | Q5_K_M | 1.449 / 1.472 GB (+1.6%) | 0.023 | 0.935 | pass |
      | Q8_0 | 2.159 / 2.165 GB (+0.3%) | 0.003 | 0.975 | pass |

      The 16-bit reference pass took 11 s with the GPU to itself; the earlier attempt shared
      it with LM Studio and failed after 18 minutes, 40% through.
- [ ] llama.cpp VRAM estimates run high. The VRAM sampler recorded what the whole card held,
      desktop included (about 1.5 GB when this run started); less that, the evaluation
      passes used about 1.9 (Q4_K_M), 2.0 (Q5_K_M) and 2.6 GB (Q8_0) against 2.08, 2.29 and
      3.01 GB predicted, 10-14% under. llama.cpp keeps the input embedding table in system
      RAM (src/llama-model.cpp: "always keep it on the CPU"); taking it out of the estimate,
      then refitting the overheads, is the likely fix (F3). The sampler has to measure the
      rise over its starting point first
- [x] Extras: `llamacpp` for the converter. The empty `unsloth` and `diffusers` extras are
      gone: toolkits install into their own environments instead
- [x] Mocked tests; one `slow` GPU test (Qwen3-0.6B end to end)
