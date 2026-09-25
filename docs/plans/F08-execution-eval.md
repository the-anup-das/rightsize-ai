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

Adapters register via the `rightsize.adapters` entry point and live behind extras (`[unsloth]`, `[llamacpp]`, `[diffusers]`). Missing extra raises `MissingExtraError` with the install line.

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

- [ ] `Adapter` protocol, `RunManifest`, `EvalReport` types
      (partial: `RunManifest` and `RunStep`; the gate reports a dict, and there is one adapter,
      so no protocol yet)
- [ ] Entry-point registration and lazy loading with `MissingExtraError`
- [ ] Unsloth adapter ported from `finetune.py`
- [x] llama.cpp quantize adapter: convert, imatrix, quantize, and `rightsize tools install`
      for pinned binaries plus the matching converter
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
- [ ] Extras populated in `pyproject.toml` (`unsloth`, `llamacpp`)
      (partial: `llamacpp` done; `unsloth` still empty)
- [x] Mocked tests; one `slow` GPU test (Qwen3-0.6B end to end)
