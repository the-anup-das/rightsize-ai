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

Unsloth QLoRA adapter (fine-tune -> merged or GGUF), llama.cpp quantize adapter, evaluation gate for LLMs. Diffusion and audio execution later.

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
- [ ] Entry-point registration and lazy loading with `MissingExtraError`
- [ ] Unsloth adapter ported from `finetune.py`
- [ ] llama.cpp quantize adapter
- [ ] Memory and speed measurement helpers
- [ ] Evaluation gate ported from `evaluate.py`; `data/quality/thresholds.yaml`
- [ ] Extras populated in `pyproject.toml` (`unsloth`, `llamacpp`)
- [ ] Mocked tests; one `slow` GPU test
