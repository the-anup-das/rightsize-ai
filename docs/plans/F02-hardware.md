# F2. Hardware database and detection

Package: `rightsize.hardware`. Phase 1. Depends on: data seed. Feeds: F3, F4, F7.

## Goal

A `Device` record (memory, bandwidth, compute architecture and capability, supported backends, OS, usable-memory fraction) for any GPU, CPU or Apple chip, later phone SoCs; a set of presets; `rightsize detect` that fills a `Device` from the user's machine; prefill from the user's Hugging Face profile.

## Why it beats what exists

Calculators hard-code a GPU dropdown with no sources. We ship an **open table with provenance per field** and detect the real machine (nvidia-smi, rocm-smi, sysctl, plus what Ollama and LM Studio report), so the user types nothing.

## Public API

```python
from rightsize.hardware import get, presets, detect, from_hf

get("RTX 4090") -> Device                  # fuzzy name match over the table
presets() -> dict[str, Device]             # curated common machines
detect() -> Device                         # local probe; never imports torch
from_hf("username") -> list[Device]        # HF profile hardwareItems (public overview API)
```

## Data

| Need | Source | Status |
|---|---|---|
| GPU / CPU / Apple specs: tflops, memory, msrp, power | HF [`hardware.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/hardware.ts). No bandwidth field. | confirmed |
| Memory bandwidth per GPU | TechPowerUp free JSON API and native MCP `techpowerup.com/gpu-specs/api/mcp` **(verify: page 403s; check licence)**; fallbacks `dbgpu` (PyPI), RightNow-GPU-Database | partial |
| Apple Silicon bandwidth | hand table from Apple newsroom / Wikipedia (M5 Max 460–614 GB/s, M5 Ultra ~1.2 TB/s) **(verify)** | manual |
| User hardware with zero data entry | HF OAuth `profile` scope -> `hardwareItems`; `GET /api/users/{u}/overview` ([docs](https://huggingface.co/docs/hub/en/hardware)) | confirmed |
| What is loaded on the box right now | Ollama `GET /api/ps`, LM Studio `GET /api/v0/models`, `nvidia-smi --query-gpu` | confirmed |
| Usable memory fraction | Apple ~70% of unified memory by default (raisable via `sysctl iogpu.wired_limit_mb`); Windows WDDM reserve; driver overhead per OS | to research |

Presets (first set): RTX 3060 12 GB, RTX 4070 12 GB, RTX 4090 24 GB, RTX 5060 Ti 16 GB, RTX 5090 32 GB, M4 Max 64 GB, M5 Max **(verify)**, T4 16 GB (Colab), L4 24 GB, A100 80 GB, H100 80 GB, MI300X 192 GB, DGX Spark 128 GB, CPU-only 32 GB RAM.

## Design

- `data/hardware/gpus.json`, `cpus.json`, `apple.json`: one record per part, every numeric field paired with `provenance`. Ingest script `scripts/ingest_hardware.py` merges sources and writes `fetched_at`.
- Fuzzy lookup: normalise vendor prefixes and memory suffixes ("4090", "RTX 4090 24GB", "GeForce RTX 4090" resolve to one record). Ambiguity returns the candidates instead of guessing.
- Detection is subprocess and httpx only, imported lazily: `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`; `rocm-smi --showmeminfo vram --json`; macOS `sysctl hw.memsize` + `system_profiler SPDisplaysDataType -json`; Windows fallback `wmic`/CIM for RAM. Ollama and LM Studio endpoints add "already loaded" context for F9.
- Multi-GPU: `detect()` returns one `Device` per GPU plus a `system_ram_gib`; aggregation policy lives in F3.

## MVP scope

Table + presets + NVIDIA / Apple / CPU detection. AMD and Intel detection in phase 2. Phone SoCs in the mobile phase.

## Follow-up research

- TechPowerUp API terms and rate limits; whether the MCP endpoint is stable.
- Usable-memory fraction per OS and driver version, with measurements.
- Unified-memory semantics (Apple, DGX Spark, Strix Halo) vs discrete VRAM in the fit engine.

## Tests

- Schema validation of every hardware record; `provenance` required.
- Detection parsers against captured tool output in `tests/fixtures/hardware/` (nvidia-smi CSV, system_profiler JSON, Ollama `/api/ps` JSON).
- Fuzzy lookup table-driven test (20 spellings -> expected record).

## Reuse

- Notebook repo `backend/routers/ai_config.py` and `frontend/.../AIPrefSettings.tsx`: the detect-and-test loop against local servers is the model for `detect()` talking to Ollama and LM Studio.

## TODO

- [ ] `Device` finalised; `data/schema/device.schema.json`, `preset.schema.json`
- [ ] `scripts/ingest_hardware.py`: `hardware.ts` -> `gpus.json` / `cpus.json` / `apple.json` with provenance
- [ ] Bandwidth merge: TechPowerUp or `dbgpu`; Apple hand table marked `hand: true`
- [ ] `data/hardware/presets.yaml` (14 presets)
- [ ] Fuzzy lookup + ambiguity handling
- [ ] `detect()`: nvidia-smi, sysctl / system_profiler, RAM read, Ollama `/api/ps`, LM Studio `/api/v0/models`
- [ ] `from_hf()` via the public overview API
- [ ] Fixtures and tests
- [ ] Verify the two **(verify)** items and record outcomes here
