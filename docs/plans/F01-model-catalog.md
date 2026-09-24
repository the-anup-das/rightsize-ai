# F1. Model catalog

Package: `rightsize.catalog`. Phase 1. Depends on: nothing. Feeds: F3, F4.

## Goal

Answer "what is this model" for any Hub repo **without downloading it**: parameters per component and dtype, architecture numbers the KV-cache formula needs (`num_hidden_layers`, `num_key_value_heads`, `head_dim`), family and modality, task, license, and the quantized variants that already exist.

## Why it beats what exists

The Hugging Face hub panel compares **file sizes** and LM Studio estimates one model at a time. We read `config.json` plus the safetensors or GGUF **header** (a few KB over HTTP Range), so context length and KV cache are modelled. vram-calc's handling of MLA, sliding-window and hybrid attention becomes a per-architecture override table anyone can extend.

## Public API

```python
from rightsize.catalog import facts, variants, search

facts("Qwen/Qwen3-14B") -> ModelFacts
facts("bartowski/Qwen3-14B-GGUF", file="Qwen3-14B-Q4_K_M.gguf") -> ModelFacts   # from GGUF header
variants("Qwen/Qwen3-14B") -> list[ModelRef]          # GGUF, MLX, AWQ, GPTQ repos via base_model back-links
search(family="llm", max_params=15e9, task="coding") -> list[ModelRef]  # curated lists + Hub search
```

`ModelFacts` is defined in `src/rightsize/types.py`.

## Data

| Need | Source | Status |
|---|---|---|
| Parameter count per dtype, per file / subfolder | safetensors header via HTTP Range (`huggingface_hub.get_safetensors_metadata` behaviour, re-implemented with httpx); model info `safetensors.parameters` ([docs](https://huggingface.co/docs/safetensors/en/metadata_parsing)) | confirmed |
| Layers, KV heads, head_dim, max context | `config.json` | confirmed |
| Quantized repos for a base model | model card `base_model`; name heuristics (`-GGUF`, `-MLX`, `-AWQ`) | confirmed / heuristic |
| GGUF tensor counts, dtypes, metadata | own reader for the GGUF header (spec in `@huggingface/gguf` and llama.cpp) | to write |
| Per-architecture KV overrides (MLA for DeepSeek, sliding window and hybrid for Gemma / Qwen3-Next, SSM hybrids) | `data/quants/kv_overrides.yaml`, hand-authored with source URL | to write |
| Curated "type" lists per family and task (names only, no metadata bundled) | `data/quants/curated_*.yaml` | to write |
| Local catalogs | LM Studio [`model-catalog`](https://github.com/lmstudio-ai/model-catalog) (open JSON); Ollama has no public registry API | confirmed |

## Design

- Pull-through cache in `~/.cache/rightsize/models/<repo>/<revision>.json` with TTL (7 days) and ETag revalidation. `offline=True` serves from cache only.
- Multi-component pipelines (diffusion: transformer + text encoders + VAE) sum per subfolder and keep the per-component breakdown in `ModelFacts.extra["components"]`.
- `confidence` drops to 0.5 when parameter count comes from a name heuristic ("7B") instead of a header.
- No `transformers`, no `huggingface_hub` dependency; httpx only. Optional `HF_TOKEN` env var for gated repos.
- Architecture normalisation: `model_type` -> family, attention type, MoE flags (`num_experts_per_tok` -> `params_active`).

## MVP scope

LLM and VLM full facts. Diffusion, audio, vision, embeddings: per-subfolder sizes and dtype only; architecture fields empty.

## Follow-up research

- Where quantized variants of non-LLM models live: city96 (GGUF diffusion), nunchaku-ai (SVDQuant), onnx-community, sherpa-onnx release assets.
- HF `.well-known/openapi.json` for any endpoint that already returns GGUF metadata.
- Robust name heuristics for quant repos (bartowski, unsloth, mlx-community, TheBloke legacy).

## Tests

- Recorded HTTP fixtures (`tests/fixtures/catalog/`) for 10 repos across families; replayed with a fake transport, no network in CI.
- GGUF reader against two small real headers checked into fixtures.
- Offline mode returns cached facts and raises a clear error on a miss.

## Reuse

- Notebook repo `data_pipeline/endpoints.py`: per-family quirks and the `THINKING_SWITCH_FAMILIES` set inform the family normalisation table.

## TODO

- [ ] `ModelFacts`, `ModelRef` finalised (types.py) and documented
- [ ] httpx Range fetch of the safetensors header; parser for the 8-byte length prefix + JSON
- [ ] `config.json` reader with architecture normalisation (`model_type` -> family, attention type, MoE)
- [ ] GGUF header reader: magic, version, tensor infos (dtype, dims), metadata KV
- [ ] `base_model` back-link and `variants()` with name heuristics
- [ ] `data/quants/kv_overrides.yaml` + schema + loader
- [ ] Disk cache with TTL and ETag; `offline` flag; `HF_TOKEN` support
- [ ] Curated type lists per family and task (names only)
- [ ] Fixtures for 10 repos; fake transport; tests
- [ ] Per-subfolder sums for multi-component pipelines
- [ ] Docs page: how facts are derived and what `confidence` means
