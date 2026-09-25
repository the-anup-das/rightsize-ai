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
facts("bartowski/Qwen_Qwen3-14B-GGUF", file="Qwen_Qwen3-14B-Q4_K_M.gguf") -> ModelFacts  # GGUF header
variants("Qwen/Qwen3-14B") -> list[Variant]           # GGUF files, MLX, AWQ, FP8 ... via base_model links
search("llm", "coding", max_params=15e9) -> list[CatalogEntry]  # curated lists, offline
```

How each number is derived and what `confidence` means: [docs/guide/model-facts.md](../guide/model-facts.md).

`ModelFacts` is defined in `src/rightsize/types.py`.

## Data

| Need | Source | Status |
|---|---|---|
| Parameter count per dtype, per file / subfolder | safetensors header via HTTP Range (`huggingface_hub.get_safetensors_metadata` behaviour, re-implemented with httpx); model info `safetensors.parameters` ([docs](https://huggingface.co/docs/safetensors/en/metadata_parsing)) | confirmed |
| Layers, KV heads, head_dim, max context | `config.json` | confirmed |
| Quantized repos for a base model | model card `base_model`; name heuristics (`-GGUF`, `-MLX`, `-AWQ`) | confirmed / heuristic |
| GGUF tensor counts, dtypes, metadata | own reader for the GGUF header (`catalog/gguf.py`); type numbers in `data/quants/ggml_types.yaml` from llama.cpp's gguf-py | done |
| Per-architecture KV overrides (MLA for DeepSeek, sliding window and hybrid for Gemma / Qwen3-Next, SSM hybrids) | `data/quants/kv_overrides.yaml`, hand-authored with source URL | to write |
| Curated lists per family and task (names, tasks, sizes) | `data/models/candidates.yaml` (LLMs), `data/models/families.yaml` (the rest), generated from Hub download listings | done |
| How quantized copies are published (tags, libraries, name patterns, runtimes, publishers) | `data/models/variants.yaml` | done |
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

- [x] `ModelFacts`, `ModelRef` finalised (types.py) and documented
- [x] httpx Range fetch of the safetensors header; parser for the 8-byte length prefix + JSON
- [x] `config.json` reader with architecture normalisation (`model_type` -> family, attention type, MoE).
      The attention block keeps what decides KV layout: layer_types, windows, MLA ranks,
      hybrid layer rules and recurrent state sizes.
- [x] GGUF header reader: magic, version, tensor infos (dtype, dims), metadata KV. Range
      requests that double in size; big-endian files; split models (every part's header);
      tokenizer lists walked, long numeric arrays skipped unfetched; a type newer than the
      table sized from tensor offsets. A GGUF repo's facts come from one file's header (Q4_K_M
      by default); the architecture keys are renamed to config.json's so the KV rules read
      both alike, checked on eight models (plain, sliding, MLA, four hybrids) at 4K-128K.
      Files are grouped by model, so a speculative-decoding draft is not the model's BF16.
      Estimates use the repo's own file per quantization
- [x] `base_model` back-link and `variants()` with name heuristics: the Hub's
      `base_model:quantized:` filter, formats told by `data/models/variants.yaml` (tag, then
      library, then name), one variant per GGUF file with effective bits per weight,
      `name_matches_base` for fine-tunes and drafts that call themselves quants.
      `rightsize variants`, MCP `list_variants`
- [x] KV rules + loader, at `data/runtimes/llama.cpp/kv_cache.yaml` rather than under quants/:
      the behaviour belongs to the runtime, not the model (schema lands with the data schemas)
- [x] Disk cache with TTL and revalidation; `offline` flag; `HF_TOKEN` support. Revalidation
      compares the commit id (`?expand[]=sha`, ~100 bytes) rather than the ETag, which hashes
      the whole model-info response, download count included; commit-id revisions never go
      stale
- [x] Curated lists per family and task: `data/models/families.yaml` (136 diffusion, speech,
      vision and embedding models from `scripts/ingest_families.py`) beside the LLM pool;
      `search()`, `rightsize search`, MCP `search_models`
- [x] Fixtures for 20 repos (`scripts/record_fixtures.py`), one per KV layout plus the
      other families (FLUX.1-dev, SDXL, Wan 2.1, four Whispers, bge-m3, ViT); fake transport;
      tests. Parameter counts come from the Hub's own summary, so recording all 11 takes
      8 seconds instead of minutes.
- [x] Per-subfolder sums for multi-component pipelines (catalog/weights.py): the folders
      model_index.json names, one variant each (the default, else fp16); the root
      single-file checkpoint, spare VAEs and ONNX/OpenVINO exports are skipped. The Hub's
      parameter summary covers one file set (FLUX's transformer, SDXL's UNet), so it is not
      used for pipelines. Gated repos list their files but serve no headers: components are
      then counted from file size at 2 bytes per parameter (4 when the default files are
      twice their fp16 variant), confidence 0.7. Repos with only PyTorch files (bge-m3,
      Kokoro) are sized from `pytorch_model*.bin` or the largest .pth. Family now also
      comes from `library_name` and more pipeline tags.
- [x] Docs page: how facts are derived and what `confidence` means
      ([docs/guide/model-facts.md](../guide/model-facts.md))
- [x] Found on the way: a folder holding the same weights sharded twice (LTX-2.5's
      transformer: 19B counted as 38B) or several checkpoints side by side (LTX-2.3: 22B as
      78B) is counted once; per-stage config lists (SegFormer) no longer fail facts(); a tied
      output head stored twice is recorded (llama.cpp's converter keeps it)
- [ ] Follow-ups: calibrate could find the exact GGUF LM Studio loaded through `variants()`
      (it reports only the base id and quant); model-first recommend for a GGUF repo (F4)
