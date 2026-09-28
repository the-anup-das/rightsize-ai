# How Rightsize knows a model without downloading it

Every estimate starts from `ModelFacts`: parameter count, the architecture numbers the KV
cache depends on, family, licence, and how sure we are of each. `rightsize.catalog.facts()`
builds it from the Hugging Face Hub with a handful of small requests and never downloads
weights. This page says where each number comes from and what `confidence` means.

```python
from rightsize.catalog import facts, variants, search

facts("Qwen/Qwen3-4B")                                    # config.json + safetensors
facts("unsloth/Qwen3-4B-GGUF")                            # the header of its Q4_K_M
facts("unsloth/Qwen3-4B-GGUF", file="Qwen3-4B-Q8_0.gguf") # or of a file you name
variants("Qwen/Qwen3-4B")          # GGUF, MLX, AWQ, FP8 ... copies already published
search("diffusion", "text-to-image", max_params=13e9)     # curated lists, offline
```

## Where each number comes from

**The model-info request** (`/api/models/<repo>?blobs=true`) lists every file with its
size, and gives the pipeline tag, library, licence, whether the repo is gated, the commit it
is at, and, for most safetensors repos, the Hub's own count of parameters per dtype.

**Architecture** comes from `config.json` (the text model's part of a vision-language
config): layers, attention and KV heads, head size, context length, and the keys that decide
how the KV cache is laid out: sliding windows, MLA ranks, which layers are recurrent and how
big their state is. Those are read by the runtime's rules in
`data/runtimes/llama.cpp/kv_cache.yaml`, not by the catalog.

**Parameters**, in the order they are tried:

| Source | When | Requests |
|---|---|---|
| The Hub's parameter summary | most safetensors repos | none extra |
| safetensors headers (8-byte length, then the JSON header, by HTTP Range) | no summary | two per file |
| File sizes over bytes per parameter | gated repos: the files are listed, the headers are not served | none |
| PyTorch files' sizes | repos with no safetensors (bge-m3, Kokoro) | none |
| The GGUF header | repos of GGUF files | a few, see below |

A repo often holds the same weights more than once, and each copy is counted once:

- variants side by side (`model.safetensors` and `model.fp16.safetensors`): the default set;
- the same weights sharded twice (LTX-2.5's transformer has a 4-shard and an 8-shard set):
  one set;
- several different checkpoints at the root (LTX-2.3's dev, distilled and LoRA files): the
  largest, with the others named in `extra["other_checkpoints"]` and confidence 0.7;
- a diffusers pipeline: only the folders `model_index.json` names, one variant each, never
  the root single-file checkpoint, spare VAEs or ONNX and OpenVINO exports.

## GGUF repos

A repo of GGUFs has no `config.json`, so its facts come from a GGUF header, read front to
back in Range requests that double in size:

```
magic "GGUF" | version | tensor count | metadata count
metadata: key, type, value           (the tokenizer's lists live here)
tensors:  name, dims, type, offset
padding, then the weights            (never fetched)
```

Without a named file, the repo is read from its Q4_K_M, the usual default of LM Studio and
Ollama, or the nearest file it has. Split models read every part's header. The header gives
the exact type and shape of every tensor, so the parameter count, the bytes of weights, which
tensors are the input embedding and the output head, and whether they are tied are all
counted, not assumed. Tensor and file types are numbered as in llama.cpp's `gguf-py` at the
tag we pin (`data/quants/ggml_types.yaml`); a type newer than that table is sized from the
gap between tensor offsets.

The architecture metadata (`qwen3.block_count`, `qwen3.attention.head_count_kv`, ...) is
renamed to the `config.json` names, and hybrid models' per-layer KV head counts (zero for a
recurrent layer) become a layer list, so the KV rules read a GGUF exactly as they read the
model it came from. Eight models were checked this way, covering plain attention, sliding
windows, MLA and four kinds of hybrid: the KV cache agrees at 4K, 32K and 128K context
(`tests/test_gguf.py`).

The header is small next to the weights but not tiny: the tokenizer's token and merge lists
are stored in it, several MB for a 150k vocabulary (13 MB for gpt-oss-20b). Those lists are
walked, not kept, and long numeric arrays are skipped without being fetched. The facts are
cached, so this happens once per model.

An estimate for a GGUF repo uses the repo's own file for each quantization: the tensor bytes
of the file whose header was read, the file size less that header for the others. A
quantization the repo does not have falls back to the bits-per-weight table, and says so.

## What confidence means

`ModelFacts.confidence` is about the parameter count, the number every memory estimate
scales with:

| Confidence | The count came from |
|---|---|
| 1.0 | the weights themselves: safetensors headers, the Hub's summary of them, a GGUF header |
| 0.7 | file sizes, which means assuming bytes per parameter (a gated repo read without a token); one choice among several checkpoints in the repo; the Hub's summary of a gated GGUF |
| 0.5 | nothing: the count is unknown |

`extra["params_counted_from"]` says which, in words. An estimate's own
`FitResult.confidence` is separate: it is about the formula, not the facts.

## When two sources disagree

They sometimes do, and each difference has a reason:

- **A tied output head stored twice.** Qwen3-0.6B ties its output head to the input
  embedding and stores it anyway, so the Hub counts 751.6M parameters. llama.cpp's converter
  keeps the copy as `output.weight`, so a GGUF you make has 751.6M; unsloth's published GGUF
  drops it and has 596M. `extra["tied_head_stored"]` records the copy.
- **Multi-token-prediction layers.** Qwen3-Next's checkpoint carries one; the converter
  leaves it out, so the GGUF is 1.65B parameters lighter. The GGUF is what llama.cpp loads.
- **Context length.** The GGUF's `context_length` is what the converter wrote, which can
  differ from `max_position_embeddings` (Granite 4.0-H: 1M against 32K).

## The cache

Facts are cached in `~/.cache/rightsize/models/<repo>/<revision>.json` (`RIGHTSIZE_CACHE_DIR`
moves it). After seven days a cached answer is checked against the repo's current commit
with one request of about a hundred bytes (`?expand[]=sha`), and reused if nothing changed.
The model-info ETag cannot do that job: it hashes the whole response, download count
included, so it changes when the repo has not. Facts for a commit id never go stale.

`--offline` (or `RIGHTSIZE_OFFLINE=1`) answers from the cache only, however old, and says
so on a miss. `HF_TOKEN` is sent when set, which lets gated repos serve their headers.

## Quantized copies already on the Hub

A model card can name its base model and how it relates to it (`base_model_relation:
quantized`), and the Hub lists everything that does: `rightsize variants Qwen/Qwen3-4B`. Each
copy's format is told by the conventions in `data/models/variants.yaml`: the Hub tag, then
the library name, then a pattern in the repo name. Supporting a new format, or a publisher's
naming habit, is an edit to that file.

The base-model link is only as good as the card that declares it. Fine-tunes and
speculative-decoding drafts sometimes call themselves quantizations too, so each variant says
whether its name is the base model's name plus format words (`name_matches_base`); the CLI
hides the ones that are not and names them underneath.

## Curated lists

`rightsize search` and `catalog.search()` read two lists generated from what people download,
offline: `data/models/candidates.yaml` for LLMs (the pool `recommend` ranks) and
`data/models/families.yaml` for diffusion pipelines, speech, vision and embedding models, by
the Hub's task names (`rightsize search --task list`). Both are rebuilt by scripts in
`scripts/`, from publisher allowlists, without re-uploads in other formats. Anything not on a
list can still be estimated by its Hub id.

## What it cannot tell yet

- A gated repo read without `HF_TOKEN` has no architecture, so no KV cache estimate.
- A repo that is only a diffusion transformer in GGUF (city96's files) is sized as that one
  file; estimate the diffusers pipeline it belongs to with `--quant Q4_K` instead.
- Repos in a model's original, non-diffusers layout (Wan's `.pth` text encoder) are counted
  from their safetensors only.
