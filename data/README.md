# rightsize-data (seed)

Everything Rightsize knows that is a *number* or a *table* lives here, not in Python. The package ships this folder, and `rightsize data update [--ref main|TAG]` fetches a newer one from this repository into `~/.cache/rightsize/data/current`. An update is used only if every file validates against the installed release's schemas and every file that release reads is present; otherwise it is refused and the current data stays. `rightsize data status` says which data is in use, `rightsize data reset` returns to the shipped copy, and `RIGHTSIZE_DATA_DIR` overrides both. A separate `rightsize-data` repository with its own releases can take over later without changing this.

| Folder | Contents | Source (to ingest) | Owner |
|---|---|---|---|
| `hardware/` | `gpus.yaml` (ingested SKU catalogue), `bandwidth.yaml` (hand-typed) + `bandwidth_wikipedia.yaml` (ingested), `presets.yaml` (curated setups) | huggingface.js SKU tables (MIT); Wikipedia GPU lists (CC BY-SA); vendor pages | F2 |
| `quants/` | Real bits-per-weight per GGUF type; per-architecture KV overrides (MLA, sliding window, hybrid); MLX/bnb/AWQ/GPTQ/FP8 descriptors | `@huggingface/gguf` `quant-descriptions.ts`, model configs | F1, F3 |
| `rules/` | Hard gates and penalties with `source_url` and a test each | HF/vLLM/SGLang quantization matrices, toolkit docs | F4 |
| `recipes/` | Renderable command/config templates per framework and stage | Each toolkit's docs, pinned by version | F5 |
| `quality/` | Base-model quality and quant penalty tables | Unsloth KL tables, Artificial Analysis, own `llama-perplexity` runs | F4 |
| `cloud/` | Datasheet tensor TFLOPS for the job-time estimate; prices are fetched at run time from SkyPilot's catalog and never stored here | NVIDIA / AMD datasheets | F7 |
| `runtimes/` | How each runtime lays out the KV cache, its memory overheads (refit from calibration records), diffusers and Whisper memory models with the measurements behind them | llama.cpp source, published benchmarks, `rightsize calibrate` | F3, F9 |
| `schema/` | JSON Schemas every file above must validate against | this repo | all |

## Rules for every record

- `source_url` and `fetched_at` are required. No number without a source.
- Prefer ingest scripts (`scripts/ingest_*.py`, to come) over hand edits; hand tables are marked `"hand": true`.
- Changing data never requires a code release.

## Units

Two units, each matching the source a reader would check the number against:

- **Device memory is GiB** (`memory_gib`, `system_ram_gib`; 1024³ bytes). A "16GB" card is `16`, which is what the vendor page says and what `nvidia-smi` reports (16376 MiB). llama.cpp prints MiB too.
- **Model and file sizes are decimal GB** (1e9 bytes), which is how Hugging Face lists file sizes and how the published numbers behind our golden tests are quoted.

`Device.memory_gb` converts the first into the second, and the fit engine only ever uses that. 16 GiB is 17.18 GB; comparing model sizes against a bare `16` made every verdict on this card 7% pessimistic. Bandwidth (`bandwidth_gbps`) is decimal GB/s, as vendors state it.

## Where the hardware numbers come from

`scripts/ingest_hf_hardware.py` pulls huggingface.js's MIT-licensed SKU tables (memory
options, TFLOPS, compute capability, MSRP, power, year) pinned to a commit.

That table has no memory bandwidth, and it is the number the speed model runs on.
TechPowerUp serves a captcha and marks its pages `noindex,nofollow`, so it is off limits;
Wikidata has no bandwidth property. `scripts/ingest_bandwidth.py` therefore reads Wikipedia's
"List of ... graphics processing units" tables, which give bus width, bus type and memory
clock per SKU, and computes

    bandwidth_gbps = bus_width_bits * memory_speed_gbps / 8

rather than copying anyone's derived column. That matters: Wikipedia's own bandwidth column
says 336 GB/s for the RTX 3060 12 GB where its bus width and clock give 360, which is what
NVIDIA publishes. The stated figure is kept beside ours as `stated_gbps` so the two can be
compared, and the ingest refuses to write if it disagrees by more than 2% with anything
hand-typed in `bandwidth.yaml`.

Records carry the vendor they came from. `_norm` drops the vendor word when matching, so
"Apple M4" and an old Mobility Radeon M4 both reduce to `m4`; without the vendor check an
Apple chip would quietly report a Radeon's bandwidth.
