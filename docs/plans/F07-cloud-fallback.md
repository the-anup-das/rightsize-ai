# F7. Cloud rental fallback

Package: `rightsize.cloud`. Phase 1 (late). Depends on: F3, F4. Feeds: `Plan.cloud_fallback`.

## Goal

When the fine-tuning box is too small (or absent), show the cheapest GPU that fits the job and an estimated cost for it, so "no fit" becomes "rent an L4 for about two hours".

## Why it beats what exists

No VRAM calculator or fine-tuning UI does this. The user gets a complete path instead of a dead end.

## Public API

```python
from rightsize.cloud import cheapest, estimate_job

cheapest(min_vram_gb: float, arch_requirements: list[str] | None = None,
         providers: list[str] | None = None) -> list[Offer]      # sorted by $/hour
estimate_job(fit: FitResult, tokens: int, epochs: int, offer: Offer) -> JobEstimate  # hours, usd, confidence
```

## Data

| Source | What | Notes |
|---|---|---|
| [SkyPilot catalog](https://github.com/skypilot-org/skypilot-catalog) (used) | one CSV of instance prices per provider, 26 providers, on demand and spot, refreshed from the providers' own APIs | public, no key; read at run time and cached a day, never shipped (the catalog repo states no licence) |
| [ComputePrices API](https://computeprices.com/docs/api) | hourly prices across providers, OpenAPI | every endpoint needs an API key (free, 750 requests a day): not used, so rightsize works without an account |
| RunPod GraphQL `gpuTypes` | live prices, availability | RunPod is in the SkyPilot catalog; a direct client adds little until availability matters |
| Modal, Lambda pricing pages | no JSON API | scrape into `data/cloud/` weekly or skip |
| Throughput | `data/cloud/gpu_tflops.yaml`: dense 16-bit tensor TFLOPS from each vendor's datasheet for H100, H200, A100, L40S, L4, A10, T4, RTX 4090 | Hugging Face's table has the non-tensor rate (a quarter of the tensor figure on an H100, half on a 4090), which would mis-rank the cards |

Cache: `data/cloud/prices.json` refreshed hourly by a scheduled job in the data repo; the SDK reads the cached ladder and can refresh on demand.

## Design

- `Offer{provider, gpu, vram_gb, usd_per_hour, region, spot: bool, url, fetched_at}`.
- Filter by the F3 requirement (VRAM after headroom, architecture gates from F4 such as FP8 needing Ada+), sort by price, return top 5.
- Job time = `tokens x epochs / throughput(gpu_class, model_size_bucket, method)`; cost = hours x price. Confidence 0.4 until F9 calibrates.
- `Plan.cloud_fallback` is filled only when the fine-tune step verdict is `no_fit` or `offload`.

## MVP scope

Price ladder from ComputePrices with RunPod as the live source; "rent this" line in the plan. **Advisory only:** launching the job on the provider is F10 (phase 3); until then the plan prints the offer and the rendered recipe for the user to run there.

## Follow-up research

- ComputePrices terms of use and attribution requirements.
- A training-time model with more than a lookup table (sequence length, gradient checkpointing, packing).

## Tests

- Fixture responses for both APIs; sorting and filtering tests; job estimate arithmetic.

## TODO

- [x] `Offer`, `JobEstimate` types (in `rightsize.types`; `Plan.cloud_fallback` carries them
      as JSON, so the Plan schema covers them)
- [x] Price client + cache: SkyPilot's catalog instead of ComputePrices, which needs a key.
      GPU memory is the catalog's weak spot: GpuInfo is MiB at most providers and GiB at
      RunPod, the instance total at AWS but one GPU's at Lambda, counted twice in some
      PrimeIntellect rows and absent at GCP, and "A100" is 40 GB at Lambda and GCP but
      80 GB at Vast. Both readings are tried against the sizes rightsize knows for the card,
      then a size in the name, then the smallest variant: never larger than the card may be
- [ ] RunPod GraphQL client (live availability; prices already come from the catalog)
- [x] Throughput table with sources (datasheet tensor TFLOPS; job time = 6 x params x tokens
      / (TFLOPS x 0.35), confidence 0.3)
- [x] `cheapest()` and `estimate_job()`; `rightsize cloud --vram GB | --model M --mode qlora`
      and the `cloud_offers` MCP tool
- [x] Fill `Plan.cloud_fallback` from F4 when the fine-tune step does not fit:
      `recommend(..., cloud=True)` / `--cloud` plans the fine-tune on the rental GPU with the
      lowest estimated job cost (a card at twice the price per hour can finish in a quarter
      of the time), runs the fine-tune rules against it, and says so in the trace
- [x] Tests with fixtures: rows copied from the catalog, one per quirk
- [ ] Multi-GPU jobs (a 32B full fine-tune needs 527 GB: no single GPU has that)
- [ ] Calibrate the 0.35 efficiency against measured runs (F9)
