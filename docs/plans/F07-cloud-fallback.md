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
| [ComputePrices API](https://computeprices.com/docs/api) | hourly prices across providers, OpenAPI | provider-agnostic ladder; terms to check |
| RunPod GraphQL `gpuTypes` | live prices, availability | direct integration |
| Modal, Lambda pricing pages | no JSON API | scrape into `data/cloud/` weekly or skip |
| Throughput table | tokens/s per GPU class for QLoRA of size buckets (published Unsloth / Axolotl numbers) | low confidence, marked |

Cache: `data/cloud/prices.json` refreshed hourly by a scheduled job in the data repo; the SDK reads the cached ladder and can refresh on demand.

## Design

- `Offer{provider, gpu, vram_gb, usd_per_hour, region, spot: bool, url, fetched_at}`.
- Filter by the F3 requirement (VRAM after headroom, architecture gates from F4 such as FP8 needing Ada+), sort by price, return top 5.
- Job time = `tokens x epochs / throughput(gpu_class, model_size_bucket, method)`; cost = hours x price. Confidence 0.4 until F9 calibrates.
- `Plan.cloud_fallback` is filled only when the fine-tune step verdict is `no_fit` or `offload`.

## MVP scope

Price ladder from ComputePrices with RunPod as the live source; "rent this" line in the plan.

## Follow-up research

- ComputePrices terms of use and attribution requirements.
- A training-time model with more than a lookup table (sequence length, gradient checkpointing, packing).

## Tests

- Fixture responses for both APIs; sorting and filtering tests; job estimate arithmetic.

## TODO

- [ ] `Offer`, `JobEstimate` types; `data/schema/offer.schema.json`
- [ ] ComputePrices client + cache
- [ ] RunPod GraphQL client
- [ ] Throughput lookup table with sources
- [ ] `cheapest()` and `estimate_job()`
- [ ] Fill `Plan.cloud_fallback` from F4 when the fine-tune step does not fit
- [ ] Tests with fixtures
