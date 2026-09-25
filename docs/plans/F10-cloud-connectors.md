# F10. Cloud provider connectors

Package: `rightsize.cloud.connectors`. **Phase 3.** Depends on: F7 (offers), F8 (adapters, run manifests), F5 (rendered recipes). Decision 2026-09-25: execution is local-only for now; connectors come after the local path is solid.

## Goal

When a plan's fine-tune or quantize step does not fit the local box, let the user run that step on a rented GPU from inside Rightsize: pick the offer F7 found, launch the rendered recipe there, stream logs, bring the artifact back, and record predicted vs actual (F9) exactly as a local run would.

## Why it beats what exists

F7 already turns "no fit" into "rent this". Connectors turn it into "run it there", with the same `Plan`, the same recipe and the same manifest. Fine-tuning UIs that offer cloud (Kiln, Predibase, Together) run their own stack; Rightsize runs the user's chosen open-source toolkit on the user's chosen provider.

## Public API

```python
from rightsize.cloud import connectors

connectors.available() -> list[str]                       # installed extras: runpod, modal, lambda, vast, skypilot, ...
job = connectors.launch(step: RenderedStep, offer: Offer, workdir: Path, sync: SyncSpec) -> RemoteJob
job.logs()            # streaming
job.wait() -> RunManifest                                 # same type as a local F8 run
job.fetch(artifact) -> Path
job.cancel()
```

CLI: `rightsize run <plan.json> --step finetune --on runpod` (local is the default and needs no flag).

## Design

- **Same adapter protocol as F8.** A connector is an `Adapter` whose `run()` provisions, executes and tears down; the recipe text does not change. Connectors live behind extras (`rightsize[runpod]`, `[modal]`, `[skypilot]`) and load lazily.
- **Provider abstraction candidates (decide in phase 3):** SkyPilot (one YAML to many clouds, spot handling, auto-stop) as the default backend, with direct connectors for RunPod (GraphQL + pods API) and Modal (Python SDK) where SkyPilot is too heavy. Shadeform's single API across 30+ providers is an alternative aggregator **(verify terms)**.
- **Data movement:** dataset and base weights upload via the provider's volume or object store; artifacts (adapter, merged weights, GGUF) download to `workdir`; nothing is kept on the provider after `fetch()` unless the user asks.
- **Secrets:** API keys from environment variables or the provider's own CLI config only; Rightsize never stores them.
- **Cost guard:** `launch()` refuses unless the F7 job estimate and a hard `max_usd` are set; auto-stop on completion or timeout.
- **Managed fine-tuning APIs** (Together, Fireworks, OpenAI, Predibase) are a different product: they run their own recipe. Out of scope unless a plan step maps exactly to one of them.

## MVP scope (phase 3)

One aggregator (SkyPilot) plus RunPod direct; fine-tune and quantize steps for LLMs; log streaming; artifact fetch; manifest into F9.

## Follow-up research

- SkyPilot's current provider list, spot support and image requirements for Unsloth / llama.cpp containers.
- RunPod and Modal API stability and rate limits; Shadeform terms.
- Container images: one per recipe family (Unsloth CUDA, llama.cpp CUDA, diffusers) pinned to the `version_tested` in the recipe.

## Tests

- Connector protocol tests with a fake provider; cost-guard tests; secrets never serialized into manifests or logs (redaction test shared with F9).
- One opt-in `slow` test per real provider, run manually.

## TODO

- [ ] Decide the aggregator (SkyPilot vs direct connectors) after phase 2
- [ ] `RemoteJob`, `SyncSpec` types; connector `Adapter` implementation with provision / execute / teardown
- [ ] SkyPilot backend; RunPod direct backend
- [ ] Container image matrix pinned to recipe versions
- [ ] Cost guard and auto-stop
- [ ] Log streaming and artifact fetch
- [ ] Manifest into F9; redaction tests
- [ ] CLI `rightsize run --on <provider>`
