# F9. Calibration loop

Package: `rightsize.telemetry`. Phase 2. Depends on: F3, F8. Feeds: `data/quality/overheads.yaml`, speed constants, public dataset.

## Goal

With explicit consent, record predicted vs measured memory and speed on the user's machine (from Ollama `/api/ps`, LM Studio's API, `nvidia-smi`, or F8 run manifests), submit anonymised measurements, and periodically refit the estimator constants. Publish the dataset.

## Why it beats what exists

No calculator learns from its errors. Rightsize's constants (runtime overheads, usable-memory fractions, speed efficiency) get better with every opt-in user, and the dataset becomes a public good others can use.

## Public API

```python
from rightsize.telemetry import status, enable, disable, record, submit

status() -> TelemetryStatus        # off by default
enable() -> None                   # prints exactly what will be sent; writes ~/.config/rightsize/consent
record(predicted: FitResult, measured: Measurement, context: dict) -> None   # local JSONL only
submit() -> SubmitResult           # explicit; never automatic in MVP
```

## Design

- **Off by default.** Nothing is recorded or sent until `rightsize telemetry on`. The command prints the exact schema and an example record before asking for confirmation.
- **Measurement schema** (`data/schema/measurement.schema.json`): device class (vendor, arch, memory bucket), OS family, runtime and version, model hash and size bucket, quant, ctx, predicted numbers, measured numbers, rightsize version, data version. **Never**: paths, usernames, hostnames, full model names for private repos, IPs.
- **Local first**: `~/.local/share/rightsize/measurements.jsonl`. `submit()` opens a PR to a public submissions repo (or posts to a tiny endpoint later); the user sees the diff.
- **Refit job** (in the data repo): weekly script reads submissions, refits overhead constants and usable fractions per runtime / OS, opens a PR against `rightsize-data` with before/after error stats.
- **Dataset**: released under CC-BY-4.0 (candidate) alongside `rightsize-data`.

## MVP scope (phase 2)

Consent flow, local recording from Ollama / LM Studio / nvidia-smi, `submit()` via PR, first refit script.

## Follow-up research

- Consent copy reviewed for clarity; anonymisation review (k-anonymity on device class x model bucket).
- Dataset licence and attribution.
- Whether to accept community-submitted measurements from the web platform without the CLI.

## Tests

- Schema validation; a redaction test that feeds paths / usernames and asserts they never appear in the record.
- Recorder parsers against fixtures (Ollama `/api/ps`, LM Studio `/api/v0/models`, nvidia-smi CSV).

## Reuse

- Notebook repo `data_pipeline/metrics.py`: measured tok/s per model / host is the seed of the dataset and the first refit input.

## TODO

- [ ] Consent flag, storage and copy; `rightsize telemetry on|off|status`
- [ ] `Measurement` type and schema; redaction rules with tests
- [ ] Recorders: Ollama, LM Studio, nvidia-smi; hook from F8 manifests
- [ ] Local JSONL store
- [ ] `submit()` via PR to a submissions repo
- [ ] Refit script producing a `rightsize-data` PR with error stats
- [ ] Dataset licence decision recorded here
