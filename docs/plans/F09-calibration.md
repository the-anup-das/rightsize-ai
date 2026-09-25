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

- [x] Consent flag, storage and copy; `rightsize telemetry on|off|status|show|export`.
      Off by default; `on` prints exactly what a record holds and asks (or takes `--yes`)
- [x] `CalibrationRecord` type (telemetry/record.py); redaction by allowlist: every
      sub-model forbids extra keys, repo ids only for repos the Hub marks public, device
      names only when they are catalogue names. Tests push a host name, a user path and a
      private repo through and check none comes out
- [x] Recorders: Ollama (`/api/ps`, which reports VRAM per model), LM Studio (`/api/v0/models`
      for what is loaded; memory per process from nvidia-smi on Linux and from the
      "GPU Process Memory" counters on Windows, where WDDM hides it from nvidia-smi). A
      reading is attributed only when one model is loaded in that runtime. First live
      result: gpt-oss-20b MXFP4 in LM Studio at ctx 8192 on an RTX 4070 Ti SUPER,
      predicted 12.35 GB, measured 11.74 GB (+5.2%)
- [ ] Hook from F8 manifests (they record peak VRAM of quantize/eval runs, not serving)
- [x] Local JSONL store (`~/.local/share/rightsize/measurements.jsonl`); `rightsize calibrate`
      prints the comparison whether or not recording is on
- [ ] `submit()` via PR to a submissions repo. For now `rightsize telemetry export FILE`
      writes the records for the user to read and attach to an issue: nothing is sent
- [x] Refit script: `scripts/refit_constants.py` fits overhead = fixed + fraction x weights
      per runtime from the residuals, prints the error before and after, and with `--write`
      updates `data/runtimes/overheads.yaml` (the constants moved there from code), but only
      with five or more records for a runtime
- [ ] Refit the speed efficiency (0.70) from measured tok/s: calibrate does not time
      generation yet; `rightsize bench` does, on one model
- [ ] Dataset licence decision recorded here (CC-BY-4.0 is the candidate; the user decides)
