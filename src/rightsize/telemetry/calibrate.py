"""Compare rightsize's prediction with what the GPU holds, for every loaded model (F9).

    rightsize calibrate

Works with or without recording on: the comparison is always printed, and appended to the
local store only when the user has turned recording on.
"""

from __future__ import annotations

from dataclasses import dataclass

from rightsize.telemetry import record as rec
from rightsize.telemetry import sources
from rightsize.types import Device


@dataclass
class Comparison:
    runtime: str
    model: str
    quant: str | None
    ctx: int | None
    predicted_gb: float | None
    measured_gb: float | None
    note: str
    record: rec.CalibrationRecord | None = None

    @property
    def error(self) -> float | None:
        if self.predicted_gb and self.measured_gb:
            return self.predicted_gb / self.measured_gb - 1
        return None


def _measure_lm_studio(loaded: list[sources.Loaded], limit_gb: float) -> dict[str, float]:
    """LM Studio's memory, per model, when it can be told apart: one model, one engine."""
    if len(loaded) != 1:
        return {}
    readings = sources.process_vram(sources.LM_STUDIO_ENGINE, limit_gb)
    return {loaded[0].name: readings[0]} if len(readings) == 1 else {}


def calibrate(device: Device, *, write: bool = True, lm_studio=None, ollama=None,
              facts_for=None) -> list[Comparison]:
    """One comparison per loaded model. ``lm_studio``, ``ollama`` and ``facts_for`` stand
    in for the live sources in tests."""
    from rightsize.fit import estimate

    if facts_for is None:
        from rightsize.catalog import facts as facts_for
    lms = sources.lm_studio() if lm_studio is None else lm_studio
    oll = sources.ollama() if ollama is None else ollama
    measured = _measure_lm_studio(lms, device.memory_gb)
    out: list[Comparison] = []
    for m in [*lms, *oll]:
        got = m.vram_gb if m.vram_gb is not None else measured.get(m.name)
        facts, pred, note = None, None, ""
        if m.hub_repo:
            try:
                facts = facts_for(m.hub_repo)
            except Exception as exc:  # a local-only or renamed model: no prediction
                note = f"no Hub facts for {m.hub_repo} ({type(exc).__name__})"
        else:
            note = "not a Hub id, so nothing to predict from"
        if facts is not None and m.quant:
            try:
                pred = estimate(facts, m.quant, device, runtime=m.runtime, ctx=m.ctx or 8192)
            except KeyError as exc:
                note = f"quant {m.quant} unknown to rightsize ({exc})"
        if got is None and not note:
            note = ("memory not attributable: more than one model or engine process"
                    if m.runtime == "lm studio" else "the runtime did not report its memory")
        record = None
        if pred is not None and got is not None:
            record = rec.build(
                source=m.runtime, device=device, runtime=m.runtime, facts=facts,
                quant=m.quant, ctx=m.ctx, predicted=pred, measured_vram_gb=round(got, 3),
            )
            if write:
                rec.append(record)
        out.append(Comparison(m.runtime, m.name, m.quant, m.ctx,
                              pred.vram_gb if pred else None, got, note, record))
    return out
