"""Rules: what cannot run where, what costs quality, what needs an extra step (F4).

Rules live in data/rules/*.yaml, one list per file:

    - id: fp8_needs_ada_or_newer
      applies_to: {quant.method: fp8, stage: [quantize, serve]}
      condition: "device.vendor == 'nvidia' and device.compute_capability < 8.9"
      effect: block
      message: FP8 W8A8 needs Ada, Hopper or Blackwell.
      source_url: https://docs.vllm.ai/...
      test:
        fires:  {device: {vendor: nvidia, compute_capability: 8.6}, quant: {method: fp8}, ...}
        silent: {device: {vendor: nvidia, compute_capability: 8.9}, quant: {method: fp8}, ...}

Effects: ``block`` removes the candidate; ``penalize`` multiplies its quality by ``penalty``;
``require`` adds a step (an importance matrix before an i-quant); ``note`` only explains.
Every rule carries the URL that states it and a test case that must fire and one that must
not, so a rule whose condition drifts from its intent fails CI rather than quietly doing
nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rightsize._data import data_dir, load_yaml
from rightsize.rules.expr import Condition

EFFECTS = ("block", "penalize", "require", "note")


@dataclass(frozen=True)
class Rule:
    id: str
    effect: str
    message: str
    source_url: str
    applies_to: Mapping[str, Any] = field(default_factory=dict)
    condition: Condition | None = None
    penalty: float = 1.0
    requires: str | None = None
    test: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, ctx: Mapping[str, Any]) -> bool:
        """Every applies_to key resolves in ctx to the value, or to one of the listed values."""
        for path, want in self.applies_to.items():
            got = _resolve(ctx, path)
            if got is _MISSING:
                return False
            wanted = want if isinstance(want, list) else [want]
            if got not in wanted:
                return False
        return True

    def fires(self, ctx: Mapping[str, Any]) -> bool:
        return self.matches(ctx) and (self.condition is None or self.condition(ctx))

    def line(self) -> str:
        return f"{self.id}: {self.message} ({self.source_url})"


_MISSING = object()


def _resolve(ctx: Mapping[str, Any], path: str) -> Any:
    node: Any = ctx
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return _MISSING
    return node


@dataclass
class Outcome:
    """What the rules made of one candidate."""

    fired: list[Rule] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(r.effect == "block" for r in self.fired)

    @property
    def multiplier(self) -> float:
        m = 1.0
        for r in self.fired:
            if r.effect == "penalize":
                m *= r.penalty
        return m

    @property
    def requires(self) -> list[str]:
        return [r.requires for r in self.fired if r.effect == "require" and r.requires]

    def trace(self) -> list[str]:
        return [r.line() for r in self.fired]


def _build(rec: Mapping[str, Any]) -> Rule:
    return Rule(
        id=rec["id"],
        effect=rec["effect"],
        message=rec["message"],
        source_url=rec["source_url"],
        applies_to=dict(rec.get("applies_to") or {}),
        condition=Condition(rec["condition"]) if rec.get("condition") else None,
        penalty=float(rec.get("penalty", 1.0)),
        requires=rec.get("requires"),
        test=rec.get("test") or {},
    )


@lru_cache(maxsize=4)
def load_rules(root: str | None = None) -> tuple[Rule, ...]:
    """Every rule under data/rules/, conditions parsed once. A bad condition fails here."""
    base = Path(root) if root else data_dir()
    rules: list[Rule] = []
    for path in sorted((base / "rules").glob("*.yaml")):
        rel = path.relative_to(base).as_posix()
        for rec in load_yaml(rel)["rules"] if not root else _read(path)["rules"]:
            rules.append(_build(rec))
    ids = [r.id for r in rules]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate rule ids: {sorted(dupes)}")
    return tuple(rules)


def _read(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def evaluate(ctx: Mapping[str, Any], rules: tuple[Rule, ...] | None = None) -> Outcome:
    return Outcome(fired=[r for r in (rules or load_rules()) if r.fires(ctx)])
