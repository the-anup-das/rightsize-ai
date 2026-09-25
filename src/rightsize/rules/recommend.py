"""Hardware-first and model-first recommendations (F4).

    rightsize.recommend(task="chat", target_device="detect")
    rightsize.recommend_for_model("Qwen/Qwen3-14B", target_device="RTX 4090")

For every candidate model and every step of a GGUF quantization ladder, the fit engine (F3)
says whether it fits and how fast it runs; the rules say what is blocked, penalised or
required; what survives is ranked and returned as Plans whose ``trace`` explains each step.

Ranking, formula ``rank.size_vs_quant.v0``:

    score = ln(effective size in B) - 0.9 x perplexity added + ln(rule multiplier)
            + 0.35 x years since 2023 + 0.005 x ln(1 + downloads last 30 days)

There is no licensed cross-model quality benchmark in the data yet, so size stands in for
capability, which holds well within a family and loosely across families - the trace says
so. 0.9 is set so the result agrees with the long-standing llama.cpp rule of thumb: a larger
model at Q4_K_M beats a smaller one at Q8_0, Q3 is roughly a toss-up, and Q2 loses. For a
14B against an 8B that is ln(14/8) = 0.56 against 0.9 x (0.175 - 0.003) = 0.15 at Q4_K_M
(the 14B wins) and 0.9 x (0.657 - 0.022) = 0.57 at Q3_K_M (level). The download term only
separates models of nearly the same size; 12M downloads against 50k is worth about 3% of
size. A mixture-of-experts model counts as the geometric mean of its total and active
parameters, a common rule of thumb for its quality rather than a measured one.

The recency term exists because size alone ranked Mistral-7B-v0.2 (December 2023) at IQ4_XS
above Qwen3-4B (August 2025) at Q8_0. Open models get much better per parameter each year:
the "Densing Law" paper (Xiao et al., arXiv 2412.04315) measures capability density
doubling about every three months. 0.35 a year is deliberately far gentler - effective size
doubling every two years - so a newer model wins among near-equals without a small new
model outranking a much larger one from a year before: Llama-3.3-70B still beats a 32B
released two months later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rightsize._data import load_yaml
from rightsize.fit import estimate, estimate_finetune, ppl_delta
from rightsize.fit.llm import gguf_bpw
from rightsize.fit.quality import band
from rightsize.rules.engine import Outcome, evaluate, load_rules
from rightsize.types import (
    Device,
    Family,
    FitResult,
    Mode,
    ModelFacts,
    ModelRef,
    Plan,
    PlanStep,
    QuantSpec,
    RuntimeSpec,
    Verdict,
)

RANK_ID = "rank.size_vs_quant.v0"
LADDER = (
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_L", "Q3_K_M", "IQ3_M", "Q2_K",
)
QUALITY_FLOORS = {"near-lossless": 0.06, "good": 0.3, "noticeable": 0.7, "any": math.inf}
_LAMBDA = 0.9
_POPULARITY = 0.005
_RECENCY_PER_YEAR = 0.35
_EPOCH = "2023-01-01"
_CODING_BONUS = 0.3  # a code model counts as about 1.35x its size for coding tasks
_SERVING_RUNTIME = "llama.cpp"


@dataclass(frozen=True)
class Candidate:
    repo: str
    facts: ModelFacts
    tasks: tuple[str, ...]
    publisher: str
    created_at: str
    downloads_30d: int
    license: str | None = None
    gated: bool = False


@dataclass
class Rejection:
    repo: str
    quant: str | None
    reason: str


@dataclass
class Result:
    """Plans, best first, and the near misses that explain what did not make it."""

    plans: list[Plan] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


@lru_cache(maxsize=1)
def load_candidates() -> tuple[Candidate, ...]:
    doc = load_yaml("models/candidates.yaml")
    out = []
    for m in doc["models"]:
        f = m["facts"]
        facts = ModelFacts(
            ref=ModelRef(repo=m["repo"]),
            family=Family.llm,
            params_total=f["params_total"],
            params_active=f.get("params_active"),
            num_layers=f["num_layers"],
            num_kv_heads=f["num_kv_heads"],
            head_dim=f["head_dim"],
            context_max=f.get("context_max"),
            dtype=f.get("dtype"),
            license=m.get("license"),
            extra=dict(f.get("extra") or {}),
        )
        out.append(
            Candidate(
                repo=m["repo"],
                facts=facts,
                tasks=tuple(m["tasks"]),
                publisher=m["publisher"],
                created_at=m["created_at"],
                downloads_30d=m["downloads_30d"],
                license=m.get("license"),
                gated=bool(m.get("gated")),
            )
        )
    return tuple(out)


def effective_params_b(facts: ModelFacts) -> tuple[float, str | None]:
    total = (facts.params_total or 0) / 1e9
    active = (facts.params_active or 0) / 1e9
    if active and active < total:
        eff = math.sqrt(total * active)
        return eff, (
            f"mixture of experts: {total:.0f}B total, {active:.1f}B active, counted as "
            f"{eff:.1f}B (geometric mean, a rule of thumb)"
        )
    return total, None


def _device_ctx(dev: Device) -> dict[str, Any]:
    return {
        "name": dev.name,
        "vendor": dev.vendor,
        "os": dev.os,
        "compute_capability": dev.compute_capability,
        "memory_gib": dev.memory_gib,
        "bandwidth_gbps": dev.bandwidth_gbps,
        "backends": list(dev.backends),
        "unified_memory": dev.unified_memory,
    }


def _model_ctx(c: Candidate) -> dict[str, Any]:
    f = c.facts
    return {
        "repo": c.repo,
        "params_b": (f.params_total or 0) / 1e9,
        "context_max": f.context_max,
        "model_type": (f.extra or {}).get("model_type"),
        "is_moe": bool(f.params_active),
        "license": c.license,
    }


def _quant_ctx(q: str, delta: float | None) -> dict[str, Any]:
    from rightsize.execution.quantize import requires_imatrix

    return {
        "method": "gguf",
        "type": q,
        "bits": gguf_bpw(q)[0],
        "is_iquant": q.startswith(("IQ", "TQ")),
        "requires_imatrix": requires_imatrix(q),
        "ppl_delta": delta,
        "band": band(delta),
    }


TASKS = ("chat", "coding", "agentic")


def _serves(c: Candidate, task: str) -> bool:
    """Which pooled models a task draws on. Agentic work uses chat models, with the tool-use
    rules (low-bit quants break tool calling) applied on top; coding uses code models and
    general ones, with a bonus for the code models."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    if task == "coding":
        return bool({"coding", "chat"} & set(c.tasks))
    return "chat" in c.tasks


def _trainer(dev: Device) -> str:
    """Who fine-tunes on this hardware; recipes land with F5."""
    return {"apple": "mlx-lm", "amd": "axolotl"}.get(dev.vendor, "unsloth")


def _years_since_epoch(created_at: str) -> float:
    """Release date as years after 2023, the pool's oldest generation; 0 when unknown."""
    import datetime as dt

    try:
        day = dt.date.fromisoformat(created_at[:10])
    except ValueError:
        return 0.0
    return max((day - dt.date.fromisoformat(_EPOCH)).days / 365.25, 0.0)


def _score(c: Candidate, delta: float, outcome: Outcome, task: str) -> tuple[float, str]:
    eff, _ = effective_params_b(c.facts)
    s = math.log(max(eff, 0.01)) - _LAMBDA * delta + math.log(max(outcome.multiplier, 1e-6))
    years = _years_since_epoch(c.created_at)
    s += _RECENCY_PER_YEAR * years
    s += _POPULARITY * math.log1p(c.downloads_30d)
    bonus = task == "coding" and "coding" in c.tasks
    if bonus:
        s += _CODING_BONUS
    how = (
        f"{RANK_ID}: ln({eff:.2f}B) - {_LAMBDA} x {delta:.3f} perplexity"
        + (f" + ln({outcome.multiplier:.2f}) rules" if outcome.multiplier != 1 else "")
        + (f" + {_RECENCY_PER_YEAR} x {years:.2f} years" if years else "")
        + f" + {_POPULARITY} x ln(1 + {c.downloads_30d:,} downloads)"
        + (f" + {_CODING_BONUS} code model" if bonus else "")
        + f" = {s:.3f}"
    )
    return s, how


def _plan(
    c: Candidate,
    q: str,
    delta: float,
    how_quality: str,
    fit: FitResult,
    target: Device,
    outcome: Outcome,
    score_line: str,
    score: float,
    mode: Mode,
    ft: FitResult | None,
    ft_device: Device | None,
) -> Plan:
    from rightsize.execution.install import LLAMA_CPP_VERSION

    bpw = gguf_bpw(q)[0]
    quant = QuantSpec(method="gguf", variant=q, bits_per_weight=bpw)
    steps: list[PlanStep] = []
    if ft is not None and ft_device is not None:
        steps.append(
            PlanStep(
                stage="finetune",
                framework=_trainer(ft_device),
                device=ft_device,
                quant=QuantSpec(method="bnb", variant="nf4") if mode is Mode.qlora else None,
                fit=ft,
            )
        )
    steps.append(
        PlanStep(stage="quantize", framework="llama.cpp", device=target, quant=quant, fit=fit,
                 recipe_id="llama.cpp/convert")
    )
    # an imatrix when llama-quantize demands one, and for every i-quant anyway: those that
    # run without one still quantize better with it, which is what the rule's note promises
    if "imatrix" in outcome.requires or q.startswith(("IQ", "TQ")):
        steps.append(
            PlanStep(stage="quantize", framework="llama.cpp", device=target, quant=quant,
                     fit=fit, recipe_id="llama.cpp/imatrix")
        )
    steps.append(
        PlanStep(stage="quantize", framework="llama.cpp", device=target, quant=quant, fit=fit,
                 recipe_id="llama.cpp/quantize")
    )
    ctx = next((int(n.split()[1].rstrip(",")) for n in fit.notes if n.startswith("ctx ")), None)
    steps.append(
        PlanStep(stage="serve", framework="llama.cpp", device=target, quant=quant,
                 runtime=RuntimeSpec(name=_SERVING_RUNTIME, version=LLAMA_CPP_VERSION, ctx=ctx),
                 fit=fit, recipe_id="llama.cpp/server")
    )
    _, moe_note = effective_params_b(c.facts)
    speed = f"{fit.speed:.0f} tok/s" if fit.speed else "speed unknown: no bandwidth for this device"
    trace = [
        f"model: {c.repo} - {c.publisher}, released {c.created_at}, "
        f"{c.downloads_30d:,} downloads in the last 30 days",
        f"quant: {q} adds {delta:.3f} perplexity on Llama-3-8B ({how_quality})",
        f"fit: {fit.verdict.value}, {fit.vram_gb:.2f} GB against "
        f"{fit.breakdown.get('usable_memory', 0):.2f} GB usable; {speed} ({fit.formula_id})",
    ]
    if ft is not None:
        trace.append(f"fine-tune: {mode.value} needs {ft.vram_gb:.2f} GB ({ft.formula_id})")
    if moe_note:
        trace.append(moe_note)
    trace.append(score_line)
    trace.append(
        "size stands in for quality: there is no licensed cross-model benchmark in the data yet"
    )
    trace.extend(outcome.trace())
    return Plan(
        rank=1,
        model=c.facts,
        mode=mode,
        steps=steps,
        score=round(score, 3),
        quality_penalty=round(delta, 4),
        trace=trace,
    )


def _evaluate_candidate(
    c: Candidate,
    quants: tuple[str, ...],
    target: Device,
    *,
    task: str,
    ctx: int,
    max_delta: float,
    ignore: frozenset[str],
    mode: Mode,
    ft: FitResult | None,
    ft_device: Device | None,
    rejected: list[Rejection],
    carried: Outcome | None = None,
) -> list[tuple[float, Plan]]:
    rules = tuple(r for r in load_rules() if r.id not in ignore)
    found = []
    for q in quants:
        delta, how = ppl_delta(q)
        if delta is None or delta > max_delta:
            rejected.append(Rejection(c.repo, q, f"quality: {band(delta)} ({how})"))
            continue
        fit = estimate(c.facts, q, target, ctx=ctx)
        if fit.verdict is Verdict.no_fit:
            rejected.append(Rejection(c.repo, q, f"does not fit: needs {fit.vram_gb:.1f} GB"))
            continue
        rule_ctx = {
            "device": _device_ctx(target),
            "model": _model_ctx(c),
            "quant": _quant_ctx(q, delta),
            "runtime": {"name": _SERVING_RUNTIME},
            "task": task,
            "stage": "serve",
            "mode": mode.value,
            "plan": {"tok_s": fit.speed, "verdict": fit.verdict.value, "ctx": ctx},
        }
        outcome = evaluate(rule_ctx, rules)
        if carried is not None:  # what the training step's rules said travels with the plan
            outcome = Outcome(fired=carried.fired + outcome.fired)
        if outcome.blocked:
            why = "; ".join(r.id for r in outcome.fired if r.effect == "block")
            rejected.append(Rejection(c.repo, q, f"blocked: {why}"))
            continue
        score, line = _score(c, delta, outcome, task)
        plan = _plan(c, q, delta, how, fit, target, outcome, line, score, mode, ft, ft_device)
        found.append((score, plan))
    return found


def rank(
    candidates: tuple[Candidate, ...],
    target_device: Device,
    *,
    task: str = "chat",
    ctx: int = 8192,
    quality: str = "noticeable",
    finetune_device: Device | None = None,
    mode: Mode = Mode.infer,
    ignore_rules: frozenset[str] = frozenset(),
    top_k: int = 5,
    one_per_model: bool = True,
    quants: tuple[str, ...] = LADDER,
) -> Result:
    """The engine behind recommend() and recommend_for_model()."""
    max_delta = QUALITY_FLOORS[quality]
    result = Result()
    scored: list[tuple[float, Plan]] = []
    for c in candidates:
        if not _serves(c, task):
            continue
        ft = None
        ft_outcome = Outcome()
        if finetune_device is not None and mode is not Mode.infer:
            ft = estimate_finetune(c.facts, finetune_device, mode)
            if ft.verdict is Verdict.no_fit:
                result.rejected.append(
                    Rejection(c.repo, None, f"{mode.value} needs {ft.vram_gb:.1f} GB to train")
                )
                continue
            ft_outcome = evaluate(
                {
                    "device": _device_ctx(finetune_device),
                    "model": _model_ctx(c),
                    "quant": {"method": "bnb"} if mode is Mode.qlora else {"method": "none"},
                    "runtime": {"name": _trainer(finetune_device)},
                    "task": task,
                    "stage": "finetune",
                    "mode": mode.value,
                },
                tuple(r for r in load_rules() if r.id not in ignore_rules),
            )
            if ft_outcome.blocked:
                why = "; ".join(r.id for r in ft_outcome.fired if r.effect == "block")
                result.rejected.append(Rejection(c.repo, None, f"training blocked: {why}"))
                continue
        found = _evaluate_candidate(
            c, quants, target_device, task=task, ctx=ctx, max_delta=max_delta,
            ignore=ignore_rules, mode=mode, ft=ft, ft_device=finetune_device,
            rejected=result.rejected, carried=ft_outcome,
        )
        if one_per_model and found:
            found = [max(found, key=lambda sp: sp[0])]
        scored.extend(found)
    scored.sort(key=lambda sp: -sp[0])
    for i, (_, plan) in enumerate(scored[:top_k], start=1):
        result.plans.append(plan.model_copy(update={"rank": i}))
    return result


def recommend(
    task: str = "chat",
    target_device: Device | str = "detect",
    *,
    finetune_device: Device | str | None = None,
    mode: Mode | str = Mode.infer,
    ctx: int = 8192,
    quality: str = "noticeable",
    allow_slow: bool = False,
    top_k: int = 5,
) -> list[Plan]:
    """Hardware-first: the best models for this device, one plan each, best first."""
    return recommend_result(
        task, target_device, finetune_device=finetune_device, mode=mode, ctx=ctx,
        quality=quality, allow_slow=allow_slow, top_k=top_k,
    ).plans


def recommend_result(
    task: str = "chat",
    target_device: Device | str = "detect",
    *,
    finetune_device: Device | str | None = None,
    mode: Mode | str = Mode.infer,
    ctx: int = 8192,
    quality: str = "noticeable",
    allow_slow: bool = False,
    top_k: int = 5,
) -> Result:
    from rightsize.hardware import resolve

    mode = Mode(mode)
    if mode is not Mode.infer and finetune_device is None:
        finetune_device = target_device
    return rank(
        load_candidates(),
        resolve(target_device),
        task=task,
        ctx=ctx,
        quality=quality,
        finetune_device=resolve(finetune_device) if finetune_device is not None else None,
        mode=mode,
        ignore_rules=frozenset({"too_slow_for_interactive_use"}) if allow_slow else frozenset(),
        top_k=top_k,
    )


def recommend_for_model(
    model: str | ModelFacts,
    target_device: Device | str = "detect",
    *,
    finetune_device: Device | str | None = None,
    mode: Mode | str = Mode.infer,
    task: str = "chat",
    ctx: int = 8192,
    quality: str = "any",
    allow_slow: bool = False,
    top_k: int = 10,
) -> Result:
    """Model-first: every quantization of one model that works on this device, best first."""
    from rightsize.catalog import facts as hub_facts
    from rightsize.hardware import resolve

    mode = Mode(mode)
    fx = model if isinstance(model, ModelFacts) else hub_facts(model)
    known = {c.repo: c for c in load_candidates()}
    c = known.get(fx.ref.repo) or Candidate(
        repo=fx.ref.repo, facts=fx, tasks=(task,), publisher=fx.ref.repo.split("/")[0],
        created_at="", downloads_30d=0, license=fx.license,
    )
    if mode is not Mode.infer and finetune_device is None:
        finetune_device = target_device
    return rank(
        (c,),
        resolve(target_device),
        task=task,
        ctx=ctx,
        quality=quality,
        finetune_device=resolve(finetune_device) if finetune_device is not None else None,
        mode=mode,
        ignore_rules=frozenset({"too_slow_for_interactive_use"}) if allow_slow else frozenset(),
        top_k=top_k,
        one_per_model=False,
    )
