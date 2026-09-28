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
from rightsize.fit.finetune import merged
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
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q4_K_M",
    "IQ4_XS",
    "Q3_K_L",
    "Q3_K_M",
    "IQ3_M",
    "Q2_K",
)
QUALITY_FLOORS = {"near-lossless": 0.06, "good": 0.3, "noticeable": 0.7, "any": math.inf}
_LAMBDA = 0.9
_POPULARITY = 0.005
_RECENCY_PER_YEAR = 0.35
_EPOCH = "2023-01-01"
_CODING_BONUS = 0.3  # a code model counts as about 1.35x its size for coding tasks


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


def _trainer(dev: Device, pinned=None):
    """Who fine-tunes on this hardware: the pinned framework, else the one whose descriptor
    claims the vendor (data/recipes/<name>/framework.yaml, finetune.default_for), so a new
    trainer is data."""
    if pinned is not None:
        return pinned
    from rightsize.registry import trainer_for

    found = trainer_for(dev.vendor)
    if found is None:
        raise ValueError(f"no framework fine-tunes on {dev.vendor} hardware")
    return found


def _default_server():
    """The framework that serves a GGUF unless another is pinned (serve.default)."""
    from rightsize.registry import framework_infos

    servers = [
        f
        for f in framework_infos().values()
        if f.serve and f.serve.default and "gguf" in f.serve.formats
    ]
    if not servers:
        raise ValueError("no framework.yaml declares a default GGUF server")
    return max(servers, key=lambda f: f.name == "llama.cpp")


def _pin(name: str | None, mode: Mode):
    """What a pinned framework decides: (trainer, server). People use one framework at a
    time, so pinning one puts it wherever it has a role: training when the plan fine-tunes,
    serving when it serves GGUF. The other role keeps its default."""
    server = _default_server()
    if not name:
        return None, server
    from rightsize.errors import NotImplementedYet
    from rightsize.registry import framework

    info = framework(name)
    trains = info.finetune is not None and mode is not Mode.infer
    serves = info.serve is not None and "gguf" in info.serve.formats
    if not (trains or serves):
        if info.finetune is not None:
            raise ValueError(
                f"{info.title} fine-tunes: plan a fine-tune with --mode lora or "
                "qlora (and --finetune-device), or pin a server"
            )
        raise NotImplementedYet(
            f"Planning with {info.title}",
            "docs/plans/F04-rules-engine.md",
            hint="Plans quantize to GGUF and serve it with llama.cpp or Ollama for now; "
            f"`rightsize frameworks {info.name}` shows the commands it has.",
        )
    return (info if trains else None), (info if serves else server)


def _runs_on(info, dev: Device | None) -> str | None:
    """Why a framework cannot run on this device, going by its descriptor's vendors."""
    if info is None or dev is None:
        return None
    vendors = (info.hardware or {}).get("vendors") or []
    if not vendors or "any" in vendors or dev.vendor in vendors:
        return None
    return f"{info.title} runs on {', '.join(vendors)} hardware; {dev.name} is {dev.vendor}"


def _trainer_quant(trainer, mode: Mode) -> QuantSpec | None:
    """How the trainer holds the frozen base weights: QLoRA's 4-bit, as the trainer does it
    (bitsandbytes NF4 for Unsloth, Axolotl and TRL; MLX's own 4-bit on a Mac)."""
    shape = trainer.finetune.qlora_quant
    if mode is not Mode.qlora or shape is None:
        return None
    return QuantSpec(**shape.model_dump())


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
    *,
    trainer=None,
    server=None,
) -> Plan:
    server = server or _default_server()
    bpw = gguf_bpw(q)[0]
    quant = QuantSpec(method="gguf", variant=q, bits_per_weight=bpw)
    steps: list[PlanStep] = []
    if ft is not None and ft_device is not None:
        trainer = _trainer(ft_device, trainer)
        role = trainer.finetune
        before = role.before.get(mode.value)
        if before is not None:
            # a step the trainer needs first: MLX trains QLoRA on a model it quantized itself
            steps.append(
                PlanStep(
                    stage=before.stage,
                    framework=trainer.name,
                    device=ft_device,
                    quant=QuantSpec(**before.quant.model_dump()) if before.quant else None,
                    fit=ft,
                    recipe_id=before.recipe,
                )
            )
        steps.append(
            PlanStep(
                stage="finetune",
                framework=trainer.name,
                device=ft_device,
                quant=_trainer_quant(trainer, mode),
                fit=ft,
                # the recipes train adapters; a mode the trainer has no recipe for (full
                # fine-tuning, for now) is planned but not rendered
                recipe_id=role.recipe if mode.value in role.modes else None,
            )
        )
        if role.writes == "adapter" and role.merge and mode.value in role.modes:
            # the conversion reads a whole 16-bit model: fold the adapter in first
            steps.append(
                PlanStep(
                    stage="export",
                    framework=trainer.name,
                    device=ft_device,
                    fit=ft,
                    recipe_id=role.merge,
                )
            )
    steps.append(
        PlanStep(
            stage="quantize",
            framework="llama.cpp",
            device=target,
            quant=quant,
            fit=fit,
            recipe_id="llama.cpp/convert",
        )
    )
    # an imatrix when llama-quantize demands one, and for every i-quant anyway: those that
    # run without one still quantize better with it, which is what the rule's note promises
    if "imatrix" in outcome.requires or q.startswith(("IQ", "TQ")):
        steps.append(
            PlanStep(
                stage="quantize",
                framework="llama.cpp",
                device=target,
                quant=quant,
                fit=fit,
                recipe_id="llama.cpp/imatrix",
            )
        )
    steps.append(
        PlanStep(
            stage="quantize",
            framework="llama.cpp",
            device=target,
            quant=quant,
            fit=fit,
            recipe_id="llama.cpp/quantize",
        )
    )
    ctx = next((int(n.split()[1].rstrip(",")) for n in fit.notes if n.startswith("ctx ")), None)
    for recipe_id in server.serve.recipes:
        steps.append(
            PlanStep(
                stage="serve",
                framework=server.name,
                device=target,
                quant=quant,
                runtime=RuntimeSpec(
                    name=server.serve.runtime, version=server.version_tested, ctx=ctx
                ),
                fit=fit,
                recipe_id=recipe_id,
            )
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
    trainer=None,
    server=None,
) -> list[tuple[float, Plan]]:
    server = server or _default_server()
    runtime = server.serve.runtime
    rules = tuple(r for r in load_rules() if r.id not in ignore)
    found = []
    for q in quants:
        delta, how = ppl_delta(q)
        if delta is None or delta > max_delta:
            rejected.append(Rejection(c.repo, q, f"quality: {band(delta)} ({how})"))
            continue
        # after a fine-tune the steps convert the model the trainer saved, not the base
        fit = estimate(
            merged(c.facts) if ft is not None else c.facts, q, target, ctx=ctx, runtime=runtime
        )
        if fit.verdict is Verdict.no_fit:
            rejected.append(Rejection(c.repo, q, f"does not fit: needs {fit.vram_gb:.1f} GB"))
            continue
        rule_ctx = {
            "device": _device_ctx(target),
            "model": _model_ctx(c),
            "quant": _quant_ctx(q, delta),
            "runtime": {"name": runtime},
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
        plan = _plan(
            c,
            q,
            delta,
            how,
            fit,
            target,
            outcome,
            line,
            score,
            mode,
            ft,
            ft_device,
            trainer=trainer,
            server=server,
        )
        found.append((score, plan))
    return found


#: Tokens the rental cost is quoted for: a dataset size a reader can scale from.
_QUOTE_TOKENS = 10_000_000
#: Rented NVIDIA GPUs run the Unsloth recipe, which wants Volta or newer.
_RENT_MIN_CC = 7.0


def _rent(c: Candidate, mode: Mode, need: FitResult, pool: list) -> tuple | None:
    """The rental GPU this fine-tune is cheapest to run on: (device, fit, offer, fallback,
    job).

    Every offer with enough memory is checked with the fine-tune estimate, so the rules
    see an ordinary Device. Among those that fit, the lowest estimated job cost wins: a
    card at twice the hourly price can finish in a quarter of the time. Offers with no
    datasheet throughput are chosen only when none has one, by hourly price."""
    from rightsize.cloud import cheapest, device_for, estimate_job

    params = c.facts.params_active or c.facts.params_total or 0
    fitting = []
    for offer in cheapest(need.vram_gb, pool=pool, top=50, min_compute_capability=_RENT_MIN_CC):
        base = device_for(offer.gpu, offer.vram_gib)
        dev = Device(
            name=f"{offer.provider} {offer.gpu}",
            vendor="nvidia",
            memory_gib=offer.vram_gib,
            compute_capability=offer.compute_capability,
            compute_arch=base.compute_arch if base else None,
            backends=["cuda"],
            os="linux",
            usable_fraction=0.92,
        )
        fit = estimate_finetune(c.facts, dev, mode)
        if fit.verdict is Verdict.fits:
            fitting.append((dev, fit, offer, estimate_job(params, _QUOTE_TOKENS, offer)))
    if not fitting:
        return None
    timed = [f for f in fitting if f[3].usd is not None]
    dev, fit, offer, job = min(timed, key=lambda f: f[3].usd) if timed else fitting[0]
    fallback = {
        "reason": f"{mode.value} needs {need.vram_gb:.1f} GB to train",
        "offer": offer.model_dump(mode="json"),
        "job_per_10m_tokens": job.model_dump(mode="json"),
        "cheapest_per_hour": fitting[0][2].model_dump(mode="json"),
    }
    return dev, fit, offer, fallback, job


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
    cloud_pool: list | None = None,
    framework: str | None = None,
) -> Result:
    """The engine behind recommend() and recommend_for_model().

    ``cloud_pool`` is a list of rental offers: a fine-tune that does not fit the fine-tune
    device is then planned on the cheapest offer it fits, instead of being rejected.
    ``framework`` pins the trainer, the server, or both (see _pin)."""
    max_delta = QUALITY_FLOORS[quality]
    result = Result()
    scored: list[tuple[float, Plan]] = []
    pinned, server = _pin(framework, mode)
    why = _runs_on(pinned, finetune_device) or (
        _runs_on(server, target_device) if framework else None
    )
    if why:
        raise ValueError(why)  # the pin, not any one model, is what cannot run here
    if pinned is not None and "nvidia" not in (pinned.hardware or {}).get("vendors", ["nvidia"]):
        cloud_pool = None  # rental GPUs are NVIDIA, which the pinned trainer does not run on
    for c in candidates:
        if not _serves(c, task):
            continue
        ft = None
        ft_outcome = Outcome()
        ft_device = finetune_device
        rental = None
        if finetune_device is not None and mode is not Mode.infer:
            ft = estimate_finetune(c.facts, finetune_device, mode)
            if ft.verdict is Verdict.no_fit and cloud_pool:
                rental = _rent(c, mode, ft, cloud_pool)
                if rental:
                    ft_device, ft = rental[0], rental[1]
            if ft.verdict is Verdict.no_fit:
                why = f"{mode.value} needs {ft.vram_gb:.1f} GB to train"
                if cloud_pool:
                    why += "; no single rental GPU has that much (multi-GPU is not planned yet)"
                result.rejected.append(Rejection(c.repo, None, why))
                continue
            ft_outcome = evaluate(
                {
                    "device": _device_ctx(ft_device),
                    "model": _model_ctx(c),
                    "quant": {
                        "method": (
                            _trainer_quant(_trainer(ft_device, pinned), mode)
                            or QuantSpec(method="none")
                        ).method
                    },
                    "runtime": {"name": _trainer(ft_device, pinned).name},
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
            c,
            quants,
            target_device,
            task=task,
            ctx=ctx,
            max_delta=max_delta,
            ignore=ignore_rules,
            mode=mode,
            ft=ft,
            ft_device=ft_device,
            rejected=result.rejected,
            carried=ft_outcome,
            trainer=pinned,
            server=server,
        )
        if rental and found:
            _, _, offer, fallback, job = rental
            cost = (
                f"about ${job.usd:,.2f} and {job.hours:g} h per 10M training tokens "
                f"(confidence {job.confidence})"
                if job.usd is not None
                else "no time estimate for this GPU"
            )
            line = (
                f"fine-tune: {fallback['reason']}, more than the {finetune_device.name} "
                f"has; rent a {offer.gpu} ({offer.vram_gib:.0f} GiB) on {offer.provider} at "
                f"${offer.usd_per_hour:.2f}/h, {cost} [{offer.source_url}]"
            )
            found = [
                (s, p.model_copy(update={"cloud_fallback": fallback, "trace": [*p.trace, line]}))
                for s, p in found
            ]
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
    cloud: bool = False,
    framework: str | None = None,
) -> list[Plan]:
    """Hardware-first: the best models for this device, one plan each, best first.

    ``cloud=True`` plans a fine-tune that does not fit the fine-tune device on the cheapest
    rental GPU it fits (F7), rather than leaving that model out. ``framework`` pins the
    toolkit: a trainer (unsloth, trl, axolotl, mlx-lm) for the fine-tune, a server (ollama,
    llama.cpp) for the GGUF."""
    return recommend_result(
        task,
        target_device,
        finetune_device=finetune_device,
        mode=mode,
        ctx=ctx,
        quality=quality,
        allow_slow=allow_slow,
        top_k=top_k,
        cloud=cloud,
        framework=framework,
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
    cloud: bool = False,
    framework: str | None = None,
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
        cloud_pool=_cloud_pool(cloud, mode),
        framework=framework,
    )


def _cloud_pool(cloud: bool, mode: Mode) -> list | None:
    """Rental offers when they could matter: asked for, and a fine-tune in the plan."""
    if not cloud or mode is Mode.infer:
        return None
    from rightsize.cloud import offers

    pool, _ = offers()
    return pool


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
    cloud: bool = False,
    framework: str | None = None,
) -> Result:
    """Model-first: every quantization of one model that works on this device, best first.
    ``framework`` pins the toolkit, as in recommend()."""
    from rightsize.catalog import facts as hub_facts
    from rightsize.hardware import resolve

    mode = Mode(mode)
    fx = model if isinstance(model, ModelFacts) else hub_facts(model)
    if (fx.extra or {}).get("gguf"):
        # Plans quantize from the original weights; a repo of ready-made GGUFs needs a
        # download step instead, which is not planned yet.
        from rightsize.errors import NotImplementedYet

        hint = f"`rightsize estimate {fx.ref.repo} --quant Q4_K_M` sizes any file in it"
        if fx.base_model:
            hint += f", and `rightsize recommend --model {fx.base_model}` ranks its base model"
        raise NotImplementedYet(
            "Model-first plans for a repo of ready-made GGUFs",
            "docs/plans/F04-rules-engine.md",
            hint=f"For now, {hint}.",
        )
    known = {c.repo: c for c in load_candidates()}
    c = known.get(fx.ref.repo) or Candidate(
        repo=fx.ref.repo,
        facts=fx,
        tasks=(task,),
        publisher=fx.ref.repo.split("/")[0],
        created_at="",
        downloads_30d=0,
        license=fx.license,
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
        cloud_pool=_cloud_pool(cloud, mode),
        framework=framework,
    )
