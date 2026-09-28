"""Quantize to any format a toolkit produces: ``rightsize quantize MODEL --to fp8`` (F8).

GGUF has its own pipeline (execution/quantize.py: convert, importance matrix, quantize, the
KL gate). Every other format comes from a recipe that declares it as a target - llm-compressor
for fp8 and w4a16, Transformers for nf4, Optimum Intel for OpenVINO, MLX LM, Sentence
Transformers for ONNX, CTranslate2 for Whisper - so a toolkit that adds a format adds a
target to its recipe and nothing here changes. The step runs through the same runner as any
plan, in the toolkit's own environment, and the manifest sets the output's size beside the
prediction from quants/formats.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rightsize.types import (
    Device,
    FitResult,
    Mode,
    ModelFacts,
    Plan,
    PlanStep,
    QuantSpec,
    RunManifest,
    Verdict,
)

Log = Callable[[str], None]
FORMULA_ID = "quantize.size.v0"


def targets() -> dict[str, list[tuple[Any, Any]]]:
    """Every format --to accepts, with the (recipe, target) pairs that produce it."""
    from rightsize.registry import all_recipes

    found: dict[str, list[tuple[Any, Any]]] = {}
    for recipe in all_recipes().values():
        for target in recipe.targets:
            found.setdefault(target.name, []).append((recipe, target))
    return found


def choose(target: str, framework: str | None = None) -> tuple[Any, Any]:
    """The recipe that produces ``target``; ``framework`` picks among several."""
    known = targets()
    if target not in known:
        raise KeyError(
            f"no toolkit produces {target!r}; formats: {', '.join(['gguf', *sorted(known)])}"
        )
    options = [(r, t) for r, t in known[target] if framework in (None, r.framework)]
    if not options:
        makers = sorted({r.framework for r, _ in known[target]})
        raise KeyError(f"{framework} does not produce {target}; {', '.join(makers)} does")
    return options[0]


def output_gb(fx: ModelFacts, bpw: float, spec: Any) -> float | None:
    """The weights a toolkit writes: the body at the format's bits, the input embedding and
    an output head of its own at the bits the target says they keep. A head tied to the
    embedding is saved once, even where the source checkpoint stores it twice (F1's
    tied_head_stored)."""
    if not fx.params_total:
        return None
    extra = fx.extra
    body = fx.params_total - int(extra.get("tied_head_stored") or 0)
    vocab, hidden = extra.get("vocab_size"), extra.get("hidden_size")
    if not (vocab and hidden):
        return body * bpw / 8 / 1e9
    table = vocab * hidden
    head = table if extra.get("tie_word_embeddings") is False else 0
    bits = (
        (body - table - head) * bpw
        + table * (spec.embedding_bits or bpw)
        + head * (spec.head_bits or bpw)
    )
    return bits / 8 / 1e9


def plan_for(
    model: str, target: str, device: Device, *, framework: str | None = None, revision: str = "main"
) -> Plan:
    """A one-step plan that turns ``model`` into ``target``, with the output size predicted
    from the target's entry in quants/formats.yaml."""
    from rightsize.catalog import facts
    from rightsize.fit.formats import lookup

    recipe, spec = choose(target, framework)
    fx = facts(model, revision)
    bpw = lookup(spec.size_from).bpw if spec.size_from else None
    size = output_gb(fx, bpw, spec) if bpw else None
    notes = [
        f"{target}: about {bpw} bits per weight ({spec.size_from})"
        if bpw
        else f"{target}: no size model"
    ]
    if spec.embedding_bits:
        notes.append(f"the input embedding stays at {spec.embedding_bits:g} bits")
    notes.append("the quantizer's own memory use is not modelled yet")
    fit = FitResult(
        verdict=Verdict.fits,
        vram_gb=0.0,
        confidence=0.4,
        formula_id=FORMULA_ID,
        breakdown={"file_gb": round(size, 3)} if size else {},
        notes=notes,
    )
    # a converter that quantizes on the way (CTranslate2) is a quantize step in a plan
    stage = recipe.stage if recipe.stage == "export" else "quantize"
    step = PlanStep(
        stage=stage,
        framework=recipe.framework,
        device=device,
        quant=QuantSpec(method=target, variant=target, bits_per_weight=bpw),
        fit=fit,
        recipe_id=recipe.id,
    )
    trace = [f"{target} with {recipe.framework} ({recipe.id}, {recipe.verified})"]
    if size:
        trace.append(f"predicted output: {size:.2f} GB")
    return Plan(rank=1, model=fx, mode=Mode.infer, steps=[step], score=0.0, trace=trace)


def quantize_to(
    model: str,
    target: str,
    device: Device,
    *,
    framework: str | None = None,
    workdir: str | None = None,
    inputs: dict[str, Any] | None = None,
    install_missing: bool = False,
    dry_run: bool = False,
    revision: str = "main",
    log: Log = print,
    echo: Log | None = None,
) -> RunManifest:
    """Run the recipe that produces ``target`` on ``model``; see run_plan for the rest."""
    from rightsize.execution.runner import run_plan

    plan = plan_for(model, target, device, framework=framework, revision=revision)
    recipe, spec = choose(target, framework)
    slug = model.replace("/", "__")
    values: dict[str, Any] = {"model": model, **spec.inputs}
    if recipe.writes:
        values[recipe.writes[0]] = f"{slug}-{target}"
    values.update(inputs or {})
    return run_plan(
        plan,
        workdir=workdir,
        inputs=values,
        install_missing=install_missing,
        dry_run=dry_run,
        log=log,
        echo=echo,
    )
