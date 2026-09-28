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

import datetime as _dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rightsize.types import (
    Device,
    FitResult,
    Measurement,
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
    """The weights a target's toolkit writes; the fit engine's format_weights_gb."""
    from rightsize.fit.llm import format_weights_gb

    return format_weights_gb(fx, bpw, spec.embedding_bits, spec.head_bits, spec.bits_over)


def plan_for(
    model: str,
    target: str,
    device: Device,
    *,
    framework: str | None = None,
    revision: str = "main",
    evaluate: bool = False,
    bits: float | None = None,
) -> Plan:
    """A one-step plan that turns ``model`` into ``target``, with the output size predicted
    from the target's entry in quants/formats.yaml, or from ``bits`` for a target whose bits
    are one of its inputs. ``evaluate`` adds the target's gate as a second step, run in the
    same toolkit's environment."""
    from rightsize.catalog import facts
    from rightsize.fit.formats import lookup

    recipe, spec = choose(target, framework)
    fx = facts(model, revision)
    bpw = bits if spec.bits_from_input else (lookup(spec.size_from).bpw if spec.size_from else None)
    size = output_gb(fx, bpw, spec) if bpw else None
    notes = [
        f"{target}: about {bpw} bits per weight ({spec.bits_from_input or spec.size_from})"
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
    steps = [step]
    trace = [f"{target} with {recipe.framework} ({recipe.id}, {recipe.verified})"]
    if size:
        trace.append(f"predicted output: {size:.2f} GB")
    if evaluate:
        if spec.gate is None:
            raise ValueError(f"{target} has no quality gate yet: {recipe.id} names none")
        steps.append(
            PlanStep(
                stage="evaluate",
                framework=recipe.framework,
                device=device,
                quant=step.quant,
                fit=fit,
                recipe_id=spec.gate.recipe,
            )
        )
        trace.append(f"gate: {spec.gate.recipe}, run in {recipe.framework}'s environment")
    return Plan(rank=1, model=fx, mode=Mode.infer, steps=steps, score=0.0, trace=trace)


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
    evaluate: bool = False,
    eval_chunks: int = 100,
    ctx: int = 8192,
    revision: str = "main",
    log: Log = print,
    echo: Log | None = None,
) -> RunManifest:
    """Run the recipe that produces ``target`` on ``model``; see run_plan for the rest.
    ``evaluate`` runs the target's gate on the output afterwards and records its verdict in
    the manifest, as the GGUF pipeline does. A target whose bits are one of its inputs
    (modelopt-auto) gets them from what ``device`` has room for at ``ctx``, unless
    ``inputs`` names them."""
    from rightsize.execution.runner import run_plan

    recipe, spec = choose(target, framework)
    inputs = dict(inputs or {})
    bits: float | None = None
    if spec.bits_from_input:
        if spec.bits_from_input in inputs:
            bits = float(inputs[spec.bits_from_input])
        else:
            from rightsize.catalog import facts
            from rightsize.fit.llm import bits_that_fit

            runtime = (
                "vllm"
                if spec.serve_format in ("modelopt", "compressed-tensors", "fp8")
                else "llama.cpp"
            )
            bits, parts = bits_that_fit(facts(model, revision), device, ctx=ctx, runtime=runtime)
            if bits is not None and spec.bits_over == "linear":
                bits = parts["bits_linear"]
            if bits is None:
                raise ValueError(
                    f"{device.name} has no room for {model} at ctx {ctx} even with the weights "
                    f"at 0 bits: KV cache {parts['kv_cache']} GB, overhead {parts['overhead']} GB"
                )
            bits = min(bits, 16.0)
            inputs[spec.bits_from_input] = bits
            log(
                f"{spec.bits_from_input} {bits:g}: what {device.name} has room for at ctx {ctx} "
                f"with {runtime} ({parts['weights_max_gb']} GB for the weights after "
                f"{parts['kv_cache']} GB of KV cache and {parts['overhead']} GB of overhead)"
            )
    plan = plan_for(
        model, target, device, framework=framework, revision=revision, evaluate=evaluate, bits=bits
    )
    slug = model.replace("/", "__")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(workdir or Path("runs") / f"{slug}-{stamp}")
    values: dict[str, Any] = {"model": model, **spec.inputs}
    out = f"{slug}-{target}"
    if recipe.writes:
        values[recipe.writes[0]] = out
    if evaluate:
        from rightsize.execution.quantize import wikitext

        values.update(
            {
                "candidate": out,
                "modelopt_state": "modelopt_state.pth",  # what modelopt/ptq writes beside it
                # absolute: the script runs inside the run folder
                "eval_file": "wiki.test.raw" if dry_run else str(wikitext(log).resolve()),
                "chunks": eval_chunks,
                "gate_file": "gate.json",
                **spec.gate.inputs,
            }
        )
    values.update(inputs)
    manifest = run_plan(
        plan,
        workdir=run_dir,
        inputs=values,
        install_missing=install_missing,
        dry_run=dry_run,
        log=log,
        echo=echo,
    )
    if evaluate and not dry_run:
        record_gate(manifest, target, run_dir / str(values["gate_file"]), log)
    return manifest


def record_gate(manifest: RunManifest, target: str, path: Path, log: Log) -> None:
    """Read the numbers the gate recipe wrote, judge them against quality/gate_thresholds.yaml
    and put both into the manifest, beside the run that produced them."""
    from rightsize.execution.llamacpp import gate, write_manifest

    numbers = {
        k: float(v)
        for k, v in json.loads(path.read_text(encoding="utf-8")).items()
        if isinstance(v, int | float)
    }
    manifest.gate[target] = gate(numbers)
    step = manifest.steps[-1]
    for kind in ("kld_mean", "top1_agreement", "ppl", "cosine_mean"):
        if kind in numbers:
            step.measurements.append(Measurement(kind=kind, value=numbers[kind], note=target))
    write_manifest(manifest, path.parent)
    keys = ("kld_mean", "top1_agreement", "ppl", "ppl_base", "cosine_mean", "cosine_min")
    shown = ", ".join(f"{k} {numbers[k]:.4g}" for k in keys if k in numbers)
    log(f"gate {manifest.gate[target]['verdict']} for {target}: {shown}")
