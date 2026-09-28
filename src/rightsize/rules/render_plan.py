"""Turn a Plan into the commands that carry it out (F4, F5).

Each step names a recipe; the plan knows the model, the quantization and whether an
importance matrix comes first, which is enough to fill in every input with a sensible file
name. The names each toolkit expects (its binaries, its output files) come from its
framework.yaml ``defaults``, so a new toolkit brings its own. Anything passed by the caller
wins, so the defaults never have to be right for everyone - only runnable for someone who
accepts them.
"""

from __future__ import annotations

from typing import Any

from rightsize.registry import framework_infos, render
from rightsize.registry import get as get_recipe
from rightsize.registry.schema import RenderedStep


def _defaults(plan: Any) -> dict[str, Any]:
    """The plan's quantization, and each framework's own defaults for the values its
    recipes read (file names, binaries), the first framework in the plan first."""
    slug = plan.model.ref.repo.replace("/", "__")
    # the GGUF type from the quantize step: a fine-tune step's quant (QLoRA's nf4) comes
    # first in a two-stage plan and is not a llama.cpp type
    quant = next(
        (
            s.quant.variant
            for s in plan.steps
            if s.quant and s.quant.method == "gguf" and s.quant.variant
        ),
        "Q4_K_M",
    )
    name = plan.model.ref.repo.rsplit("/", 1)[-1].lower()
    values: dict[str, Any] = {"quant": quant}
    infos = framework_infos()
    for step in plan.steps:
        info = infos.get(step.framework)
        for key, template in (info.defaults if info else {}).items():
            values.setdefault(
                key,
                template.replace("{slug}", slug).replace("{quant}", quant).replace("{name}", name),
            )
    return values


def _finetune_values(plan: Any, values: dict[str, Any]) -> None:
    """Inputs for a fine-tune step, and the directory the convert step reads after it.

    llama.cpp converts a whole 16-bit model. Unsloth writes one (merged_dir); the other
    trainers write an adapter, and their recipes say to merge it into merged_dir first, so
    the convert step reads the same directory whichever trainer ran."""
    ft = next((s for s in plan.steps if s.stage == "finetune" and s.recipe_id), None)
    if ft is None:
        return
    slug = plan.model.ref.repo.replace("/", "__")
    qlora = ft.quant is not None
    values.update(
        {
            "model": plan.model.ref.repo,
            "load_in_4bit": qlora,
            "adapter": "qlora" if qlora else "lora",
        }
    )
    # where things land; a trainer's framework.yaml can say otherwise (Axolotl merges into
    # <output_dir>/merged)
    values.setdefault("output_dir", f"{slug}-finetune")
    values.setdefault("merged_dir", f"{slug}-merged")
    values.setdefault("adapter_dir", f"{slug}-adapters")
    values["model_dir"] = values["merged_dir"]


def _typed(recipe: Any, name: str, value: Any) -> Any:
    """A value in the form the recipe declares: Python recipes spell booleans True/False."""
    spec = recipe.inputs[name]
    if isinstance(value, bool) and spec.type == "enum" and spec.values:
        return next((v for v in spec.values if v.lower() == str(value).lower()), value)
    return value


def render_plan(plan: Any, **inputs: Any) -> list[RenderedStep]:
    return [render(recipe, **wanted) for _step, recipe, wanted in step_values(plan, **inputs)]


def step_values(plan: Any, **inputs: Any) -> list[tuple[Any, Any, dict[str, Any]]]:
    """(plan step, recipe, the inputs it renders with) for every step that has a recipe:
    what render_plan renders, and what a run needs to find each step's outputs."""
    values = _defaults(plan)
    _finetune_values(plan, values)
    serve = next((s for s in plan.steps if s.stage == "serve"), None)
    if serve is not None and serve.runtime and serve.runtime.ctx:
        values["ctx"] = serve.runtime.ctx
    values.update(inputs)
    # llama.cpp's chain: the server loads the quantized file, and llama-quantize reads the
    # importance matrix the step before wrote. Set after the caller's values, so renaming an
    # output renames what reads it; a plan without llama.cpp has neither.
    recipe_ids = {s.recipe_id for s in plan.steps}
    if "output_gguf" in values:
        values.setdefault("model_gguf_served", values["output_gguf"])
    if "llama.cpp/imatrix" in recipe_ids and "output_file" in values:
        values.setdefault("imatrix", values["output_file"])
    trains_on = _trains_on(plan, recipe_ids)
    out: list[tuple[Any, Any, dict[str, Any]]] = []
    for step in plan.steps:
        if not step.recipe_id:
            continue
        recipe = get_recipe(step.recipe_id)
        wanted = {k: _typed(recipe, k, v) for k, v in values.items() if k in recipe.inputs}
        if step.stage == "serve" and "model_gguf" in recipe.inputs:
            wanted["model_gguf"] = values.get("model_gguf_served", wanted.get("model_gguf"))
        if (
            trains_on
            and trains_on in values
            and "model" in recipe.inputs
            and step.stage in ("finetune", "export")
        ):
            # the model the step before produced: MLX trains, and fuses, on its 4-bit copy
            wanted["model"] = values[trains_on]
        out.append((step, recipe, wanted))
    return out


def _trains_on(plan: Any, recipe_ids: set[str | None]) -> str | None:
    """The value a fine-tune step reads as its model when its trainer ran a step first:
    MLX trains QLoRA on the model it just quantized (framework.yaml, finetune.before)."""
    ft = next((s for s in plan.steps if s.stage == "finetune" and s.recipe_id), None)
    info = framework_infos().get(ft.framework) if ft else None
    before = info.finetune.before.get(plan.mode.value) if info and info.finetune else None
    return before.model_from if before and before.recipe in recipe_ids else None
