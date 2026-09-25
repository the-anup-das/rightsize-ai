"""Turn a Plan into the commands that carry it out (F4, F5).

Each step names a recipe; the plan knows the model, the quantization and whether an
importance matrix comes first, which is enough to fill in every input with a sensible file
name. Anything passed by the caller wins, so the defaults never have to be right for
everyone - only runnable for someone who accepts them.
"""

from __future__ import annotations

from typing import Any

from rightsize.registry import get as get_recipe
from rightsize.registry import render
from rightsize.registry.schema import RenderedStep


def _defaults(plan: Any) -> dict[str, Any]:
    slug = plan.model.ref.repo.replace("/", "__")
    quant = next((s.quant.variant for s in plan.steps if s.quant and s.quant.variant), "Q4_K_M")
    return {
        # llama.cpp binaries, found on PATH or under .tools/llama.cpp
        "convert_script": "convert_hf_to_gguf.py",
        "quantize_bin": "llama-quantize",
        "imatrix_bin": "llama-imatrix",
        "perplexity_bin": "llama-perplexity",
        "python": "python",
        # files: a local snapshot of the Hub repo, then GGUFs beside it
        "model_dir": slug,
        "outfile": f"{slug}-f16.gguf",
        "outtype": "auto",
        "model_gguf": f"{slug}-f16.gguf",
        "input_gguf": f"{slug}-f16.gguf",
        "output_gguf": f"{slug}-{quant}.gguf",
        "calibration_file": "calibration_datav3.txt",
        "output_file": f"{slug}-imatrix.gguf",
        "quant": quant,
    }


def render_plan(plan: Any, **inputs: Any) -> list[RenderedStep]:
    values = _defaults(plan)
    has_imatrix = any(s.recipe_id == "llama.cpp/imatrix" for s in plan.steps)
    if has_imatrix:
        values["imatrix"] = values["output_file"]
    values.update(inputs)
    out: list[RenderedStep] = []
    for step in plan.steps:
        if not step.recipe_id:
            continue
        recipe = get_recipe(step.recipe_id)
        wanted = {k: v for k, v in values.items() if k in recipe.inputs}
        out.append(render(recipe, **wanted))
    return out
