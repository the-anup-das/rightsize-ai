"""Generate docs/frameworks/*.md from the recipe registry (F5).

    uv run python -m rightsize.registry.docs

One page per framework, built from the same recipes rightsize renders, so the docs cannot
describe a command the tool would not produce. tests/test_framework_docs.py fails when the
pages fall behind the recipes.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from rightsize.registry.loader import all_recipes
from rightsize.registry.schema import Recipe

_VERIFIED = {
    "run": "run end to end",
    "help": "flags checked against the tool's --help",
    "docs": "from the documentation; not yet run here",
}


def _example(recipe: Recipe) -> str:
    from rightsize.registry.render import render_text

    values = {}
    for name, spec in recipe.inputs.items():
        if spec.default is not None:
            values[name] = spec.default
        elif spec.values:
            values[name] = spec.values[0]
        elif spec.required:
            values[name] = f"<{name}>"
    return render_text(recipe.template, values)


def page(framework: str, recipes: list[Recipe]) -> str:
    lines = [f"# {framework}", "", "Generated from `data/recipes/` - do not edit by hand.", ""]
    lines += ["| recipe | stage | families | checked | version |", "|---|---|---|---|---|"]
    for r in recipes:
        lines.append(
            f"| `{r.id}` | {r.stage} | {', '.join(r.families)} | {r.verified} | "
            f"{r.version_tested or ''} |"
        )
    for r in recipes:
        lines += ["", f"## `{r.id}`", ""]
        lines.append(f"{r.stage.capitalize()} step; {_VERIFIED[r.verified]}"
                     + (f" at {r.version_tested}." if r.version_tested else "."))
        if r.install_line:
            lines += ["", f"Install: {r.install_line}"]
        fence = "bash" if r.kind == "command" else (r.language or "")
        lines += ["", "```" + fence, _example(r), "```", ""]
        if r.inputs:
            lines += ["| input | type | default | notes |", "|---|---|---|---|"]
            for name, spec in r.inputs.items():
                default = "required" if spec.required and spec.default is None else spec.default
                notes = spec.help or (", ".join(spec.values) if spec.values else "")
                lines.append(f"| `{name}` | {spec.type} | {default if default is not None else ''}"
                             f" | {notes} |")
        if r.notes:
            lines += [""] + [f"- {n}" for n in r.notes]
        lines += ["", f"Source: <{r.source_doc_url}>"]
    return "\n".join(lines) + "\n"


def pages() -> dict[str, str]:
    by_framework: dict[str, list[Recipe]] = defaultdict(list)
    for r in all_recipes().values():
        by_framework[r.framework].append(r)
    out = {}
    for fw, recipes in sorted(by_framework.items()):
        out[f"{fw}.md"] = page(fw, sorted(recipes, key=lambda r: r.id))
    index = ["# Frameworks", "", "Generated from `data/recipes/` - do not edit by hand.", "",
             "| framework | recipes | stages |", "|---|---|---|"]
    for fw, recipes in sorted(by_framework.items()):
        stages = ", ".join(sorted({r.stage for r in recipes}))
        index.append(f"| [{fw}]({fw}.md) | {len(recipes)} | {stages} |")
    out["README.md"] = "\n".join(index) + "\n"
    return out


def main(out_dir: str = "docs/frameworks") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in pages().items():
        (out / name).write_text(text, encoding="utf-8")
        sys.stdout.write(f"wrote {out / name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
