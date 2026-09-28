"""Generate docs/frameworks/*.md from the recipe registry (F5).

    uv run python -m rightsize.registry.docs

One page per framework, built from its framework.yaml and the same recipes rightsize
renders, so the docs cannot describe a command the tool would not produce.
tests/test_framework_docs.py fails when the pages fall behind the data.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from rightsize.registry.loader import all_recipes, framework_infos
from rightsize.registry.schema import FrameworkInfo, Recipe

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


def _runs_on(info: FrameworkInfo) -> str:
    hw = info.hardware
    parts = [", ".join(v for v in hw.get("vendors", []) if v != "any") or "any hardware"]
    if hw.get("os"):
        parts.append("on " + ", ".join(hw["os"]))
    if hw.get("min_compute_capability"):
        parts.append(f"NVIDIA compute capability {hw['min_compute_capability']}+")
    return "; ".join(parts)


def _about(info: FrameworkInfo) -> list[str]:
    lines = [info.summary, ""]
    lines.append(f"- Runs on: {_runs_on(info)}")
    if info.formats:
        lines.append(f"- Writes: {', '.join(info.formats)}")
    lines.append(f"- Install: `{info.install.line}`"
                 + (f" ({info.install.note})" if info.install.note else ""))
    if info.finetune:
        role = info.finetune
        where = [v for v in role.default_for if v != "*"]
        default = (f"the default trainer on {', '.join(where)}"
                   + (" and wherever no other framework is" if "*" in role.default_for else "")
                   if role.default_for else "not a default trainer; choose it by name")
        lines.append(f"- Fine-tunes: {', '.join(role.modes)}; {default}")
    lines.append(f"- Home: <{info.homepage}>")
    return lines + [""]


def page(framework: str, recipes: list[Recipe]) -> str:
    info = framework_infos().get(framework)
    title = info.title if info else framework
    lines = [f"# {title}", "", "Generated from `data/recipes/` - do not edit by hand.", ""]
    if info:
        lines += _about(info)
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
        if r.targets:
            names = " or ".join(f"`--to {t.name}`" for t in r.targets)
            lines += ["", f"`rightsize quantize MODEL` runs it with {names}."]
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
    infos = framework_infos()
    index = ["# Frameworks", "", "Generated from `data/recipes/` - do not edit by hand.", "",
             "Each framework is a folder in `data/recipes/`: a `framework.yaml` saying what it "
             "is for, where it runs, how it installs and how its steps join a plan, and one "
             "YAML file per recipe. A plugin package adds a framework the same way; see "
             "CONTRIBUTING.md.", "",
             "| framework | recipes | stages | runs on |", "|---|---|---|---|"]
    for fw, recipes in sorted(by_framework.items()):
        stages = ", ".join(sorted({r.stage for r in recipes}))
        runs_on = _runs_on(infos[fw]) if fw in infos else ""
        index.append(f"| [{fw}]({fw}.md) | {len(recipes)} | {stages} | {runs_on} |")
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
