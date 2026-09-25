"""Minimal template renderer for recipes (F5). No jinja2 in the core.

Supported syntax, deliberately small:
  {{ name }}                     value substitution (paths are quoted when they contain spaces)
  {% if name %} ... {% endif %}  include the block when the value is truthy
  {% if not name %} ... {% endif %}
Blocks do not nest. Anything else is a template error at load time, not at run time.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from rightsize.registry.schema import Recipe, RenderedStep

_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_IF = re.compile(
    r"\{%\s*if\s+(not\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}(.*?)\{%\s*endif\s*%\}", re.DOTALL
)


class TemplateError(ValueError):
    pass


def validate_template(text: str) -> None:
    stripped = _IF.sub("", text)
    if "{%" in stripped or "%}" in stripped:
        raise TemplateError("unbalanced or nested {% %} block")
    if re.search(r"\{\{(?![^}]*\}\})", stripped):
        raise TemplateError("unterminated {{ }}")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value)
    return shlex.quote(s) if " " in s and not (s.startswith('"') and s.endswith('"')) else s


def render_text(template: str, values: dict[str, Any]) -> str:
    def _if(m: re.Match) -> str:
        negate, name, body = m.group(1), m.group(2), m.group(3)
        truthy = bool(values.get(name))
        return body if (truthy != bool(negate)) else ""

    out = _IF.sub(_if, template)

    def _var(m: re.Match) -> str:
        name = m.group(1)
        if name not in values or values[name] is None:
            raise TemplateError(f"missing value for {{{{ {name} }}}}")
        return _fmt(values[name])

    out = _VAR.sub(_var, out)
    return " ".join(out.split()) if "\n" not in template.strip() else out.strip()


def coerce_inputs(recipe: Recipe, given: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, spec in recipe.inputs.items():
        v = given.get(name, spec.default)
        if v is None:
            if spec.required:
                raise TemplateError(f"recipe {recipe.id}: input {name!r} is required")
            values[name] = None
            continue
        if spec.type == "enum" and spec.values and str(v) not in spec.values:
            raise TemplateError(f"recipe {recipe.id}: {name}={v!r} not in {spec.values}")
        if spec.type == "int":
            v = int(v)
        elif spec.type == "float":
            v = float(v)
        elif spec.type == "bool":
            v = bool(v)
        values[name] = v
    unknown = set(given) - set(recipe.inputs)
    if unknown:
        raise TemplateError(f"recipe {recipe.id}: unknown inputs {sorted(unknown)}")
    return values


def render(recipe: Recipe, **given: Any) -> RenderedStep:
    values = coerce_inputs(recipe, given)
    text = render_text(recipe.template, values)
    argv = shlex.split(text, posix=True) if recipe.kind == "command" else None
    return RenderedStep(
        recipe_id=recipe.id,
        framework=recipe.framework,
        stage=recipe.stage,
        kind=recipe.kind,
        text=text,
        argv=argv,
        install_line=recipe.install_line,
        notes=list(recipe.notes),
        source_doc_url=recipe.source_doc_url,
    )
