"""A small, safe expression language for rule conditions (F4).

Rule conditions come from data files that anyone can edit, so they must never reach
``eval``. They are parsed with Python's own parser, which gives familiar syntax for free,
and then walked against a whitelist: comparisons, ``and`` / ``or`` / ``not``, ``in``,
dotted names and literals. Anything else - a call, a subscript, a lambda, an attribute
starting with an underscore - is refused when the rule is loaded, not when it runs.

    device.vendor == 'nvidia' and device.compute_capability < 8.9
    quant.method in ['fp8', 'nvfp4'] and not device.os == 'linux'

Missing data is neither true nor false. Conditions use three-valued logic, so an unknown
stays unknown through ``not``, ``and`` and ``or``, and a rule fires only when its condition
is definitely true: a rule about compute capability says nothing about a device whose
capability we do not know, and a block firing on missing data would hide good plans for
the wrong reason.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any

#: The names a condition may start from. Keeping this closed means a typo like
#: ``devcie.vendor`` is a load error instead of a rule that silently never fires.
ROOTS = frozenset({"device", "model", "quant", "runtime", "task", "stage", "mode", "plan"})

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class ExprError(ValueError):
    """A condition that is not allowed, raised at load time."""


class Condition:
    """A parsed, checked condition. Compile once at load; evaluate per candidate."""

    def __init__(self, source: str) -> None:
        self.source = source
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as exc:
            raise ExprError(f"cannot parse {source!r}: {exc.msg}") from exc
        _check(tree.body, source)
        self._tree = tree.body

    def __call__(self, ctx: Mapping[str, Any]) -> bool:
        """True only when the condition definitely holds; unknown does not fire."""
        return _truth(_eval(self._tree, ctx)) is True

    def truth(self, ctx: Mapping[str, Any]) -> Any:
        """True, False or UNKNOWN, for callers that want to say why a rule stayed silent."""
        return _truth(_eval(self._tree, ctx))

    def __repr__(self) -> str:
        return f"Condition({self.source!r})"


def _check(node: ast.AST, source: str) -> None:
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v, source)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _check(node.operand, source)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise ExprError(f"{source!r}: operator {type(op).__name__} is not allowed")
        _check(node.left, source)
        for c in node.comparators:
            _check(c, source)
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ExprError(f"{source!r}: attribute {node.attr!r} is not allowed")
        _check(node.value, source)
    elif isinstance(node, ast.Name):
        if node.id not in ROOTS and node.id not in ("True", "False", "None"):
            raise ExprError(f"{source!r}: unknown name {node.id!r}; use one of {sorted(ROOTS)}")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, str | int | float | bool | type(None)):
            raise ExprError(f"{source!r}: literal {node.value!r} is not allowed")
    elif isinstance(node, ast.List | ast.Tuple | ast.Set):
        for e in node.elts:
            if not isinstance(e, ast.Constant):
                raise ExprError(f"{source!r}: collections may hold literals only")
            _check(e, source)
    else:
        raise ExprError(f"{source!r}: {type(node).__name__} is not allowed in a condition")


_MISSING = object()


class _Unknown:
    """Kleene's third truth value. A comparison against missing data is neither true nor
    false, and it has to stay that way through ``not``: with plain booleans,
    ``not device.os == 'macos'`` came out True for a device whose OS we did not know, so a
    negated block rule would have fired on an absence of information."""

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()
_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _truth(value: Any) -> Any:
    if value is UNKNOWN or value is _MISSING:
        return UNKNOWN
    return bool(value)


def _eval(node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_truth(_eval(v, ctx)) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(v is False for v in values):
                return False
            return UNKNOWN if any(v is UNKNOWN for v in values) else True
        if any(v is True for v in values):
            return True
        return UNKNOWN if any(v is UNKNOWN for v in values) else False
    if isinstance(node, ast.UnaryOp):
        v = _truth(_eval(node.operand, ctx))
        return UNKNOWN if v is UNKNOWN else not v
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = _eval(right_node, ctx)
            if left is _MISSING or right is _MISSING:
                return UNKNOWN
            if isinstance(op, _ORDERING) and (left is None or right is None):
                return UNKNOWN
            try:
                if not _COMPARE[type(op)](left, right):
                    return False
            except TypeError:  # a string against a number, or "in" a scalar: odd data
                return UNKNOWN
            left = right
        return True
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, ctx)
        if base is _MISSING or base is None:
            return _MISSING
        if isinstance(base, Mapping):
            return base.get(node.attr, _MISSING)
        return getattr(base, node.attr, _MISSING)
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        return ctx.get(node.id, _MISSING)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return [e.value for e in node.elts if isinstance(e, ast.Constant)]
    raise ExprError(f"cannot evaluate {type(node).__name__}")  # _check makes this unreachable
