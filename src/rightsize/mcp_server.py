"""MCP server exposing the SDK as tools (F6). Requires the ``mcp`` extra.

Tools planned: recommend, estimate_memory, list_hardware, detect_hardware,
list_frameworks, render_recipe. Schemas come from the Pydantic types so the
SDK, CLI and MCP never drift. See docs/plans/F06-surfaces.md.
"""

from __future__ import annotations


def main() -> int:
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        from rightsize.errors import MissingExtraError

        raise MissingExtraError("mcp", "The Rightsize MCP server") from exc
    from rightsize.errors import NotImplementedYet

    raise NotImplementedYet("mcp_server", "docs/plans/F06-surfaces.md")
