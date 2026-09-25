"""The MCP surface: tools answer, and their input schemas do not change by accident (F6)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from rightsize.mcp_server import build_server  # noqa: E402

SNAPSHOT = Path(__file__).parent / "fixtures" / "mcp_tools.json"


def _call(name: str, args: dict) -> object:
    """A tool's result as the client sees it. Structured content when the SDK gives it (2.x
    wraps non-dict returns as {"result": ...}); otherwise the text blocks, one per item."""

    async def go():
        result = await build_server().call_tool(name, args)
        structured = getattr(result, "structured_content", None) or getattr(
            result, "structuredContent", None
        )
        if isinstance(structured, dict):
            return structured.get("result", structured)
        if structured is not None:
            return structured
        items = [json.loads(block.text) for block in result.content]
        return items[0] if len(items) == 1 else items

    return asyncio.run(go())


def _tool_schemas() -> dict:
    """input_schema in the MCP SDK 2.x, inputSchema in 1.x."""

    async def go():
        return {
            t.name: getattr(t, "input_schema", None) or t.inputSchema
            for t in await build_server().list_tools()
        }

    return asyncio.run(go())


def test_tool_input_schemas_match_the_snapshot() -> None:
    """An agent's saved calls break if a parameter is renamed or retyped, so the schemas are
    pinned. Set RIGHTSIZE_UPDATE_SNAPSHOTS=1 to accept a deliberate change."""
    current = _tool_schemas()
    if os.environ.get("RIGHTSIZE_UPDATE_SNAPSHOTS") or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert current == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_the_six_planned_tools_are_there() -> None:
    names = set(_tool_schemas())
    assert {"recommend", "estimate_memory", "list_hardware", "detect_hardware",
            "list_frameworks", "render_recipe"} <= names


def test_recommend_returns_plans_with_their_commands() -> None:
    out = _call("recommend", {"task": "chat", "device": "RTX 3060 12GB", "top_k": 2})
    assert len(out["plans"]) == 2
    first = out["plans"][0]
    assert first["rank"] == 1 and first["trace"]
    assert any(c["text"].startswith("llama-server") for c in first["commands"])
    assert all(c["verified"] in ("run", "help", "docs") for c in first["commands"])


def test_list_hardware_filters() -> None:
    out = _call("list_hardware", {"query": "4090"})
    assert out and all("4090" in d["name"] for d in out)


def test_render_recipe() -> None:
    out = _call("render_recipe", {"recipe_id": "llama.cpp/server",
                                  "inputs": {"server_bin": "llama-server", "model_gguf": "m.gguf"}})
    assert out["text"].startswith("llama-server -m m.gguf")
