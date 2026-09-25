"""The MCP server over a real stdio pipe, the way an editor or desktop client runs it.

Slow (it starts a subprocess), so opt-in: uv run pytest -m slow. In-process tool calls do
not prove the transport works; this does.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

pytest.importorskip("mcp")
pytestmark = pytest.mark.slow


def test_stdio_handshake_tools_and_a_plan() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    import rightsize

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "rightsize.cli", "mcp"])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
            tools = await session.list_tools()
            result = await session.call_tool("recommend", {"device": "RTX 4090", "top_k": 1})
            sc = getattr(result, "structured_content", None) or getattr(
                result, "structuredContent", None
            )
            return info, len(tools.tools), sc or json.loads(result.content[0].text)

    info, n_tools, payload = asyncio.run(run())
    assert info.name == "rightsize" and info.version == rightsize.__version__
    assert n_tools >= 6
    assert payload["plans"][0]["rank"] == 1
