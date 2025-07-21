#!/usr/bin/env python3
"""Universal client that connects to the HUD helper MCP server.

    # 1. start HTTP server exposing all tools
    python -m hud.tools.helper.mcp_server http

    # 2. run this client
    python examples/mcp_client.py http

Alternatively run both over stdio in a single process:

    python examples/mcp_client.py stdio
"""

from __future__ import annotations

import sys
import asyncio
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client, StdioServerParameters

HTTP_ENDPOINT = "http://localhost:8040/mcp"


async def run_http():
    async with streamablehttp_client(HTTP_ENDPOINT) as (r, w, _):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            await demo(sess)


async def run_stdio():
    params = StdioServerParameters(command="python", args=["-m", "hud.tools.helper.mcp_server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as sess:
            await sess.initialize()
            await demo(sess)


async def demo(sess: ClientSession):
    tools = await sess.list_tools()
    print("Tools:", [t.name for t in tools.tools])

    # Bash
    res = await sess.call_tool("bash", {"command": "echo hi && whoami"})
    print("Bash →", res.content[0].text.strip())

    # Computer variants
    for comp in ("computer", "computer_anthropic", "computer_openai"):
        if any(t.name == comp for t in tools.tools):
            if comp == "computer_openai":
                payload = {"type": "screenshot"}
            else:
                payload = {"action": "screenshot"}
            res = await sess.call_tool(comp, payload)
            imgs = [c for c in res.content if isinstance(c, types.ImageContent)]
            print(f"{comp}: {len(imgs)} image blocks returned")

    # Edit
    res = await sess.call_tool("edit_file", {"command": "view", "path": __file__})
    print("Edit snippet:", res.content[0].text.split("\n", 3)[-1][:120], "…")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "http"
    asyncio.run(run_http() if mode == "http" else run_stdio())
