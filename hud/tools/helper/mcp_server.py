#!/usr/bin/env python3
"""Parameterised FastMCP server for HUD tools.

Usage
-----
Run with default (stdio, all tools):

    python -m hud.tools.helper.mcp_server

Streamable HTTP on :8040 exposing computer + bash only:

    python -m hud.tools.helper.mcp_server http --tools computer bash

Arguments
~~~~~~~~~
transport   stdio (default) | http
--tools     list of tool names to expose (default = all)
--port      HTTP port (default 8040)
"""

from __future__ import annotations

import argparse
import logging

from mcp.server.fastmcp import FastMCP

from hud.tools.bash import BashTool
from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.computer.openai import OpenAIComputerTool
from hud.tools.edit import EditTool

from .utils import register_instance_tool

TOOL_MAP = {
    "computer": HudComputerTool,
    "computer_anthropic": AnthropicComputerTool,
    "computer_openai": OpenAIComputerTool,
    "bash": BashTool,
    "edit_file": EditTool,
}

logging.basicConfig(level=logging.INFO)


def build_server(names: list[str] | None = None, *, port: int = 8040) -> FastMCP:
    server = FastMCP("HUD", port=port)
    selected = names or list(TOOL_MAP.keys())

    for name in selected:
        cls = TOOL_MAP.get(name)
        if cls is None:
            raise SystemExit(f"Unknown tool '{name}'. Choices: {list(TOOL_MAP)}")
        register_instance_tool(server, name, cls())
    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="hud-mcp", description="Run HUD FastMCP server")
    parser.add_argument("transport", nargs="?", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--tools", nargs="*", help="Tool names to expose (default: all)")
    parser.add_argument("--port", type=int, default=8040, help="HTTP port (default 8040)")
    args = parser.parse_args()

    mcp = build_server(args.tools, port=args.port)

    if args.transport == "http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
