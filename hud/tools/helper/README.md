# HUD Helper Package

This sub-package bundles utilities that make it trivial to expose HUD
Python tool classes as **Model Context Protocol (MCP)** tools.

## Contents

| File | Purpose |
|------|---------|
| `utils.py` | `register_instance_tool` – wrap a class instance into a FastMCP tool with auto-generated JSON schema |
| `mcp_server.py` | CLI server (stdio/HTTP). Tool names: `computer`, `computer_anthropic`, `computer_openai`, `bash`, `edit_file` |

## Quick start

### 1 — Run a server (stdio)
```bash
python -m hud.tools.helper.mcp_server               # exposes all tools on stdio
```

### 2 — Run a server (HTTP)
```bash
python -m hud.tools.helper.mcp_server http --port 8040 \
       --tools computer bash   # expose only two tools
```
This starts a Streamable-HTTP MCP server at `http://localhost:8040/mcp`.

### 3 — From a client
```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://localhost:8040/mcp") as (r, w, _):
    async with ClientSession(r, w) as sess:
        await sess.initialize()
        res = await sess.call_tool("bash", {"command": "echo hi"})
        print(res.content[0].text)
```

## Advanced: registering custom tools

```python
from mcp.server.fastmcp import FastMCP
from hud.tools.helper import register_instance_tool

class MyTool:
    async def __call__(self, name: str) -> str:   # type-hints generate schema!
        return f"Hello {name}!"

mcp = FastMCP("Custom")
register_instance_tool(mcp, "my_tool", MyTool())

mcp.run(transport="stdio")
```

The helper inspects `MyTool.__call__`, removes `*args/**kwargs`, and FastMCP
automatically derives an input schema and registers the tool. 