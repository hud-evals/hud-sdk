from __future__ import annotations

import asyncio
import inspect

import pytest

from hud.tools.bash import BashTool
from hud.tools.computer.hud import HudComputerTool
from hud.tools.edit import EditTool
from hud.tools.helper import register_instance_tool


@pytest.mark.asyncio
async def test_bash_tool_echo():
    tool = BashTool()

    # Monkey-patch the private _session methods so no subprocess is spawned
    class _FakeSession:
        async def run(self, cmd: str):
            from hud.tools.base import ToolResult

            return ToolResult(output=f"mocked: {cmd}")

        async def start(self):
            return None

    tool._session = _FakeSession()  # type: ignore[attr-defined]

    result = await tool(command="echo hello")
    assert result.output == "mocked: echo hello"


@pytest.mark.asyncio
async def test_bash_tool_restart_and_no_command():
    from hud.tools.base import ToolError, ToolResult

    tool = BashTool()

    class _FakeSession:
        async def run(self, cmd: str):
            return ToolResult(output="ran")

        async def start(self):
            return None

        def stop(self):
            return None

    tool._session = _FakeSession()  # type: ignore[attr-defined]

    # Monkey-patch _BashSession.start to avoid launching a real shell
    async def _dummy_start(self):
        self._started = True
        from types import SimpleNamespace

        # minimal fake process attributes used later
        self._process = SimpleNamespace(returncode=None)

    import hud.tools.bash as bash_mod

    bash_mod._BashSession.start = _dummy_start  # type: ignore[assignment]

    # restart=True returns system message
    res = await tool(command="ignored", restart=True)
    assert res.system == "tool has been restarted."

    # Calling without command raises ToolError
    with pytest.raises(ToolError):
        await tool()


@pytest.mark.asyncio
async def test_edit_tool_flow(tmp_path):
    file_path = tmp_path / "demo.txt"

    edit = EditTool()

    # create
    res = await edit(command="create", path=str(file_path), file_text="hello\nworld\n")
    assert "File created" in (res.output or "")

    # view
    res = await edit(command="view", path=str(file_path))
    assert "hello" in (res.output or "")

    # replace
    res = await edit(command="str_replace", path=str(file_path), old_str="world", new_str="earth")
    assert "has been edited" in (res.output or "")

    # insert
    res = await edit(command="insert", path=str(file_path), insert_line=1, new_str="first line\n")
    assert res


@pytest.mark.asyncio
async def test_base_executor_simulation():
    from hud.tools.executors.base import BaseExecutor

    exec = BaseExecutor()
    res = await exec.execute("echo test")
    assert "SIMULATED" in (res.output or "")
    shot = await exec.screenshot()
    assert isinstance(shot, str) and len(shot) > 0


@pytest.mark.asyncio
async def test_edit_tool_view(tmp_path):
    # Create a temporary file
    p = tmp_path / "sample.txt"
    p.write_text("Sample content\n")

    tool = EditTool()
    result = await tool(command="view", path=str(p))
    assert result.output is not None
    assert "Sample content" in result.output


@pytest.mark.asyncio
async def test_computer_tool_screenshot():
    comp = HudComputerTool()
    blocks = await comp(action="screenshot")
    assert any(getattr(b, "data", None) for b in blocks)


def test_register_instance_tool_signature():
    """Helper should expose same user-facing parameters (no *args/**kwargs)."""

    class Dummy:
        async def __call__(self, *, x: int, y: str) -> str:
            return f"{x}-{y}"

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("test")
    fn = register_instance_tool(mcp, "dummy", Dummy())
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    assert [p.name for p in params] == ["x", "y"], "*args/**kwargs should be stripped"


def test_build_server_subset():
    """Ensure build_server registers only requested tools."""
    from hud.tools.helper.mcp_server import build_server

    mcp = build_server(["bash"])
    names = [t.name for t in asyncio.run(mcp.list_tools())]
    assert names == ["bash"]
