from __future__ import annotations

import asyncio
import os
from typing import Any

from .base import CLIResult, ToolError, ToolResult


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False

    async def start(self) -> None:
        if self._started:
            await asyncio.sleep(0)
            return

        self._process = await asyncio.create_subprocess_shell(  # noqa: S604
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str) -> CLIResult:
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            await asyncio.sleep(0)
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash did not return in {self._timeout} seconds and must be restarted",
            ) from None

        if self._process.stdin is None:
            raise ToolError("stdin is None")
        if self._process.stdout is None:
            raise ToolError("stdout is None")
        if self._process.stderr is None:
            raise ToolError("stderr is None")

        # Send command to the process
        self._process.stdin.write(command.encode() + f"; echo '{self._sentinel}'\n".encode())
        await self._process.stdin.drain()

        # Read output from the process, until the sentinel is found
        sentinel_line = f"{self._sentinel}\n"
        sentinel_bytes = sentinel_line.encode()

        try:
            raw_out: bytes = await asyncio.wait_for(
                self._process.stdout.readuntil(sentinel_bytes),
                timeout=self._timeout,
            )
            output = raw_out.decode()[: -len(sentinel_line)]
        except (asyncio.TimeoutError, asyncio.LimitOverrunError):
            self._timed_out = True
            raise ToolError(
                f"timed out: bash did not return in {self._timeout} seconds and must be restarted",
            ) from None

        # Attempt non-blocking stderr fetch (may return empty)
        try:
            error_bytes = await asyncio.wait_for(self._process.stderr.read(), timeout=0.01)
            error = error_bytes.decode().rstrip("\n")
        except asyncio.TimeoutError:
            error = ""

        return CLIResult(output=output, error=error)


class BashTool:
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    _session: _BashSession | None

    def __init__(self) -> None:
        self._session = None

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs: Any
    ) -> ToolResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")
