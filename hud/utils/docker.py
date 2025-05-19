from typing import Optional, Sequence
from aiodocker.containers import DockerContainer
from aiodocker.stream import Stream
from aiohttp import ClientTimeout

from hud.utils.common import ExecuteResult

async def execute_command_in_container(
    container: DockerContainer,
    command: Sequence[str],
    timeout_secs: Optional[float] = 30.,
) -> ExecuteResult:
    """
    Execute a command in the container.

    Args:
        command: Command to execute
        workdir: Working directory for the command

    Returns:
        ExecuteResult: Result of the command execution
    """
    exec_result = await container.exec(
        cmd=command,
    )
    output: Stream = exec_result.start(timeout=ClientTimeout(timeout_secs), detach=False)

    stdout_data = bytearray()
    stderr_data = bytearray()

    while True:
        message = await output.read_out()
        if message is None:
            break
        if message.stream == 1:
            stdout_data.extend(message.data)
        elif message.stream == 2:
            stderr_data.extend(message.data)

    inspection = await exec_result.inspect()

    return ExecuteResult(
        stdout=bytes(stdout_data),
        stderr=bytes(stderr_data),
        exit_code=inspection.get("ExitCode", -1),
    )

