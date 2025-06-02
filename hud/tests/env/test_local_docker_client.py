from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import hud
from hud.env.local_docker_client import (
    LocalDockerClient,
    _stream_logs,
    wait_until_healthy_container,
)


async def fake_log_generator(lines: list[str]):
    for line in lines:
        await asyncio.sleep(0.01)
        yield line


async def error_log_generator():
    await asyncio.sleep(0.01)
    raise Exception("Simulated log error")


@pytest.mark.asyncio
async def test_stream_logs_normal(mocker, caplog):
    caplog.set_level("INFO")
    log_lines = ["Log line 1", "Log line 2", "Log line 3"]

    container = MagicMock()
    container.id = "abc123def456"
    container.log = MagicMock(return_value=fake_log_generator(log_lines))

    await _stream_logs(container)

    for expected in log_lines:
        assert f"container abc123def456 | {expected}" in caplog.text


@pytest.mark.asyncio
async def test_wait_healthy_container(mocker):
    container = MagicMock()
    container.id = "abc123"
    container.show = AsyncMock(
        side_effect=[
            {"Config": {"Healthcheck": {}}, "State": {"Health": {"Status": "starting"}}},
            {"Config": {"Healthcheck": {}}, "State": {"Health": {"Status": "starting"}}},
            {"Config": {"Healthcheck": {}}, "State": {"Health": {"Status": "healthy"}}},
        ]
    )

    await wait_until_healthy_container(container, timeout=5)
    assert container.show.call_count >= 3


@pytest.mark.asyncio
async def test_container_crashes_before_healthy():
    container = MagicMock()
    container.id = "abc123"
    container.show = AsyncMock(
        side_effect=[
            {
                "Config": {"Healthcheck": {}},
                "State": {"Health": {"Status": "starting"}, "Status": "running"},
            },
            {
                "Config": {"Healthcheck": {}},
                "State": {"Health": {"Status": "starting"}, "Status": "exited"},
            },
        ]
    )

    with pytest.raises(RuntimeError, match="crashed"):
        await wait_until_healthy_container(container, timeout=5)


@pytest.mark.asyncio
async def test_container_starting_times_out():
    container = MagicMock()
    container.id = "abc123"
    container.show = AsyncMock(
        return_value={
            "Config": {"Healthcheck": {}},
            "State": {"Health": {"Status": "starting"}, "Status": "running"},
        }
    )

    with pytest.raises(TimeoutError, match="not healthy"):
        await wait_until_healthy_container(container, timeout=1)  # small timeout for speed


class MockContainer:
    def __init__(self, id: str):
        self.id = id
        self.log_lines = ["Log line 1", "Log line 2"]

    async def show(self):
        await asyncio.sleep(0.5)  # Simulate async behavior
        return {
            "Config": {"Healthcheck": {"Interval": 1_000_000}},
            "State": {"Health": {"Status": "healthy"}, "Status": "running"},
        }

    def log(self, **kwargs):
        return fake_log_generator(self.log_lines)

    async def start(self):
        await asyncio.sleep(0.5)  # Simulate async start

    def check_logs(self, caplog):
        for line in self.log_lines:
            assert f"container {self.id} | {line}" in caplog.text


@pytest.mark.asyncio
async def test_create_local_docker_client(mocker, caplog):
    caplog.set_level("INFO")
    container = MockContainer(id="123abc456def")
    image = "test-image:latest"
    host_config = {"NetworkMode": "host"}
    container_config = {
        "Image": image,
        "Tty": True,
        "OpenStdin": True,
        "Cmd": None,
        "HostConfig": host_config,
    }

    mock_create = mocker.patch(
        "hud.env.local_docker_client.aiodocker.containers.DockerContainers.create",
        return_value=container,
    )
    mock_wait_until_healthy_container = mocker.patch(
        "hud.env.local_docker_client.wait_until_healthy_container",
    )
    spy_stream_logs = mocker.spy(hud.env.local_docker_client, "_stream_logs")

    client = await LocalDockerClient.create("test-image:latest", host_config=host_config)
    assert isinstance(client, LocalDockerClient)
    assert client.container_id == "123abc456def"

    mock_create.assert_called_once_with(config=container_config)
    mock_wait_until_healthy_container.assert_called_once_with(container, 1)
    spy_stream_logs.assert_called_once_with(container)
    container.check_logs(caplog)
