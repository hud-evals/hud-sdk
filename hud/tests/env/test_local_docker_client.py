from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hud.env.local_docker_client import _stream_logs, wait_until_healthy_container


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
