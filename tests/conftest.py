"""Shared fixtures and helpers for HUD SDK unit tests.

All tests are strictly off-line: any attempt to hit the real HUD API or
external LLM providers should raise immediately.  The fixtures below monkey-
patch the relevant entry-points so that importing production code in a test
context never leaves the local process.
"""
from __future__ import annotations

import asyncio
from typing import Any
import sys

import pytest

from hud.utils.config import expand_config
from hud.utils.common import FunctionConfig

# ---------------------------------------------------------------------------
# Dummy no-op implementation of hud.env.client.Client interface --------------
# ---------------------------------------------------------------------------

class DummyClient:
    """A minimal stand-in that satisfies the async interface used by Environment."""

    async def invoke(self, _config):  # type: ignore[override]
        # Return predictable tuple of (result, stdout, stderr)
        return "ok", b"", b""

    async def get_status(self):  # type: ignore[override]
        from hud.types import EnvironmentStatus

        return EnvironmentStatus.RUNNING

    async def close(self):  # type: ignore[override]
        return None

    # LocalDockerClient expects execute() + set_source_path() for build flow.
    async def execute(self, _cmd, **_kw):  # noqa: D401
        return {"stdout": b"", "stderr": b"", "exit_code": 0}

    def set_source_path(self, _p):
        # noop
        pass


# dummy_client can be synchronous; Environment code only expects awaitable methods on the client itself.
@pytest.fixture()
def dummy_client():
    """Provides an *already constructed* DummyClient instance."""

    return DummyClient()


# ---------------------------------------------------------------------------
# Global monkey-patches to neuter network & docker ---------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_network_calls(monkeypatch):
    """Prevent accidental HTTP calls in unit tests."""

    def _boom(*_a, **_kw):  # noqa: D401 – simple boom function
        raise RuntimeError("Network call attempted during offline unit test")

    # HUD internal HTTP helper
    monkeypatch.setattr("hud.server.requests.make_request", _boom)
    # OpenAI / Anthropic imports inside agents
    monkeypatch.setitem(sys.modules, "openai", object())
    monkeypatch.setitem(sys.modules, "anthropic", object())


# ---------------------------------------------------------------------------
# Patch environment-creation helpers so that hud.gym.make() is instant -------
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_env_creation(monkeypatch, dummy_client):
    """Route all Local/Remote client .create() calls to our DummyClient."""

    async def _return_dummy(*_a, **_kw):
        return dummy_client, {}

    # Local docker environments
    monkeypatch.setattr(
        "hud.env.local_docker_client.LocalDockerClient.create", _return_dummy, raising=False
    )
    # Remote docker environments
    monkeypatch.setattr(
        "hud.env.remote_docker_client.RemoteDockerClient.create", _return_dummy, raising=False
    )
    # Remote non-docker environments
    monkeypatch.setattr(
        "hud.env.remote_client.RemoteClient.create", _return_dummy, raising=False
    )

    # get_gym_id -> just echo back what it got (avoids HTTP)
    async def _fake_get_gym_id(name: str):  # noqa: D401
        return name

    monkeypatch.setattr("hud.utils.common.get_gym_id", _fake_get_gym_id)


# ---------------------------------------------------------------------------
# pytest-asyncio automatic event-loop fixture --------------------------------
# ---------------------------------------------------------------------------

try:
    import pytest_asyncio  # type: ignore
except ImportError:  # pragma: no cover – dev extra should install this
    pytest.skip("pytest-asyncio required for async tests", allow_module_level=True)


@pytest_asyncio.fixture(scope="session")
def event_loop():  # noqa: D401
    """Create an instance of the default event loop for session scope."""

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()