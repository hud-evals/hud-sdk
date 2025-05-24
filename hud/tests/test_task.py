from __future__ import annotations

from dataclasses import dataclass

from hud.task import Task


@dataclass
class FakeChat:
    role: str
    content: str


@dataclass
class FakeSample:
    input: str | list[FakeChat]
    choices: list[str] | None = None
    target: str | list[str] | None = None
    sandbox: str | tuple[str, str] | None = None
    metadata: dict | None = None
    id: str | None = None
    files: dict | None = None
    setup: str | None = None


def test_from_inspect_sample_qa():
    samp = FakeSample(input="Hello", sandbox=None)
    task = Task.from_inspect_sample(samp)  # type: ignore[arg-type]
    assert task.gym == "qa"
    assert task.setup is None


def test_from_inspect_sample_docker():
    samp = FakeSample(input="Run ls", sandbox="docker", setup="echo hi")
    task = Task.from_inspect_sample(samp)  # type: ignore[arg-type]
    from hud.types import CustomGym

    assert isinstance(task.gym, CustomGym)
    assert task.gym.location == "local"
    # setup converted to FunctionConfig list
    assert task.setup is not None and isinstance(task.setup, list)
