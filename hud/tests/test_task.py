from __future__ import annotations

from dataclasses import dataclass

from hud.task import Task
from hud.types import CustomGym


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


def test_serialization():
    my_gym = CustomGym(
        location="remote",
        image_or_build_context="hud/hud-gym:latest",
    )
    task = Task(
        id="123",
        prompt="Test",
        setup=[("echo", "test")],
        gym=my_gym,
    )

    serialized = task.serialize()
    assert serialized["gym"]["location"] == "remote"
    assert serialized["gym"]["image_or_build_context"] == "hud/hud-gym:latest"
    assert serialized["setup"] == [["echo", "test"]]
    assert serialized["prompt"] == "Test"
    assert serialized["id"] == "123"


def test_serialization_nondocker_gym():
    task = Task(
        id="123",
        prompt="Test",
        setup=[("echo", "test")],
        gym="hud-browser",
    )
    serialized = task.serialize()
    assert serialized["gym"] == "hud-browser"
    assert serialized["setup"] == [["echo", "test"]]
    assert serialized["prompt"] == "Test"
    assert serialized["id"] == "123"


def test_deserialize_docker_gym():
    serialized = {
        "id": "123",
        "prompt": "Test",
        "setup": [["echo", "test"]],
        "gym": {
            "location": "remote",
            "image_or_build_context": "hud/hud-gym:latest",
        },
    }
    task = Task.from_serialized(serialized)
    assert task.id == "123"
    assert task.prompt == "Test"
    assert task.setup == [("echo", "test")]

    assert isinstance(task.gym, CustomGym)
    assert task.gym.location == "remote"
    assert task.gym.image_or_build_context == "hud/hud-gym:latest"


def test_deserialize_nondocker_gym():
    serialized = {
        "id": "123",
        "prompt": "Test",
        "setup": [["echo", "test"]],
        "gym": "hud-browser",
    }
    task = Task.from_serialized(serialized)
    assert task.id == "123"
    assert task.prompt == "Test"
    assert task.setup == [("echo", "test")]
    assert task.gym == "hud-browser"
