from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hud.task import Task
from hud.taskset import TaskSet, load_taskset


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        Task(
            id="task-1",
            prompt="Test task 1",
            gym="hud-browser",
            setup=("goto", "https://example.com"),
            evaluate=("page_contains", "Example"),
        ),
        Task(id="task-2", prompt="Test task 2", gym="qa", evaluate=("response_includes", "answer")),
        Task(
            id="task-3",
            prompt="Test task 3 with metadata",
            gym="hud-browser",
            metadata={"difficulty": "easy", "category": "navigation"},
        ),
    ]


@pytest.fixture
def sample_taskset(sample_tasks):
    """Create a sample TaskSet."""
    return TaskSet(
        id="test-taskset-123", description="Test TaskSet for unit tests", tasks=sample_tasks
    )


def test_taskset_creation(sample_tasks):
    """Test TaskSet initialization and properties."""
    taskset = TaskSet(description="My test set", tasks=sample_tasks)

    assert taskset.description == "My test set"
    assert len(taskset.tasks) == 3
    assert taskset.id is None  # No ID when created locally

    # Test with ID
    taskset_with_id = TaskSet(id="custom-id", description="Test", tasks=sample_tasks[:1])
    assert taskset_with_id.id == "custom-id"


def test_taskset_supports_integer_indexing(sample_taskset):
    """TaskSet should support accessing tasks by integer index."""
    # Positive indexing
    assert sample_taskset[0].id == "task-1"
    assert sample_taskset[1].id == "task-2"
    assert sample_taskset[2].id == "task-3"

    # Negative indexing
    assert sample_taskset[-1].id == "task-3"
    assert sample_taskset[-2].id == "task-2"

    # Out of bounds should raise IndexError
    with pytest.raises(IndexError):
        _ = sample_taskset[10]
    with pytest.raises(IndexError):
        _ = sample_taskset[-10]


def test_taskset_length_returns_number_of_tasks(sample_taskset):
    """len(TaskSet) should return the number of tasks."""
    assert len(sample_taskset) == 3

    # Empty taskset
    empty = TaskSet(tasks=[])
    assert len(empty) == 0

    # Single task
    single = TaskSet(tasks=[sample_taskset[0]])
    assert len(single) == 1


@pytest.mark.asyncio
async def test_taskset_upload_posts_tasks_to_api(sample_taskset, mocker):
    """TaskSet.upload should POST tasks to the platform API."""
    # Arrange
    expected_response = {"id": "uploaded-taskset-456"}
    mock_make_request = mocker.patch("hud.taskset.make_request", new_callable=AsyncMock)
    mock_make_request.return_value = expected_response

    # Act
    await sample_taskset.upload(
        name="My Evaluation Set", description="Testing upload functionality", api_key="test-api-key"
    )

    # Assert API was called correctly
    mock_make_request.assert_called_once()
    call_kwargs = mock_make_request.call_args[1]

    assert call_kwargs["method"] == "POST"
    assert "tasksets" in call_kwargs["url"]
    assert call_kwargs["api_key"] == "test-api-key"

    # Verify request payload structure
    payload = call_kwargs["json"]
    assert payload["name"] == "My Evaluation Set"
    assert payload["description"] == "Testing upload functionality"
    assert len(payload["tasks"]) == 3

    # Verify task data is included
    first_task = payload["tasks"][0]
    assert first_task["prompt"] == "Test task 1"
    assert first_task["gym"] == "hud-browser"


@pytest.mark.asyncio
async def test_load_taskset_fetches_and_deserializes_tasks(mocker):
    """load_taskset should fetch tasks from API and create TaskSet object."""
    # Arrange - realistic API response
    api_response = {
        "evalset": [
            {
                "id": "task-1",
                "prompt": "Navigate to checkout",
                "gym": "hud-browser",
                "setup": {"function": "goto", "args": ["https://shop.example.com"]},
                "evaluate": {"function": "page_contains", "args": ["Checkout"]},
            },
            {
                "id": "task-2",
                "prompt": "Answer the question",
                "gym": "qa",
                "evaluate": {"function": "response_includes", "args": ["42"]},
            },
        ]
    }

    mock_make_request = mocker.patch("hud.taskset.make_request", new_callable=AsyncMock)
    mock_make_request.return_value = api_response

    # Act
    loaded_taskset = await load_taskset("test-taskset-id", api_key="test-key")

    # Assert
    assert isinstance(loaded_taskset, TaskSet)
    assert loaded_taskset.id == "test-taskset-id"
    assert len(loaded_taskset) == 2

    # Verify task details
    task1 = loaded_taskset[0]
    assert task1.prompt == "Navigate to checkout"
    assert task1.gym == "hud-browser"
    # Check setup exists (don't assume specific type)
    assert task1.setup is not None

    task2 = loaded_taskset[1]
    assert task2.prompt == "Answer the question"
    # Check evaluate exists
    assert task2.evaluate is not None

    # Verify API call
    mock_make_request.assert_called_once()
    call_kwargs = mock_make_request.call_args[1]
    assert "tasksets/test-taskset-id/tasks" in call_kwargs["url"]
    assert call_kwargs["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_load_taskset_propagates_api_errors(mocker):
    """load_taskset should propagate API errors to caller."""
    # Arrange
    mock_make_request = mocker.patch("hud.taskset.make_request", new_callable=AsyncMock)
    mock_make_request.side_effect = ValueError("Invalid taskset ID")

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid taskset ID"):
        await load_taskset("invalid-id", api_key="test-key")


def test_taskset_initializes_with_minimal_configuration():
    """TaskSet should work with just a list of tasks."""
    task = Task(prompt="Simple task")
    taskset = TaskSet(tasks=[task])

    assert len(taskset) == 1
    assert taskset[0].prompt == "Simple task"
    assert taskset.id is None
    assert taskset.description is None


def test_taskset_with_mixed_gym_types(sample_tasks):
    """Test TaskSet with different gym types."""
    # Add a custom gym task
    from pathlib import Path

    from hud.types import CustomGym

    custom_task = Task(
        prompt="Custom environment task",
        gym=CustomGym(location="local", image_or_build_context=Path("/test/path")),
        evaluate=("custom_eval", "arg"),
    )

    mixed_tasks = [*sample_tasks, custom_task]
    taskset = TaskSet(tasks=mixed_tasks)

    assert len(taskset) == 4
    # Verify the custom gym task is properly included
    assert isinstance(taskset[3].gym, CustomGym)


@pytest.mark.asyncio
async def test_taskset_with_local_gym_spec():
    """Test TaskSet with a local gym spec."""
    from pathlib import Path

    from hud.types import CustomGym

    custom_task = Task(
        prompt="Custom environment task",
        gym=CustomGym(location="local", image_or_build_context=Path("/test/path")),
    )

    taskset = TaskSet(tasks=[custom_task])

    with pytest.raises(
        ValueError,
        match="Local build contexts are not supported for remote tasksets, "
        "attach an image or existing gym id.",
    ):
        await taskset.upload(name="Test", description="Test", api_key="test-api-key")
