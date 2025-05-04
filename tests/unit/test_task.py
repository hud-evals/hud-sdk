# Test cases for the hud.Task class
import pytest
from hud.task import Task
from hud.types import CustomGym  # Import CustomGym if needed for specific tests
from hud.utils.common import FunctionConfig

def test_task_initialization_valid():
    """Tests basic Task initialization with valid arguments."""
    task = Task(
        prompt="Test prompt",
        gym="hud-browser",
        setup=("goto", "example.com"),
        evaluate=("page_contains", ["success"])
    )
    assert task.prompt == "Test prompt"
    assert task.gym == "hud-browser"
    # Using tuple format for setup/evaluate as per docs
    assert task.setup == ("goto", "example.com")
    assert task.evaluate == ("page_contains", ["success"])

def test_task_initialization_valid_dict_config():
    """Tests Task initialization with dict format for setup/evaluate."""
    task = Task(
        prompt="Test prompt dict",
        gym="hud-browser",
        setup={"function": "goto", "args": ["example.com"]},
        evaluate=[{"function": "page_contains", "args": ["success"]}] # type: ignore # List of dicts seems valid per FunctionConfigs, but checker struggles
    )
    assert task.prompt == "Test prompt dict"
    assert task.gym == "hud-browser"
    assert isinstance(task.setup, FunctionConfig)
    assert task.setup.function == "goto"
    assert task.evaluate and isinstance(task.evaluate, list)
    assert isinstance(task.evaluate[0], FunctionConfig)

def test_task_initialization_valid_custom_gym():
    """Tests Task initialization with a CustomGym object."""
    custom_gym = CustomGym(location="remote", dockerfile="FROM ubuntu:latest")
    task = Task(
        prompt="Custom gym prompt",
        gym=custom_gym
    )
    assert task.prompt == "Custom gym prompt"
    assert task.gym == custom_gym

def test_task_initialization_minimal():
    """Tests Task initialization with only required arguments."""
    task = Task(prompt="Minimal prompt", gym="qa") # Use a valid Gym literal like 'qa'
    assert task.prompt == "Minimal prompt"
    assert task.gym == "qa"
    assert task.setup is None
    assert task.evaluate is None

def test_task_initialization_invalid_gym_type():
    """Tests Task initialization with an invalid gym type string."""
    # Pydantic validation should catch invalid enum values for the Literal 'Gym'
    # We expect a ValidationError from Pydantic if 'invalid-gym' is not in the Gym Literal
    # Note: Depending on implementation, this might raise during Pydantic validation
    # or potentially later if not strictly validated by Pydantic.
    # Let's assume Pydantic validation for now.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
          # type: ignore # Intentionally passing invalid type for testing
        Task(prompt="Invalid gym", gym="invalid-gym-string")

def test_task_initialization_invalid_setup_format():
    """Tests Task initialization with invalid 'setup' formats."""
    from pydantic import ValidationError
    # String is valid ('function name only'), so test other invalid structures
    # Test non-string function name in tuple
    with pytest.raises(ValidationError):
        Task(prompt="Invalid setup", gym="hud-browser", setup=(1, "arg")) # type: ignore
    # Test unexpected primitive type
    with pytest.raises(ValidationError):
        Task(prompt="Invalid setup", gym="hud-browser", setup=123)  # type: ignore

def test_task_initialization_invalid_evaluate_format():
    """Tests Task initialization with invalid 'evaluate' formats."""
    from pydantic import ValidationError
    # Similar checks as setup
    with pytest.raises(ValidationError):
        Task(prompt="Invalid eval", gym="hud-browser", evaluate=(1, ["arg"])) # type: ignore
    with pytest.raises(ValidationError):
        Task(prompt="Invalid eval", gym="hud-browser", evaluate=123)  # type: ignore

# Potential future test: Check specific known gym types if there's a predefined list?
# Potential future test: Check prompt length or content constraints if any? 