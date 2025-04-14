from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.task import Task

logger = logging.getLogger("hud.utils.config")

class HudStyleConfig(BaseModel):
    function: str  # Format: "x.y.z"
    args: list[Any] # Must be json serializable

    def __len__(self) -> int:
        return len(self.args)

    def __getitem__(self, index: int) -> Any:
        return self.args[index]
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.args)
    
    def __str__(self) -> str:
        return f"{self.function}({', '.join(str(arg) for arg in self.args)})"

# Type alias for the shorthand config, which just converts to function name and args
ShorthandConfig = tuple[str | dict[str, Any] | list[str] | list[dict[str, Any]], ...]

# Type alias for multiple config formats
HudStyleConfigs = ShorthandConfig | HudStyleConfig | list[HudStyleConfig] | dict[str, Any] | str

def _is_valid_python_name(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))

def _validate_hud_config(config: dict) -> HudStyleConfig:
    """Validate and convert a dictionary to an HudStyleConfig."""
    if not isinstance(config.get("function"), str):
        raise ValueError("function must be a string")
    
    # Validate function path components
    _split_and_validate_path(config["function"])

    args = config["args"] if isinstance(config.get("args"), list) else [config["args"]]
    
    # Create a proper HudStyleConfig object instead of using cast
    return HudStyleConfig(function=config["function"], args=args)

def _split_and_validate_path(path: str) -> None:
    """Split a function path into components, validating each part."""
    parts = path.split(".")

    if not parts:
        raise ValueError("Empty function path")
    
    # Validate each part
    for part in parts:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in path: {part}")

def expand_config(config: HudStyleConfigs) -> list[HudStyleConfig]:
    """
    Process a config into a standardized list of HudStyleConfig objects.
    
    Args:
        config: Can be:
            - A tuple where first element is function name and rest are args
            - A HudStyleConfig object
            - A dictionary with "function" and "args" keys
            - A list of HudStyleConfig objects
            
    Returns:
        list[HudStyleConfig]: List of standardized configurations
        
    Raises:
        ValueError: If the configuration format is invalid
    """
    logger.debug("Processing config: %s", config)

    # If it's already a HudStyleConfig, just wrap it in a list
    if isinstance(config, HudStyleConfig):
        return [config]
    
    # If it's a list of HudStyleConfigs, return as is
    if isinstance(config, list) and all(isinstance(item, HudStyleConfig) for item in config):
        return config
    
    # Handle dictionary configuration
    if isinstance(config, dict):
        return [_validate_hud_config(config)]
    
    if isinstance(config, str):
        return [HudStyleConfig(function=config, args=[])]
    
    # Handle tuple format
    if isinstance(config, tuple):
        if len(config) < 1 or not isinstance(config[0], str):
            error_msg = f"Invalid tuple configuration. Expected tuple[str, ...], got: {type(config)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # First element is the function name, rest are args
        function_name = config[0]
        args = list(config[1:]) if len(config) > 1 else []
        
        return [HudStyleConfig(function=function_name, args=args)]
    
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

def create_config(task: Task | None = None, config: HudStyleConfigs | None = None, function: str | None = None) -> list[HudStyleConfig]:
    """
    Create a configuration based on provided inputs.
    
    Args:
        task: Task object with configuration
        config: Direct configuration (expanded or not)
        function: Function name to use
        
    Returns:
        list[HudStyleConfig]: List of standardized configurations
        
    Logic:
        1) If explicit config: expand and return HudStyleConfig with func of the function, and args of expanded config
        2) If task has the specified function defined: use that
        3) If no task function: check for task._config and use that
        4) If no _config: use task.id and create private_[function]
    """
    # If no function provided, just expand the config and return it directly
    if function is None:
        if config:
            return expand_config(config)
        raise ValueError("Either function or config must be provided")
    
    # Case 1: Explicit config provided
    if config:
        expanded_configs = expand_config(config)
        return [HudStyleConfig(function=function, args=expanded_configs)]
    
    # Must have a task for the remaining cases
    if task is None:
        raise ValueError("Either task or config must be provided")
    
    # Case 2: Task has the specified function attribute
    task_config = getattr(task, function, None)
    if task_config and len(task_config) > 0:
        expanded_configs = expand_config(task_config)
        return [HudStyleConfig(function=function, args=expanded_configs)]
    
    # Case 3: Check for _config
    if hasattr(task, "config") and task.config:
        return [HudStyleConfig(function=function, args=[task.config])]
    
    # Case 4: Use task.id
    if task.id:
        return [HudStyleConfig(function=f"private_{function}", args=[task.id])]
    
    # No valid configuration found
    raise ValueError(f"Task has no {function}, _config, or id")

