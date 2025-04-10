from __future__ import annotations

import logging
import re
from typing import Any, cast

from lark import Lark, Token, Transformer
from pydantic import BaseModel

logger = logging.getLogger("hud.utils.config")

# Grammar for parsing function calls
FUNC_CALL_GRAMMAR = r"""
    ?start: funccall

    funccall: funcname "(" [funcargs] ")"
    funcname: CNAME ("." CNAME)*
    funcargs: funcarg ("," funcarg)* [","]
    funcarg: NUMBER -> number
           | STRING -> string
           | "[" [funcargs] "]" -> list
           | funccall -> nested_call

    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %import common.ESCAPED_STRING -> STRING
    %ignore WS
"""

class FuncCallTransformer(Transformer):
    def funccall(self, items: list[Any]) -> dict[str, Any]:
        funcname = items[0]
        args = items[1] if len(items) > 1 else []
        return {"function": ".".join(funcname), "args": args}
    
    def funcname(self, items: list[Token]) -> list[str]:
        return [str(token) for token in items]
    
    def funcargs(self, items: list[Any]) -> list[Any]:
        return list(items)
    
    def number(self, items: list[Token]) -> float:
        return float(items[0].value)
    
    def string(self, items: list[Token]) -> str:
        return items[0].value[1:-1]  # Remove quotes
    
    def list(self, items: list[Any]) -> list[Any]:
        return items[0] if items else []
    
    def nested_call(self, items: list[Any]) -> Any:
        return items[0]
    
    def CNAME(self, token: Token) -> str:
        return str(token)

# Create the parser
func_call_parser = Lark(FUNC_CALL_GRAMMAR, parser="lalr", transformer=FuncCallTransformer())

class ExpandedConfig(BaseModel):
    function: str  # Format: "x.y.z"
    args: list[Any]  # Must be json serializable

HudStyleConfig = str | ExpandedConfig | list[str] | list[ExpandedConfig]

def _is_valid_python_name(name: str) -> bool:
    """Check if a string is a valid Python identifier."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))

def _validate_expanded_config(config: dict) -> ExpandedConfig:
    """Validate and convert a dictionary to an ExpandedConfig."""
    if not isinstance(config.get("function"), str):
        raise ValueError("function must be a string")
    if not isinstance(config.get("args"), list):
        raise ValueError("args must be a list")
    
    # Validate function path components
    _split_and_validate_path(config["function"])
    
    return cast(ExpandedConfig, config)

def _split_and_validate_path(path: str) -> None:
    """Split a function path into components, validating each part."""
    parts = path.split(".")

    if not parts:
        raise ValueError("Empty function path")
    
    # Validate each part
    for part in parts:
        if not _is_valid_python_name(part):
            raise ValueError(f"Invalid Python identifier in path: {part}")

def _process_string_config(config: str) -> list[ExpandedConfig]:
    """Process a string configuration into ExpandedConfig format.
    
    The string should be in the format of a function call, e.g.:
    - "chrome.maximize()"
    - "browser.open('https://example.com')"
    - "app.set_position([100, 200])"
    """
    result = func_call_parser.parse(config)
    return [cast(ExpandedConfig, result)]

def expand_config(config: HudStyleConfig) -> list[ExpandedConfig]:
    """
    Process a configuration into a standardized list of ExpandedConfig formats.
    
    Args:
        config: The configuration, which can be:
            - String (function name): "chrome.maximize()"
            - String (function with args): "chrome.activate_tab(5)"
            - ExpandedConfig: {"function": "chrome.maximize", "args": []}
            - List of any of the above
            
    Returns:
        list[ExpandedConfig]: List of standardized configurations with function and args
        
    Raises:
        ValueError: If the configuration format is not recognized or contains
            invalid Python identifiers
    """
    logger.debug("Processing config: %s", config)
    
    # Handle list of configurations
    if isinstance(config, list):
        results = []
        for item in config:
            results.extend(expand_config(item))
        return results
    
    # Handle string configurations
    if isinstance(config, str):
        return _process_string_config(config)
    
    # Validate dictionary configurations
    if isinstance(config, dict):
        return [_validate_expanded_config(cast(dict, config))]
    
    # Unknown configuration type
    error_msg = f"Unknown configuration type: {type(config)}"
    logger.error(error_msg)
    raise ValueError(error_msg)
