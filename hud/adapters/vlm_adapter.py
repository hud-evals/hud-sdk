"""Adapter for Vision-Language Models that output structured text."""

from __future__ import annotations

import json
import re
from typing import Any

from .common import Adapter


class VLMAdapter(Adapter):
    """Adapter for VLM agents that output structured action text.
    
    Expected VLM output format:
    ```
    ACTION: click(100, 200)
    ACTION: type("Hello world")
    ACTION: scroll(down, 5)
    DONE: Task completed
    ```
    
    Or JSON format:
    ```json
    {"action": "click", "coordinate": [100, 200]}
    {"action": "type", "text": "Hello world"}
    ```
    """
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        format: str = "structured",  # "structured" or "json"
    ):
        super().__init__()
        self.agent_width = width
        self.agent_height = height
        self.format = format
        
    def preprocess(self, action: Any) -> Any:
        """Parse VLM text response into action dictionary."""
        if not isinstance(action, str):
            return action
            
        if self.format == "json":
            return self._parse_json_action(action)
        else:
            return self._parse_structured_action(action)
            
    def _parse_json_action(self, text: str) -> dict[str, Any]:
        """Parse JSON-formatted action."""
        try:
            # Handle multiple JSON objects in response
            lines = text.strip().split('\n')
            for line in lines:
                if line.strip().startswith('{'):
                    return json.loads(line.strip())
            raise ValueError("No JSON action found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON action: {text}") from e
            
    def _parse_structured_action(self, text: str) -> dict[str, Any]:
        """Parse structured text action like 'click(100, 200)'."""
        # Match patterns like: click(100, 200), type("text"), scroll(down, 5)
        patterns = {
            r'click\((\d+),\s*(\d+)\)': lambda m: {
                "action": "left_click",
                "coordinate": [int(m.group(1)), int(m.group(2))]
            },
            r'right_click\((\d+),\s*(\d+)\)': lambda m: {
                "action": "right_click", 
                "coordinate": [int(m.group(1)), int(m.group(2))]
            },
            r'type\("([^"]+)"\)': lambda m: {
                "action": "type",
                "text": m.group(1)
            },
            r'type\(\'([^\']+)\'\)': lambda m: {
                "action": "type",
                "text": m.group(1)
            },
            r'press\(([^)]+)\)': lambda m: {
                "action": "key",
                "text": m.group(1).strip('"\'')
            },
            r'scroll\((\w+),\s*(\d+)\)': lambda m: {
                "action": "scroll",
                "scroll_direction": m.group(1),
                "scroll_amount": int(m.group(2)),
                "coordinate": [960, 540]  # Default to center
            },
            r'wait\((\d+(?:\.\d+)?)\)': lambda m: {
                "action": "wait",
                "duration": float(m.group(1))
            },
            r'screenshot\(\)': lambda m: {
                "action": "screenshot"
            },
            r'done\(\)': lambda m: {
                "action": "response",
                "text": "Task completed"
            },
        }
        
        # Try each pattern
        for pattern, builder in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return builder(match)
                
        # If no pattern matches, try to extract any text after "ACTION:"
        action_match = re.search(r'ACTION:\s*(.+)', text, re.IGNORECASE)
        if action_match:
            return self._parse_structured_action(action_match.group(1).strip())
            
        return {"type": "response", "text": text}


class QwenAdapter(VLMAdapter):
    """Adapter specifically tuned for Qwen-VL models."""
    
    def __init__(self):
        super().__init__(width=1920, height=1080, format="structured")
        
    def preprocess(self, action: Any) -> Any:
        """Handle Qwen-specific response format."""
        if isinstance(action, str):
            # Qwen sometimes adds explanation before action
            # Extract just the action part
            lines = action.strip().split('\n')
            for line in lines:
                if 'ACTION:' in line.upper() or any(
                    pattern in line.lower() 
                    for pattern in ['click(', 'type(', 'scroll(', 'press(']
                ):
                    # Found action line, parse it
                    if 'ACTION:' in line.upper():
                        action_text = line.split('ACTION:', 1)[1].strip()
                    else:
                        action_text = line.strip()
                    return super().preprocess(action_text)
                    
        return super().preprocess(action)