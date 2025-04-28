from __future__ import annotations

from typing import Any, ClassVar

from hud.adapters.common import CLA, Adapter
from hud.adapters.common.types import (
    CLAKey,
    ClickAction,
    DragAction,
    MoveAction,
    Point,
    PressAction,
    ResponseAction,
    ScreenshotFetch,
    ScrollAction,
    TypeAction,
    WaitAction,
)


class OperatorAdapter(Adapter):
    KEY_MAP: ClassVar[dict[str, CLAKey]] = {
        "return": "enter",
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
    }
    
    def __init__(self) -> None:
        super().__init__()
        # OpenAI Computer Use default dimensions
        self.agent_width = 1024
        self.agent_height = 768
        
    def _map_key(self, key: str) -> CLAKey:
        """Map a key to its standardized form."""
        return self.KEY_MAP.get(key.lower(), key.lower())  # type: ignore
        
    def convert(self, data: Any) -> CLA:
        """Convert a Computer Use action to a HUD action"""
        try:
            action_type = data.get("type")
            
            if action_type == "click":
                x, y = data.get("x", 0), data.get("y", 0)
                button = data.get("button", "left")
                return ClickAction(point=Point(x=x, y=y), button=button)
                
            elif action_type == "double_click":
                x, y = data.get("x", 0), data.get("y", 0)
                return ClickAction(
                    point=Point(x=x, y=y),
                    button="left",
                    pattern=[100]
                )
                
            elif action_type == "scroll":
                x, y = data.get("x", 0), data.get("y", 0)
                scroll_x = data.get("scroll_x", 0)
                scroll_y = data.get("scroll_y", 0)
                return ScrollAction(
                    point=Point(x=x, y=y),
                    scroll=Point(x=scroll_x, y=scroll_y)
                )
                
            elif action_type == "type":
                text = data.get("text", "")
                return TypeAction(text=text, enter_after=False)
                
            elif action_type == "wait":
                ms = data.get("ms", 1000)
                return WaitAction(time=ms)
                
            elif action_type == "move":
                x, y = data.get("x", 0), data.get("y", 0)
                return MoveAction(point=Point(x=x, y=y))
                
            elif action_type == "keypress":
                keys = data.get("keys", [])
                return PressAction(keys=[self._map_key(k) for k in keys])
                
            elif action_type == "drag":
                path = data.get("path", [])
                points = [Point(x=p.get("x", 0), y=p.get("y", 0)) for p in path]
                return DragAction(path=points)
            
            elif action_type == "screenshot":
                return ScreenshotFetch()
            
            elif action_type == "response":
                return ResponseAction(text=data.get("text", ""))
            else:
                raise ValueError(f"Unsupported action type: {action_type}")
                
        except Exception as e:
            raise ValueError(f"Invalid action: {data}. Error: {e!s}") from e
