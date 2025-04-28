# ruff: noqa: S101

from __future__ import annotations

from typing import Any, ClassVar

from hud.adapters.common import CLA, Adapter
from hud.adapters.common.types import (
    CLAKey,
    ClickAction,
    DragAction,
    MoveAction,
    Point,
    PositionFetch,
    PressAction,
    ResponseAction,
    ScreenshotFetch,
    ScrollAction,
    TypeAction,
    WaitAction,
)


class ClaudeAdapter(Adapter):
    KEY_MAP: ClassVar[dict[str, CLAKey]] = {
        "return": "enter",
        "super": "win",
        "super_l": "win",
        "super_r": "win",
        "right shift": "shift",
        "left shift": "shift",
    }

    def __init__(self) -> None:
        super().__init__()
        self.agent_width = 1024  # Claude's preferred width
        self.agent_height = 768  # Claude's preferred height

    def _map_key(self, key: str) -> CLAKey:
        """Map a key to its standardized form."""
        return self.KEY_MAP.get(key.lower(), key.lower())  # type: ignore
    
    def convert(self, data: Any) -> CLA:
        try:
            action_type = data.get("action")

            if action_type == "key":
                assert "text" in data
                if "+" in data["text"]:
                    keys: list[CLAKey] = [
                        self._map_key(k) for k in (data["text"].split("+"))
                    ]
                    assert len(keys) > 0
                    return PressAction(keys=keys)
                return PressAction(keys=[self._map_key(data["text"])])

            elif action_type == "type":
                assert "text" in data
                return TypeAction(
                    text=data["text"],
                    enter_after=False,
                )

            elif action_type == "mouse_move":
                # 'coordinate' should be provided as an array [x, y].
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return MoveAction(point=Point(x=coord[0], y=coord[1]))

            elif action_type == "left_click":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return ClickAction(point=Point(x=coord[0], y=coord[1]), button="left")

            elif action_type == "left_click_drag":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                if (
                    len(self.memory) == 0
                    or (
                        self.memory[-1] is not MoveAction
                        and self.memory[-1] is not ClickAction
                    )
                    or self.memory[-1].point is None
                ):
                    raise ValueError(
                        "Left click drag must be preceded by a move or click action"
                    )
                else:
                    return DragAction(
                        path=[self.memory[-1].point, Point(x=coord[0], y=coord[1])]
                    )

            elif action_type == "right_click":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return ClickAction(point=Point(x=coord[0], y=coord[1]), button="right")

            elif action_type == "middle_click":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return ClickAction(point=Point(x=coord[0], y=coord[1]), button="wheel")

            elif action_type == "double_click":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return ClickAction(
                    point=Point(x=coord[0], y=coord[1]), button="left", pattern=[100]
                )

            elif action_type == "triple_click":
                assert "coordinate" in data
                coord = data["coordinate"]
                assert isinstance(coord, list)
                assert len(coord) == 2
                return ClickAction(
                    point=Point(x=coord[0], y=coord[1]),
                    button="left",
                    pattern=[100, 100],
                )

            elif action_type == "scroll":
                assert "scroll_direction" in data
                direction = data["scroll_direction"]

                if direction == "up":
                    scroll = Point(x=0, y=-data["scroll_amount"])
                elif direction == "down":
                    scroll = Point(x=0, y=data["scroll_amount"])
                elif direction == "left":
                    scroll = Point(x=-data["scroll_amount"], y=0)
                elif direction == "right":
                    scroll = Point(x=data["scroll_amount"], y=0)
                else:
                    raise ValueError(f"Unsupported scroll direction: {direction}")

                return ScrollAction(
                    point=Point(x=data["coordinate"][0], y=data["coordinate"][1]),
                    scroll=scroll,
                )

            elif action_type == "screenshot":
                return ScreenshotFetch()

            elif action_type == "cursor_position":
                return PositionFetch()

            elif action_type == "wait":
                assert "duration" in data
                return WaitAction(time=data["duration"])

            elif action_type == "response":
                return ResponseAction(text=data.get("text", ""))

            else:
                raise ValueError(f"Unsupported action type: {action_type}")
        except AssertionError:
            raise ValueError(f"Invalid action: {data}") from None
