from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# Base class for all actions
class CLAAction(BaseModel):
    type: str


# Basic Point model for coordinates
class Point(BaseModel):
    x: int
    y: int


# CLICK ACTION (supports extra options)
class ClickAction(CLAAction):
    type: Literal["click"] = "click"
    point: Point | None = None
    selector: str | None = None
    button: Literal["left", "right", "wheel", "back", "forward"] = "left"
    pattern: list[int] | None = None  # [delay_1, delay_2, ...]


# PRESS ACTION for key presses/hotkeys
class PressAction(CLAAction):
    type: Literal["press"] = "press"
    keys: list[str]


# TYPE ACTION for text typing
class TypeAction(CLAAction):
    type: Literal["type"] = "type"
    text: str
    enter_after: bool | None = False


# SCROLL ACTION
class ScrollAction(CLAAction):
    type: Literal["scroll"] = "scroll"
    point: Point | None = None
    scroll: Point | None = None


# MOVE ACTION for mouse movement
class MoveAction(CLAAction):
    type: Literal["move"] = "move"
    point: Point | None = None
    selector: str | None = None
    offset: Point | None = None


# WAIT ACTION
class WaitAction(CLAAction):
    type: Literal["wait"] = "wait"
    time: int  # in milliseconds


# DRAG ACTION
class DragAction(CLAAction):
    type: Literal["drag"] = "drag"
    path: list[Point]
    pattern: list[int] | None = None  # [delay_1, delay_2, ...]


# SCREENSHOT ACTION
class ScreenshotFetch(CLAAction):
    type: Literal["screenshot"] = "screenshot"


class PositionFetch(CLAAction):
    type: Literal["position"] = "position"


# Union of all possible actions
CLA = Annotated[
    Union[
        ClickAction,
        PressAction,
        TypeAction,
        ScrollAction,
        MoveAction,
        WaitAction,
        DragAction,
        ScreenshotFetch,
        PositionFetch,
    ],
    Field(discriminator="type"),
]
