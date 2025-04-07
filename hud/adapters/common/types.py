from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

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
    point: Optional[Point] = None
    selector: Optional[str] = None
    button: Literal["left", "right", "wheel", "back", "forward"] = "left"
    pattern: Optional[list[int]] = None  # [delay_1, delay_2, ...]
    hold_keys: Optional[list[CLAKey]] = None


# PRESS ACTION for key presses/hotkeys
class PressAction(CLAAction):
    type: Literal["press"] = "press"
    keys: list[CLAKey]

# KEYDOWN ACTION for key presses/hotkeys
class KeyDownAction(CLAAction):
    type: Literal["keydown"] = "keydown"
    keys: list[CLAKey]

# KEYUP ACTION for key presses/hotkeys
class KeyUpAction(CLAAction):
    type: Literal["keyup"] = "keyup"
    keys: list[CLAKey]

# TYPE ACTION for text typing
class TypeAction(CLAAction):
    type: Literal["type"] = "type"
    text: str
    selector: Optional[str] = None
    enter_after: Optional[bool] = False


# SCROLL ACTION
class ScrollAction(CLAAction):
    type: Literal["scroll"] = "scroll"
    point: Optional[Point] = None
    scroll: Optional[Point] = None
    hold_keys: Optional[list[CLAKey]] = None


# MOVE ACTION for mouse movement
class MoveAction(CLAAction):
    type: Literal["move"] = "move"
    point: Optional[Point] = None
    selector: Optional[str] = None
    offset: Optional[Point] = None


# WAIT ACTION
class WaitAction(CLAAction):
    type: Literal["wait"] = "wait"
    time: int  # in milliseconds


# DRAG ACTION
class DragAction(CLAAction):
    type: Literal["drag"] = "drag"
    path: list[Point]
    pattern: Optional[list[int]] = None  # [delay_1, delay_2, ...]
    hold_keys: Optional[list[CLAKey]] = None

class CustomAction(CLAAction):
    type: Literal["custom"] = "custom"
    script: str

# SCREENSHOT ACTION
class ScreenshotFetch(CLAAction):
    type: Literal["screenshot"] = "screenshot"


class PositionFetch(CLAAction):
    type: Literal["position"] = "position"


class CustomAction(CLAAction):
    type: Literal["custom"] = "custom"
    action: str

# Union of all possible actions
CLA = Annotated[
    Union[
        ClickAction,
        PressAction,
        KeyDownAction,
        KeyUpAction,
        TypeAction,
        ScrollAction,
        MoveAction,
        WaitAction,
        DragAction,
        CustomAction,
        ScreenshotFetch,
        PositionFetch,
        CustomAction,
    ],
    Field(discriminator="type"),
]


CLAKey = Literal[
    # Control keys
    "backspace",
    "tab",
    "enter",
    "shift",
    "shiftleft",
    "shiftright",
    "ctrl",
    "ctrlleft",
    "ctrlright",
    "alt",
    "altleft",
    "altright",
    "pause",
    "capslock",
    "esc",
    "escape",
    "space",
    "pageup",
    "pagedown",
    "end",
    "home",
    "left",
    "up",
    "right",
    "down",
    "select",
    "print",
    "execute",
    "printscreen",
    "prtsc",
    "insert",
    "delete",
    "help",
    "sleep",
    # Special keys
    "numlock",
    "scrolllock",
    "clear",
    "separator",
    "modechange",
    "apps",
    "browserback",
    "browserfavorites",
    "browserforward",
    "browserhome",
    "browserrefresh",
    "browsersearch",
    "browserstop",
    "launchapp1",
    "launchapp2",
    "launchmail",
    "launchmediaselect",
    "playpause",
    "stop",
    "prevtrack",
    "nexttrack",
    "volumemute",
    "volumeup",
    "volumedown",
    "zoom",
    # Modifier keys
    "win",
    "winleft",
    "winright",
    "command",
    "option",
    "optionleft",
    "optionright",
    "fn",
    # Numpad keys
    "num0",
    "num1",
    "num2",
    "num3",
    "num4",
    "num5",
    "num6",
    "num7",
    "num8",
    "num9",
    "multiply",
    "add",
    "subtract",
    "decimal",
    "divide",
    # Function keys
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    # Language-specific keys
    "hanguel",
    "hangul",
    "hanja",
    "kana",
    "kanji",
    "junja",
    "convert",
    "nonconvert",
    "yen",
    # Characters
    "\t",
    "\n",
    "\r",
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
]
