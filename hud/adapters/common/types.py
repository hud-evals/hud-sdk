from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field


# Helper function to format logs for display
def _format_logs_for_display(
    logs: str | dict[str, Any] | list[str | dict[str, Any]] | None,
    max_log_len: int = 277,
) -> str:
    if not logs:
        return ""
    log_repr = repr(logs)
    truncated_log = log_repr[:max_log_len] + "..." if len(log_repr) > max_log_len else log_repr
    return f" │ Logs: {truncated_log}"


# Base class for all actions
class CLAAction(BaseModel):
    type: str
    logs: str | dict[str, Any] | list[str | dict[str, Any]] | None = None

    def __str__(self) -> str:
        # Basic representation for actions that don't have a specific override
        # This base __str__ will NOT include logs by default, subclasses should handle it.
        attributes = ", ".join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in self.model_dump().items()
            if k != "type" and v is not None and k != "logs"
        )
        action_str = f"{self.type.capitalize()}Action ({attributes})"
        action_str += _format_logs_for_display(self.logs)  # Add logs if present via helper
        return action_str


# Basic Point model for coordinates
class Point(BaseModel):
    x: int
    y: int


# CLICK ACTION (supports extra options)
class ClickAction(CLAAction):
    type: Literal["click"] = "click"
    point: Point | None = None
    button: CLAButton = "left"
    pattern: list[int] | None = None
    hold_keys: list[CLAKey] | None = None

    def __str__(self) -> str:
        parts = ["💥 Click"]
        if self.point:
            parts.append(f"at ({self.point.x}, {self.point.y})")
        if self.button != "left":
            parts.append(f"with {self.button} button")
        if self.hold_keys:
            parts.append(f"holding {self.hold_keys}")
        action_str = " ".join(parts)
        action_str += _format_logs_for_display(self.logs)
        return action_str


# PRESS ACTION for key presses/hotkeys
class PressAction(CLAAction):
    type: Literal["press"] = "press"
    keys: list[CLAKey]

    def __str__(self) -> str:
        action_str = f"🎹 Press keys: {'+'.join(self.keys)}"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# KEYDOWN ACTION for key presses/hotkeys
class KeyDownAction(CLAAction):
    type: Literal["keydown"] = "keydown"
    keys: list[CLAKey]

    def __str__(self) -> str:
        action_str = f"👇 KeyDown: {'+'.join(self.keys)}"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# KEYUP ACTION for key presses/hotkeys
class KeyUpAction(CLAAction):
    type: Literal["keyup"] = "keyup"
    keys: list[CLAKey]

    def __str__(self) -> str:
        action_str = f"👆 KeyUp: {'+'.join(self.keys)}"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# TYPE ACTION for text typing
class TypeAction(CLAAction):
    type: Literal["type"] = "type"
    text: str
    enter_after: bool | None = False

    def __str__(self) -> str:
        action_str = f'✍️ Type: "{self.text}"'
        if self.enter_after:
            action_str += " (and press Enter)"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# SCROLL ACTION
class ScrollAction(CLAAction):
    type: Literal["scroll"] = "scroll"
    point: Point | None = None
    scroll: Point | None = None
    hold_keys: list[CLAKey] | None = None

    def __str__(self) -> str:
        parts = ["📄 Scroll"]
        if self.point:
            parts.append(f"at ({self.point.x}, {self.point.y})")
        if self.scroll:
            parts.append(f"by ({self.scroll.x}, {self.scroll.y})")
        if self.hold_keys:  # Added hold_keys for scroll
            parts.append(f"holding {self.hold_keys}")
        action_str = " ".join(parts)
        action_str += _format_logs_for_display(self.logs)
        return action_str


# MOVE ACTION for mouse movement
class MoveAction(CLAAction):
    type: Literal["move"] = "move"
    point: Point | None = None
    offset: Point | None = None

    def __str__(self) -> str:
        parts = ["✨ Move"]
        if self.point:
            parts.append(f"to ({self.point.x},{self.point.y})")
        if self.offset:
            parts.append(f"by ({self.offset.x},{self.offset.y})")
        action_str = " ".join(parts)
        action_str += _format_logs_for_display(self.logs)
        return action_str


# WAIT ACTION
class WaitAction(CLAAction):
    type: Literal["wait"] = "wait"
    time: int

    def __str__(self) -> str:
        action_str = f"💤 Wait for {self.time}ms"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# DRAG ACTION
class DragAction(CLAAction):
    type: Literal["drag"] = "drag"
    path: list[Point]
    pattern: list[int] | None = None  # [delay_1, delay_2, ...]
    hold_keys: list[CLAKey] | None = None

    def __str__(self) -> str:
        parts = ["🤏 Drag"]
        if self.path and len(self.path) > 0:
            if len(self.path) == 1:
                parts.append(f"at ({self.path[0].x},{self.path[0].y})")
            else:
                parts.append(
                    f"from ({self.path[0].x}, {self.path[0].y}) to "
                    f"({self.path[-1].x}, {self.path[-1].y})"
                )
        if self.hold_keys:  # Added hold_keys for drag
            parts.append(f"holding {self.hold_keys}")
        action_str = " ".join(parts)
        action_str += _format_logs_for_display(self.logs)
        return action_str


# RESPONSE ACTION from agent
class ResponseAction(CLAAction):
    type: Literal["response"] = "response"
    text: str  # The final textual response from the agent

    def __str__(self) -> str:
        displayed_text = self.text if len(self.text) < 50 else self.text[:47] + "..."
        action_str = f'💬 Response: "{displayed_text}"'
        action_str += _format_logs_for_display(self.logs)
        return action_str


# SCREENSHOT ACTION
class ScreenshotFetch(CLAAction):
    type: Literal["screenshot"] = "screenshot"

    def __str__(self) -> str:
        action_str = "📸 Screenshot"
        action_str += _format_logs_for_display(self.logs)
        return action_str


class PositionFetch(CLAAction):
    type: Literal["position"] = "position"

    def __str__(self) -> str:
        action_str = "📍 Position"
        action_str += _format_logs_for_display(self.logs)
        return action_str


class CustomAction(CLAAction):
    type: Literal["custom"] = "custom"
    action: str

    def __str__(self) -> str:
        action_str = f"⚙️ Custom: {self.action}"
        action_str += _format_logs_for_display(self.logs)
        return action_str


# Union of all possible actions
CLA = Annotated[
    ClickAction
    | PressAction
    | KeyDownAction
    | KeyUpAction
    | TypeAction
    | ResponseAction
    | ScrollAction
    | MoveAction
    | WaitAction
    | DragAction
    | CustomAction
    | ScreenshotFetch
    | PositionFetch,
    Field(discriminator="type"),
]


CLAKey: TypeAlias = Literal[
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
    "start",
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


CLAButton: TypeAlias = Literal["left", "right", "middle", "back", "forward"]
