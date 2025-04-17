"""
PyAutoGUI adapter implementation for Rosetta.
Converts CLA actions to string representations of PyAutoGUI commands.
"""
from __future__ import annotations

import time  # Add missing import for time.sleep()
from typing import Any

import pyautogui  # Add pyautogui import

# PyAutoGUI button mapping reference
# BUTTON_NAME_MAPPING = {LEFT: 1, MIDDLE: 2, RIGHT: 3, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
VALID_BUTTON_VALUES = [
    "left", "middle", "right", "1", "2", "3", "4", "5", "6", "7", "back", "forward"]

# Map 'back' to button 4 and 'forward' to button 5 (common convention)
BUTTON_NAME_MAP = {
    "back": "4",
    "forward": "5",
    "wheel": "middle"
}


class RosettaBase:
    """
    Base class for Rosetta adapters that convert from CLA to specific action formats.
    Opposite of the Adapter class in hud-gym/hud/adapters/common.
    """
    
    def __init__(self) -> None:
        """Initialize the adapter."""
        self.memory = []
    
    def preprocess(self, cla_action: dict[str, Any]) -> dict[str, Any]:
        """
        Optional preprocessing of CLA action before conversion.
        Can be overridden by subclasses.
        
        Args:
            cla_action: A CLA action in dictionary format
            
        Returns:
            Preprocessed CLA action
        """
        return cla_action
    
    def convert(self, cla_action: dict[str, Any]) -> Any:
        """
        Convert a CLA action to a specific format.
        Must be implemented by subclasses.
        
        Args:
            cla_action: A CLA action in dictionary format
            
        Returns:
            Action in the target format
        """
        raise NotImplementedError("Subclasses must implement convert()")
    
    def translate(self, cla_action: dict[str, Any]) -> Any:
        """
        Translate a CLA action to a specific format and execute it.
        
        Args:
            cla_action: A CLA action in dictionary format
            
        Returns:
            Result of the execution
        """
        cla_action = self.preprocess(cla_action)
        target_action = self.convert(cla_action)
        self.memory.append(target_action)
        return {"command": target_action}
    
    def translate_list(self, cla_actions: list[dict[str, Any]]) -> list[Any]:
        """
        Translate a list of CLA actions and execute them.
        
        Args:
            cla_actions: A list of CLA actions in dictionary format
            
        Returns:
            list of execution results
        """
        #assert isinstance(cla_actions, list)
        return [self.translate(action) for action in cla_actions]


class PyAutoGUIRosetta(RosettaBase):
    """
    PyAutoGUI adapter for Rosetta.
    Executes PyAutoGUI commands based on CLA actions.
    """
    
    def __init__(self) -> None:
        """Initialize the PyAutoGUI adapter."""
        super().__init__()
        
    def execute(self, cla_action: dict[str, Any]) -> None:
        """
        Execute a PyAutoGUI command based on a CLA action.
        
        Args:
            cla_action: A CLA action in dictionary format
        """
        action_type = cla_action.get("type")
        if not action_type:
            raise ValueError("Action must have a 'type' field")
        
        # Check for unsupported 'selector' field
        if cla_action.get("selector"):
            raise ValueError("Selector-based actions are not supported by PyAutoGUI adapter")
        
        match action_type:
            case "click":
                self._execute_click(cla_action)
            case "press":
                self._execute_press(cla_action)
            case "keyup":
                self._execute_key_up(cla_action)
            case "keydown":
                self._execute_key_down(cla_action)
            case "type":
                self._execute_type(cla_action)
            case "scroll":
                self._execute_scroll(cla_action)
            case "move":
                self._execute_move(cla_action)
            case "wait":
                self._execute_wait(cla_action)
            case "drag":
                self._execute_drag(cla_action)
            case "screenshot":
                # Currently does nothing, matching original behavior.
                # Could be changed to pyautogui.screenshot() if needed.
                pass
            case "position":
                # Original returned code string, now raises error as it's not an execution action.
                raise ValueError("Position 'action' cannot be executed, it retrieves information.")
            case "custom":
                # Executing arbitrary custom code is unsafe. Raise error.
                raise ValueError("Custom actions are not supported for direct execution "
                                 "due to security concerns.")
            case _:
                raise ValueError(f"Unsupported action type: {action_type}")
    
    def execute_sequence(self, cla_actions: list[dict[str, Any]]) -> None:
        """
        Execute a sequence of PyAutoGUI commands based on CLA actions.
        
        Args:
            cla_actions: A list of CLA actions in dictionary format
        """
        for action in cla_actions:
            self.execute(action)
    
    def _execute_key_down(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI keyDown command."""
        for key in cla_action.get("keys", []):
            pyautogui.keyDown(key)

    def _execute_key_up(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI keyUp command."""
        for key in reversed(cla_action.get("keys", [])):
            pyautogui.keyUp(key)

    def _execute_click(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI click command."""
        point = cla_action.get("point")
        button = cla_action.get("button", "left")
        pattern = cla_action.get("pattern")
        hold_keys = cla_action.get("hold_keys", [])
        
        if not point:
            raise ValueError("Click action must have a 'point' field")
        
        x = point.get("x")
        y = point.get("y")
        
        # Map 'back' and 'forward' to their numerical equivalents
        if button in BUTTON_NAME_MAP:
            button = BUTTON_NAME_MAP[button]
        
        # Validate button value
        if button not in VALID_BUTTON_VALUES:
            raise ValueError(f"Invalid button value '{button}'. "
                             f"Valid values are: {', '.join(VALID_BUTTON_VALUES)}")
        
        # Execute key down commands if hold_keys is specified
        if hold_keys:
            self._execute_key_down({"keys": hold_keys})
        
        # Handle clicks and intervals based on pattern
        if pattern and len(pattern) > 0:
            if len(pattern) == 1:
                # For a single pattern value (double-click), use clicks parameter
                interval = pattern[0] / 1000.0
                pyautogui.click(x=x, y=y, clicks=2, interval=interval, button=button)
            else:
                # For multiple pattern values, generate individual click commands with precise
                # First click
                pyautogui.click(x=x, y=y, button=button)
                
                # Subsequent clicks with waits in between
                for delay in pattern:
                    # Convert milliseconds to seconds for the sleep command
                    sleep_time = delay / 1000.0
                    time.sleep(sleep_time)
                    pyautogui.click(x=x, y=y, button=button)
        else:
            # Simple click
            pyautogui.click(x=x, y=y, button=button)
        
        # Execute key up commands if hold_keys is specified
        if hold_keys:
            self._execute_key_up({"keys": hold_keys})

    def _execute_press(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI press command."""
        keys = cla_action.get("keys")
        
        if not keys:
            raise ValueError("Press action must have a 'keys' field")
        
        if len(keys) == 1:
            # Single key press
            pyautogui.press(keys[0])
        else:
            # Multiple keys (hotkey)
            pyautogui.hotkey(*keys) # Pass keys as separate arguments

    def _execute_type(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI type command."""
        text = cla_action.get("text")
        enter_after = cla_action.get("enter_after", False)
        
        if text is None:
            raise ValueError("Type action must have a 'text' field")
        
        # Handle empty text
        if not text:
            if enter_after:
                pyautogui.press("enter")
            return # Do nothing if text is empty and no enter_after
        
        # Directly type the text
        pyautogui.typewrite(text)
        
        # Add press enter if requested
        if enter_after:
            pyautogui.press("enter")

    def _execute_scroll(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI scroll command."""
        point = cla_action.get("point")
        scroll = cla_action.get("scroll")
        hold_keys = cla_action.get("hold_keys", [])
        
        if not scroll:
            raise ValueError("Scroll action must have a 'scroll' field")
        
        # Execute key down commands if hold_keys is specified
        if hold_keys:
            self._execute_key_down({"keys": hold_keys})
        
        # Move to point first if specified
        if point:
            x = point.get("x")
            y = point.get("y")
            pyautogui.moveTo(x=x, y=y)
        
        scroll_x = scroll.get("x", 0)
        scroll_y = scroll.get("y", 0)
        
        if scroll_y != 0:
            pyautogui.scroll(clicks=scroll_y)
        elif scroll_x != 0:
            pyautogui.hscroll(clicks=scroll_x)
        
        # Execute key up commands if hold_keys is specified
        if hold_keys:
            self._execute_key_up({"keys": hold_keys})

    def _execute_move(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI move command."""
        point = cla_action.get("point")
        offset = cla_action.get("offset")
        
        if point:
            x = point.get("x")
            y = point.get("y")
            pyautogui.moveTo(x=x, y=y, duration=0.1)
        elif offset:
            x_offset = offset.get("x", 0)
            y_offset = offset.get("y", 0)
            pyautogui.moveRel(xOffset=x_offset, yOffset=y_offset, duration=0.1)
        else:
            raise ValueError("Move action must have 'point' or 'offset' field")

    def _execute_wait(self, cla_action: dict[str, Any]) -> None:
        """Execute a time.sleep command."""
        time_ms = cla_action.get("time", 1000)  # Default to 1 second
        seconds = time_ms / 1000.0
        
        time.sleep(seconds)

    def _execute_drag(self, cla_action: dict[str, Any]) -> None:
        """Execute a PyAutoGUI drag command."""
        path = cla_action.get("path", [])
        pattern = cla_action.get("pattern", [])
        hold_keys = cla_action.get("hold_keys", [])
        
        if not path or len(path) < 2:
            raise ValueError("Drag action must have a 'path' field with at least 2 points")
        
        # Execute key down commands if hold_keys is specified
        if hold_keys:
            self._execute_key_down({"keys": hold_keys})
        
        # Move to first position
        start_x = path[0].get("x")
        start_y = path[0].get("y")
        pyautogui.moveTo(x=start_x, y=start_y, duration=0.1)
        
        # Handle path with pattern if provided
        if pattern and len(pattern) > 0:
            # First drag to second point
            x = path[1].get("x")
            y = path[1].get("y")
            pyautogui.dragTo(x=x, y=y, duration=0.1)
            
            # For subsequent points, add delays between drags
            pattern_index = 0
            for i in range(2, len(path)):
                # Add delay based on pattern
                if pattern_index < len(pattern):
                    delay = pattern[pattern_index] / 1000.0  # Convert ms to seconds
                    time.sleep(delay)
                    pattern_index = (pattern_index + 1) % len(pattern)
                
                # Add next drag
                point = path[i]
                x = point.get("x")
                y = point.get("y")
                pyautogui.dragTo(x=x, y=y, duration=0.1)
        else:
            # Handle multiple waypoints without pattern
            if len(path) > 2:
                # Generate individual dragTo commands for each waypoint
                for i in range(1, len(path)):
                    point = path[i]
                    x = point.get("x")
                    y = point.get("y")
                    duration = 0.2 / (len(path) - 1)  # Distribute duration evenly
                    pyautogui.dragTo(x=x, y=y, duration=duration)
            else:
                # Simple start to end drag
                end_x = path[-1].get("x")
                end_y = path[-1].get("y")
                pyautogui.dragTo(x=end_x, y=end_y, duration=0.2)
        
        # Execute key up commands if hold_keys is specified
        if hold_keys:
            self._execute_key_up({"keys": hold_keys})
