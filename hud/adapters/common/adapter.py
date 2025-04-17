from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from pydantic import TypeAdapter, ValidationError

from .types import CLA

if TYPE_CHECKING:
    from typing_extensions import TypeIs

ImageType = np.ndarray[Any, Any] | Image.Image | str | None


def _is_numpy_array(observation: Any) -> TypeIs[np.ndarray]:
    """Check if the observation is a numpy array, without requiring numpy."""
    try:
        import numpy as np  # type: ignore

        return isinstance(observation, np.ndarray)
    except (ModuleNotFoundError, NameError):
        return False


class Adapter:
    def __init__(self) -> None:
        self.memory = []

        self.agent_width = 1920
        self.agent_height = 1080
        self.env_width = 1920
        self.env_height = 1080

    def preprocess(self, action: Any) -> Any:
        return action

    def convert(self, action: Any) -> CLA:
        if action is None:
            raise ValueError("Please provide a valid action")
        try:
            return TypeAdapter(CLA).validate_python(action)
        except ValidationError as e:
            raise ValueError(f"Invalid action type in conversion: {action}") from e

    def json(self, action: CLA) -> Any:
        if action is None:
            raise ValueError("Please provide a valid action")
        try:
            validated = TypeAdapter(CLA).validate_python(action)
            return validated.model_dump()
        except ValidationError as e:
            raise ValueError(f"Invalid action type in json creation: {action}") from e

    def rescale(self, observation: ImageType) -> str | None:
        """
        Resize the observation (image) to agent-specific dimensions.

        Args:
            observation: Image data, which can be:
                - numpy array
                - PIL Image
                - base64 string (PNG) # TODO: JPG

        Returns:
            Base64-encoded string of the resized image (PNG format)
        """
        if observation is None:
            return None

        # Handle different input types.
        if _is_numpy_array(observation):
            # Convert numpy array to PIL Image
            img = Image.fromarray(observation)
        elif isinstance(observation, Image.Image):
            img = observation
        elif isinstance(observation, str):
            # Assume it's a base64 string
            try:
                import base64
                import io

                # Remove header if present (e.g., 'data:image/png;base64,')
                if "," in observation:
                    observation = observation.split(",")[1]
                # Decode base64 string to bytes
                img_bytes = base64.b64decode(observation)
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {e}") from None
        else:
            raise ValueError(f"Unsupported observation type: {type(observation)}")

        # Update environment dimensions
        self.env_width, self.env_height = img.size

        # Resize to agent dimensions
        resized_img = img.resize((self.agent_width, self.agent_height), Image.Resampling.LANCZOS)

        # Always convert to base64 string
        import base64
        import io

        buffered = io.BytesIO()
        resized_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def postprocess_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Rescale action coordinates from agent dimensions to environment dimensions.

        Args:
            action: Action dictionary with coordinates

        Returns:
            Action with rescaled coordinates
        """
        if not action:
            return action

        # Calculate scaling factors
        x_scale = self.env_width / self.agent_width
        y_scale = self.env_height / self.agent_height

        # Deep copy to avoid modifying the original
        processed_action = action.copy()

        # Rescale based on action type and structure
        if "point" in processed_action and processed_action["point"] is not None:
            # For actions with a single point (click, move)
            processed_action["point"]["x"] = int(processed_action["point"]["x"] * x_scale)
            processed_action["point"]["y"] = int(processed_action["point"]["y"] * y_scale)

        if (path := processed_action.get("path")) is not None:
            # For actions with a path (drag)
            for point in path:
                point["x"] = int(point["x"] * x_scale)
                point["y"] = int(point["y"] * y_scale)

        if "scroll" in processed_action and processed_action["scroll"] is not None:
            # For scroll actions
            processed_action["scroll"]["x"] = int(processed_action["scroll"]["x"] * x_scale)
            processed_action["scroll"]["y"] = int(processed_action["scroll"]["y"] * y_scale)

        return processed_action

    def adapt(self, action: Any) -> CLA:
        # any preprocessing steps
        action = self.preprocess(action)

        # convert to CLA
        action = self.convert(action)
        self.memory.append(action)

        # convert to json and apply coordinate rescaling
        action_dict = self.json(action)
        rescaled_action = self.postprocess_action(action_dict)

        # convert back to CLA
        return TypeAdapter(CLA).validate_python(rescaled_action)

    def adapt_list(self, actions: list[Any]) -> list[CLA]:
        if not isinstance(actions, list):
            raise ValueError("Please provide a list of actions")
        
        return [self.adapt(action) for action in actions]
