from abc import ABC, abstractmethod
from hud.adapters.common.types import CLA
from hud.env.environment import Observation


class Agent(ABC):
    @abstractmethod
    def predict(self, observation: Observation) -> tuple[list[CLA], bool]:
        """Predict the next action based on the observation.
        
        Returns:
            tuple[list[CLA], bool]: A tuple containing the list of actions and a boolean indicating if the agent believes it has completed the task.
        """
        pass