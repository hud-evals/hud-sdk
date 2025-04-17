from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Generic

from hud.adapters import Adapter, CLA
from hud.env.environment import Observation

# Generic type for different client types (Anthropic, OpenAI, etc.)
ClientT = TypeVar('ClientT')
ActionT = TypeVar('ActionT')

class Agent(Generic[ClientT, ActionT], ABC):
    """
    Base class for all agents.
    
    Implements a three-stage prediction process:
    1. preprocess - Prepare observation data (e.g., rescale screenshot)
    2. fetch_response - Make API calls to get model response
    3. postprocess - Convert model actions to HUD format
    
    Subclasses only need to implement the fetch_response method.
    """
    
    def __init__(self, client: ClientT | None = None, adapter: Adapter | None = None):
        """
        Initialize the agent.
        
        Args:
            client: The client to use for API calls
            adapter: The adapter to use for preprocessing and postprocessing
        """
        self.client = client
        self.adapter = adapter
    
    def preprocess(self, observation: Observation) -> Observation:
        """
        Preprocess the observation before sending to the model.
        
        Args:
            observation: The raw observation from the environment
            
        Returns:
            Observation: The processed observation ready for the model
        """
        if not self.adapter or not observation.screenshot:
            return observation
            
        # Create a new observation with the rescaled screenshot
        processed_obs = Observation(
            text=observation.text,
            screenshot=self.adapter.rescale(observation.screenshot)
        )
        return processed_obs
    
    @abstractmethod
    async def fetch_response(self, observation: Observation) -> tuple[list[ActionT], bool]:
        """
        Fetch a response from the model based on the observation.
        
        Args:
            observation: The preprocessed observation
            
        Returns:
            tuple[list[ActionT], bool]: A tuple containing the list of raw actions and a
                                       boolean indicating if the agent believes it has
                                       completed the task
        """
        pass
    
    def postprocess(self, actions: list[ActionT]) -> list[CLA]:
        """
        Convert model actions to HUD actions.
        
        Args:
            actions: The raw actions from the model
            
        Returns:
            Sequence[CLA]: The actions converted to HUD format
        """
        if not self.adapter:
            raise ValueError("Cannot postprocess actions without an adapter")
        
        return self.adapter.adapt_list(actions)
    
    async def predict(self, observation: Observation) -> tuple[list[CLA] | list[ActionT], bool]:
        """
        Predict the next action based on the observation.
        
        Implements the full three-stage prediction process.
        
        Args:
            observation: The observation from the environment
            
        Returns:
            tuple[list[CLA] | list[ActionT], bool]: A tuple containing the list of actions and a boolean
                                                       indicating if the agent believes it has completed the task
        """
        # Stage 1: Preprocess the observation
        processed_obs = self.preprocess(observation)
        
        # Stage 2: Fetch response from the model
        actions, done = await self.fetch_response(processed_obs)
        
        # Stage 3: Postprocess the actions if we have an adapter
        if self.adapter and actions:
            hud_actions = self.postprocess(actions)
            return hud_actions, done
            
        # If no adapter, return actions as is
        return actions, done