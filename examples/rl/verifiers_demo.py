import asyncio
import logging
from typing import Dict, Tuple, List
from openai import AsyncOpenAI, OpenAI
from datasets import Dataset

import verifiers as vf
from typing import Dict, Any, List, Union

# Define type aliases
Messages = Union[str, List[Dict[str, str]]]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
Info = Dict[str, Any]

from hud.task import Task
import hud.gym as gym
from hud.adapters import Adapter

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HUDGym(vf.Environment):
    """HUD gym environment."""
    
    def __init__(
        self,
        tasks: List[Task],
        adapter: Adapter,
        client: OpenAI | None = None,
        model: str | None = None,
        max_steps: int = 10,
        **kwargs,
    ):
        """Initialize with a single HUD task and adapter."""
        # Create system prompt with thinking and action tags
        system_prompt = (
            "You are an AI assistant that helps users complete computer tasks. "
            "You can see screenshots and perform actions to interact with interfaces.\n\n"
            "In each turn, think step-by-step inside <think>...</think> tags, "
            "then perform actions inside <action>...</action> tags.\n\n"
            "Available actions:\n"
            "- click x,y (click at coordinates)\n"
            "- type \"text\" (type the given text)\n"
            "- scroll up/down (scroll in direction)\n"
            "- message \"text\" (send a message or response)\n"
            "- done \"response\" (task complete with final response)\n\n"
            "Here is an example of how to use the tools:\n\n"
            "--- Example --- \n"
            "Observation: Question: What is 2+2?\nPlease provide your answer using a response action.\n\n"
            "<think>The user is asking a simple math question. The sum of 2 and 2 is 4..</think>\n"
            "<action>done \"4\"</action>\n"
            "--- End Example ---\n\n"
            "Always use the required format with thinking and action tags to answer the question."
        )
        
        # Create dataset with system prompt for all tasks
        prompts = []
        answers = []
        task_ids = []
        
        for t in tasks:
            prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": t.prompt}
            ])
            answers.append("")
            task_ids.append(t.id or "default")
        
        dataset = Dataset.from_dict({
            "prompt": prompts,
            "answer": answers,
            "task": task_ids,
        })
        
        rubric = vf.Rubric()
        
        def hud_reward_func(completion, **kwargs) -> float:
            """Get reward from HUD environment evaluation."""
            # The reward will be set in the state during rollout
            state = kwargs.get('state', {})
            return state.get('reward', 0.0)
        
        rubric.add_reward_func(hud_reward_func)
        
        super().__init__(
            client=client,
            model=model,
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )
        self.tasks = tasks
        self.task_map = {t.id or f"task_{i}": t for i, t in enumerate(tasks)}
        self.adapter = adapter
        self.max_steps = max_steps
    
    def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
        """Synchronous rollout method compatible with verifiers."""
        # Create async client for gym operations
        async_client = AsyncOpenAI(api_key=client.api_key)
        
        # Run the async rollout in a new event loop
        return asyncio.run(self._async_rollout(client, async_client, model, prompt, answer, task, info, sampling_args, **kwargs))
    
    async def _async_rollout(
        self,
        sync_client: OpenAI,
        async_client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
        """Multi-turn rollout using gym.make and env.step."""
        
        logger.info(f"Starting rollout for task: {task}")
        
        # Get the specific task for this rollout
        current_task = self.task_map.get(task, self.tasks[0])
        
        # Create HUD environment
        hud_env = await gym.make(current_task)
        
        try:
            # Reset environment
            obs, _ = await hud_env.reset()
            
            conversation = prompt.copy() if isinstance(prompt, list) else []
            all_responses = []
            
            for step in range(self.max_steps):
                # Add observation to conversation
                if obs and obs.text:
                    conversation.append({"role": "user", "content": obs.text})
                
                # Get model response
                response = self.get_model_response(
                    prompt=conversation,
                    client=sync_client,
                    model=model,
                    sampling_args=sampling_args,
                    message_type="chat",
                )
                
                if response is None:
                    logger.warning("Model returned no response. Ending rollout.")
                    break
                    
                assistant_content = response or ""
                conversation.append({"role": "assistant", "content": assistant_content})
                all_responses.append(assistant_content)
                
                try:
                    # Extract actions from assistant response
                    # For now, treat the entire response as a single action
                    raw_actions = [assistant_content]  # Raw model output
                    actions = self.adapter.adapt_list(raw_actions)  # Convert to HUD actions
                except Exception as e:
                    logger.error(f"Action adaptation failed: {e}")
                    break # exit if no valid actions
                
                # Step environment
                next_obs, _, env_done, _ = await hud_env.step(actions)
                obs = next_obs
                
                # Check if done
                if env_done or "done" in assistant_content.lower():
                    logger.info("Done condition met. Ending rollout.")
                    break
            
            # Evaluate and get reward
            try:
                eval_result = await hud_env.evaluate()
                reward = float(eval_result.get("reward", 0.0))
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                reward = 0.0
            
            # Return final completion and state
            completion = conversation[-1:] if conversation else [{"role": "assistant", "content": ""}]
            
            state = {
                "responses": all_responses,
                "done": True,
                "conversation": conversation,
                "hud_env": hud_env,
                "reward": reward,
                "final_observation": obs.dict() if obs and hasattr(obs, 'dict') else obs,
            }
            
            logger.info(f"Rollout for task {task} finished with reward: {reward}")
            return completion, state
            
        finally:
            await hud_env.close()
    
    @classmethod
    def from_task_json(cls, task_json: Dict, adapter: Adapter, **kwargs) -> "HUDGym":
        """Create HUDGym from task JSON specification."""
        task = Task.from_dict(task_json)
        return cls(task=task, adapter=adapter, **kwargs)
