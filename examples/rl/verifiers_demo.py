import asyncio
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union, Tuple
from openai import AsyncOpenAI
from datasets import Dataset
import time
from dataclasses import dataclass, field
from collections import defaultdict

import verifiers as vf
from typing import Dict, Any, List, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

# verifier type aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Union[Completion, ChatCompletion, None]
ChatMessageField = Literal["role", "content"]
ChatMessage = Dict[ChatMessageField, str]
Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]


from hud.task import Task
import hud.gym as gym
from hud.adapters import Adapter
from hud.adapters.common.adapter import Adapter as BaseAdapter


@dataclass
class Stats:
    """Statistics tracking for HUDGym."""
    total_rollouts: int = 0
    successful_rollouts: int = 0
    failed_rollouts: int = 0
    total_reward: float = 0.0
    total_time: float = 0.0
    error_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    
    def record_rollout(self, reward: float, duration: float, successful: bool, error_type: str = None):
        """Record a completed rollout."""
        self.total_rollouts += 1
        self.total_reward += reward
        self.total_time += duration
        
        if successful:
            self.successful_rollouts += 1
        else:
            self.failed_rollouts += 1
            if error_type:
                self.error_counts[error_type] += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_rollouts / self.total_rollouts if self.total_rollouts > 0 else 0.0
    
    @property
    def avg_reward(self) -> float:
        """Calculate average reward."""
        return self.total_reward / self.total_rollouts if self.total_rollouts > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        """Calculate average rollout time."""
        return self.total_time / self.total_rollouts if self.total_rollouts > 0 else 0.0
    
    def summary(self) -> str:
        """Get stats summary."""
        return (f"Rollouts: {self.total_rollouts} | "
                f"Success: {self.success_rate:.1%} | "
                f"Avg Reward: {self.avg_reward:.3f} | "
                f"Avg Time: {self.avg_time:.2f}s")

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BasicAdapter(BaseAdapter):
    """Adapter that extracts actions from model output."""
    
    def preprocess(self, model_output: str) -> str:
        """Extract action from model output."""
        # Look for action tags
        if "<action>" in model_output and "</action>" in model_output:
            start = model_output.find("<action>") + 8
            end = model_output.find("</action>")
            action = model_output[start:end].strip()
            return action
        
        # Look for basic commands
        lines = model_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if any(cmd in line.lower() for cmd in ['click', 'type', 'scroll', 'message', 'done']):
                return line
        
        # Default: return the whole output
        return model_output.strip()
    
    def convert(self, action_str: str):
        """Convert action string to CLA format."""
        # Parse basic action commands
        action_str = action_str.strip().lower()
        
        if action_str.startswith('click'):
            # Click action
            return {"type": "click", "point": {"x": 100, "y": 100}}
        elif action_str.startswith('type'):
            # Extract text to type
            if '"' in action_str:
                text = action_str.split('"')[1]
                return {"type": "type", "text": text}
            else:
                return {"type": "type", "text": ""}
        elif action_str.startswith('done'):
            # Extract response text if provided       
            if '"' in action_str:
                text = action_str.split('"')[1]
                return {"type": "response", "text": text}
            else:
                return {"type": "response", "text": "Task completed"}
        else:
            # if no recognized action, we need to end the task
            return {}  # will throw an error


class HUDGym(vf.MultiTurnEnv):
    """HUD gym environment with statistics tracking."""
    
    def __init__(
        self,
        tasks: List[Task],
        adapter: Adapter,
        max_turns: int = 10,
        enable_stats: bool = True,
        **kwargs,
    ):
        self.tasks = tasks
        self.task_map = {t.id or f"task_{i}": t for i, t in enumerate(tasks)}
        self.adapter = adapter
        self.max_turns = max_turns
        self.stats = Stats() if enable_stats else None
        self.message_type = "chat"

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
            task_ids.append(t.id)
        
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
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            max_turns=max_turns,
            **kwargs,
        )
    
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the multi-turn interaction is complete."""
        # Check if environment is done
        if state.get('env_done', False):
            return True
            
        # Check if last assistant message contains "done" action
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            if (isinstance(last_message, dict) and 
                last_message.get('role') == 'assistant' and 
                'done' in last_message.get('content', '').lower()):
                return True
        
        return False
    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Message, State]:
        """Process assistant's action and return environment observation."""
        task_id = state.get('task', 'default')
        
        # Initialize environment if this is the first call
        if 'hud_env' not in state:
            logger.info(f"Initializing HUD environment for task: {task_id}")
            current_task = self.task_map.get(task_id, list(self.task_map.values())[0])
            hud_env = await gym.make(current_task)
            obs, _ = await hud_env.reset()
            
            state['hud_env'] = hud_env
            state['step_count'] = 0
            state['env_done'] = False
            state['rollout_start_time'] = time.time()
            
            # Return initial observation
            obs_content = obs.text if obs and hasattr(obs, 'text') else str(obs)
            return {"role": "user", "content": obs_content or ""}, state
        
        # Process assistant's last message
        hud_env = state['hud_env']
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            if isinstance(last_message, dict) and last_message.get('role') == 'assistant':
                assistant_content = last_message.get('content', '')
                
                try:
                    # Extract actions from assistant response
                    raw_actions = [assistant_content]
                    actions = self.adapter.adapt_list(raw_actions)
                    
                    # Step environment
                    next_obs, _, env_done, _ = await hud_env.step(actions)
                    
                    # Update state
                    state['step_count'] = state.get('step_count', 0) + 1
                    state['env_done'] = env_done
                    
                    # Check if done
                    if env_done or "done" in assistant_content.lower():
                        # Evaluate and get reward
                        try:
                            eval_result = await hud_env.evaluate()
                            reward = float(eval_result.get("reward", 0.0))
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")
                            reward = 0.0
                        
                        state['reward'] = reward
                        state['env_done'] = True
                        
                        # Record stats
                        if self.stats:
                            rollout_time = time.time() - state.get('rollout_start_time', 0)
                            self.stats.record_rollout(
                                reward=reward,
                                duration=rollout_time,
                                successful=reward > 0,
                                error_type=None
                            )
                        
                        # Close environment
                        await hud_env.close()
                        
                        logger.info(f"Task {task_id} completed with reward: {reward}")
                        return {"role": "user", "content": f"Task completed. Final reward: {reward}"}, state
                    
                    # Return next observation
                    obs_content = next_obs.text if next_obs and hasattr(next_obs, 'text') else str(next_obs)
                    return {"role": "user", "content": obs_content or ""}, state
                    
                except Exception as e:
                    logger.error(f"Action processing failed: {e}")
                    state['env_done'] = True
                    state['error_type'] = type(e).__name__
                    
                    # Record failed rollout
                    if self.stats:
                        rollout_time = time.time() - state.get('rollout_start_time', 0)
                        self.stats.record_rollout(
                            reward=0.0,
                            duration=rollout_time,
                            successful=False,
                            error_type=type(e).__name__
                        )
                    
                    await hud_env.close()
                    return {"role": "user", "content": f"Error: {str(e)}"}, state
        
        # Fallback
        return {"role": "user", "content": "No valid response generated."}, state

    
    def get_stats_summary(self) -> str:
        """Get formatted statistics summary."""
        if self.stats:
            return self.stats.summary()
        return "Statistics tracking is disabled."
    
    @classmethod
    def from_task_json(cls, task_json: Dict, adapter: Adapter, **kwargs) -> "HUDGym":
        """Create HUDGym from task JSON specification."""
        task = Task.from_dict(task_json)
        return cls(tasks=[task], adapter=adapter, **kwargs)