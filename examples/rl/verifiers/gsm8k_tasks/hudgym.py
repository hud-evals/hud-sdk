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
from hud.adapters.common.adapter import Adapter
from hud.adapters.common.types import ResponseAction

from copy import deepcopy

SYSTEM_PROMPT = system_prompt = (
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

@dataclass
class Stats:
    """Statistics tracking for HUDGym."""
    total_rollouts: int = 0
    successful_rollouts: int = 0
    failed_rollouts: int = 0
    total_reward: float = 0.0
    total_time: float = 0.0
    env_creation_time: float = 0.0
    env_cleanup_time: float = 0.0
    error_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    
    def record_rollout(self, reward: float, duration: float, successful: bool, error_type: Optional[str] = None, 
                      env_creation_time: float = 0.0, env_cleanup_time: float = 0.0):
        """Record a completed rollout."""
        self.total_rollouts += 1
        self.total_reward += reward
        self.total_time += duration
        self.env_creation_time += env_creation_time
        self.env_cleanup_time += env_cleanup_time
        
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
    
    @property
    def avg_env_creation_time(self) -> float:
        """Calculate average environment creation time."""
        return self.env_creation_time / self.total_rollouts if self.total_rollouts > 0 else 0.0
    
    @property
    def avg_env_cleanup_time(self) -> float:
        """Calculate average environment cleanup time."""
        return self.env_cleanup_time / self.total_rollouts if self.total_rollouts > 0 else 0.0
    
    def summary(self) -> str:
        """Get stats summary."""
        return (f"Rollouts: {self.total_rollouts} | "
                f"Success: {self.success_rate:.1%} | "
                f"Avg Reward: {self.avg_reward:.3f} | "
                f"Avg Time: {self.avg_time:.2f}s | "
                f"Env Creation: {self.avg_env_creation_time:.3f}s | "
                f"Env Cleanup: {self.avg_env_cleanup_time:.3f}s")

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BasicAdapter(Adapter):
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
            # Default action
            return {"type": "message", "text": action_str}


class HUDGym(vf.Environment):
    """HUD gym environment with statistics tracking."""
    
    def __init__(
        self,
        tasks: List[Task],
        adapter: Adapter,
        max_turns: int = 10,
        enable_stats: bool = True,
        **kwargs,
    ):
        logger.info(f"Initializing HUDGym with {len(tasks)} tasks, max_turns={max_turns}, enable_stats={enable_stats}")
        self.max_turns = max_turns
        self.tasks_map = {t.id: t for t in tasks}
        self.adapter = adapter
        self.stats = Stats() if enable_stats else None

        # Create dataset with questions, tasks and info dict
        questions = []
        task_ids = []
        answers = []
        info = []

        for t in self.tasks_map.values():
            questions.append(t.prompt)
            task_ids.append(t.id)
            answers.append(t.metadata.get("answer", ""))
            info.append({
            "metadata": t.metadata,
            })
        
        dataset = Dataset.from_dict({
            "question": questions,
            "task": task_ids,
            "answer": answers,
            "info": info,
        })
        logger.info(f"Created dataset with {len(dataset)} entries.")
        logger.info(f"Dataset sample: {dataset[0]}")
        
        rubric = vf.Rubric()
        
        def hud_reward_func(completion, **kwargs) -> float:
            """Get reward from HUD environment evaluation."""
            # The reward will be set in the state during rollout
            state = kwargs.get('state', {})
            reward = state.get('reward', 0.0)
            logger.info(f"hud_reward_func returning reward: {reward}")
            return reward
        
        rubric.add_reward_func(hud_reward_func)
        logger.info("Added hud_reward_func to rubric.")


        super().__init__(
            dataset=dataset,
            system_prompt=SYSTEM_PROMPT,
            rubric=rubric,
            **kwargs
        )
        logger.info("HUDGym initialization complete.")

    async def rollout(self,
                    client: AsyncOpenAI,
                    model: str,
                    prompt: Messages,
                    answer: str = "",
                    task: str = "default",
                    info: Info = {},
                    sampling_args: SamplingArgs = {},
                    **kwargs) -> Tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        logger.info(f"Starting rollout for task: {task} with model: {model}")

        is_completed = False
        state = {
            'prompt': prompt,
            'completion': [],
            'answer': answer,
            'task': task,
            'info': info,
            'responses': [],
        }
        logger.info(f"Initial state created: {state}")

        assert isinstance(prompt, list)
        completion: List[ChatMessage] = []
        conversation = deepcopy(prompt)
        turn = 1

        logger.info(f"Initializing HUD environment for task: {task}")
        current_task = self.tasks_map.get(task)
        logger.info(f"Current task: {current_task}")
        
        # Track timing for stats
        rollout_start_time = time.time()
        env_creation_start = time.time()
        
        hud_env = None
        try:
            hud_env = await gym.make(current_task) # type: ignore
            obs, _ = await hud_env.reset() # for now obs is the same as prompt, no screenshots
            env_creation_time = time.time() - env_creation_start
            logger.info(f"HUD environment created in {env_creation_time:.3f}s. Initial observation: {obs}")
            
            while not is_completed and turn < self.max_turns:
                logger.info(f"Rollout turn {turn}")

                # Get model response
                logger.info("Calling get_model_response...")
                response = await self.get_model_response(
                    prompt=conversation,
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    message_type='chat'
                )
                state['responses'].append(response)
                logger.info("Received model response.")

                assert isinstance(response, ChatCompletion)
                response_text: str = response.choices[0].message.content or ""
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text
                }
                conversation.append(response_message)
                completion.append(response_message)
                logger.info(f"Appended assistant message to conversation: {response_message}")

                # Process the assistant's action with the environment
                try:
                    # Extract and adapt actions
                    raw_actions = [response_text]
                    actions = self.adapter.adapt_list(raw_actions)
                    logger.info(f"Adapted actions: {actions}")

                    # Step the environment
                    next_obs, _, _, _ = await hud_env.step(actions)
                    logger.info(f"Environment step complete.")

                    turn += 1
                    
                    # Check if task is complete
                    if isinstance(actions[0], ResponseAction):
                        # Evaluate and get reward
                        try:
                            eval_result = await hud_env.evaluate()
                            reward = float(eval_result.get("reward", 0.0))
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")
                            reward = 0.0
                        
                        state['reward'] = reward
                        logger.info(f"Task {task} completed with reward: {reward}")
                        
                        # Task completed - stats will be recorded in finally block
                        
                        # Add final environment message
                        final_message: ChatMessage = {
                            "role": "user",
                            "content": f"Task completed. Final reward: {reward}"
                        }
                        conversation.append(final_message)
                        completion.append(final_message)
                        
                        is_completed = True
                    else:
                        # Continue with next observation
                        obs_content = next_obs.text if next_obs and hasattr(next_obs, 'text') else str(next_obs)
                        env_message: ChatMessage = {
                            "role": "user", 
                            "content": obs_content or "Continuing..."
                        }
                        conversation.append(env_message)
                        completion.append(env_message)
                        logger.info(f"Appended environment observation: {env_message}")
                        
                except Exception as e:
                    logger.error(f"Action processing failed: {e}")
                    
                    # Error will be recorded in finally block
                    
                    # Add error message
                    error_message: ChatMessage = {
                        "role": "user",
                        "content": f"Error: {str(e)}"
                    }
                    conversation.append(error_message)
                    completion.append(error_message)
                    
                    state['reward'] = 0.0
                    is_completed = True

            logger.info(f"Rollout finished. Returning completion with {len(completion)} messages and final state.")
            return completion, state
            
        finally:
            # Clean up environment and record stats
            env_cleanup_start = time.time()
            env_cleanup_time = 0.0
            
            if hud_env:
                await hud_env.close()
                env_cleanup_time = time.time() - env_cleanup_start
                logger.info(f"Environment cleanup completed in {env_cleanup_time:.3f}s")
            
            # Record rollout stats (including cleanup time in total duration)
            if self.stats:
                rollout_time = time.time() - rollout_start_time
                reward = state.get('reward', 0.0)
                successful = reward > 0 and is_completed
                error_type = None if successful else "UnknownError"
                
                self.stats.record_rollout(
                    reward=reward,
                    duration=rollout_time,
                    successful=successful,
                    error_type=error_type,
                    env_creation_time=env_creation_time if 'env_creation_time' in locals() else 0.0,
                    env_cleanup_time=env_cleanup_time
                )
                logger.info(f"Recorded rollout stats: duration={rollout_time:.3f}s, env_creation={env_creation_time if 'env_creation_time' in locals() else 0.0:.3f}s, env_cleanup={env_cleanup_time:.3f}s")

if __name__ == "__main__":
    from openai import OpenAI
    import os
    import json

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key) 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_file = os.path.join(script_dir, "gsm8k_tasks", "gsm8k_test.json")
    with open(tasks_file, 'r') as f:
        tasks_data = json.load(f)
    
    print(f"Testing {len(tasks_data)} tasks...")
    
    # Create all tasks
    tasks = [Task.from_dict(task_data) for task_data in tasks_data]

    # Create HUDGym instance
    hudgym = HUDGym(
        tasks=tasks,
        adapter=BasicAdapter(),
    )
    
    results = hudgym.evaluate(
        client=client,
        model="gpt-4.1-nano",
        max_concurrent=128,
    )

    # Format and print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("Example output:")
    # Extract key information
    question = results['question'][0]
    task_id = results['task'][0]
    reward = results['reward'][0]
    
    print(f"\nTask: {task_id}")
    print(f"Question: {question}")
    print(f"Final Reward: {reward}")
    
    # Show the conversation
    print("\nConversation:")
    print("-"*40)
    completion = results['completion'][0]
    for i, msg in enumerate(completion):
        role = msg['role'].upper()
        content = msg['content']
        print(f"\n[{role}]:")
        print(content)
    
    # Show statistics if available
    if hudgym.stats:
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(hudgym.stats.summary())
    
    print("\n")

    # Make dataset from results
    hudgym.make_dataset(results, push_to_hub=True, hub_name="jdchawla29/hud-gym-gsm8k-test-gpt-4.1-nano")