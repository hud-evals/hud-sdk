from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from datasets import Dataset
import time
import yaml
from pathlib import Path
import asyncio

import verifiers as vf
import os

from openai.types.chat.chat_completion import ChatCompletion

from verifiers import (
    MessageType, ModelResponse, ChatMessage, Message, Messages,
    Info, State, SamplingArgs, RewardFunc
)


from hud.task import Task
import hud.gym as gym
from hud.adapters.common import Adapter
from hud.adapters.common.types import (
    ResponseAction, ClickAction, TypeAction, PressAction, 
    ScrollAction, WaitAction, ScreenshotFetch, Point, CLA
)

from copy import deepcopy
import json

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "hudgym_config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class HUDGymAdapter(Adapter):
    """Custom adapter for HUDGym that converts tool JSON to CLA actions."""
    
    def convert(self, action: Any) -> CLA:
        """Convert tool dictionary to CLA action."""
        if isinstance(action, str):
            action = json.loads(action.strip())
        
        assert isinstance(action, dict), f"Expected dict, got {type(action).__name__}"
        assert 'name' in action, "Action must have a 'name' field"
        
        tool_name = action['name'].lower()
        args = action.get('arguments', {})
        
        if tool_name == 'click':
            assert 'x' in args and 'y' in args
            return ClickAction(point=Point(x=args['x'], y=args['y']))
        elif tool_name == 'type':
            assert 'text' in args
            return TypeAction(text=args['text'], enter_after=False)
        elif tool_name == 'key':
            assert 'key' in args
            return PressAction(keys=[args['key'].lower()])
        elif tool_name == 'scroll':
            assert 'direction' in args and 'amount' in args
            direction = args['direction']
            amount = args['amount']
            scroll_y = -amount if direction == 'up' else amount
            return ScrollAction(scroll=Point(x=0, y=scroll_y))
        elif tool_name == 'wait':
            assert 'seconds' in args
            return WaitAction(time=int(args['seconds'] * 1000))
        elif tool_name == 'done':
            return ResponseAction(text="Task completed")
        elif tool_name == 'screenshot':
            return ScreenshotFetch()
        else:
            assert False, f"Unknown tool name: {tool_name}"

class HUDGym(vf.Environment):
    """HUD gym environment for verifiers integration."""
    
    def __init__(
        self,
        tasks: List[Task],
        adapter: Optional[Adapter] = None,
        eval_tasks: Optional[List[Task]] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        # Load configuration
        self.config = load_config(config_path)
        
        # Get defaults from config
        self.max_turns = kwargs.pop('max_turns', self.config['defaults']['max_turns'])
        self.adapter = adapter or HUDGymAdapter()
        
        # Store tasks for later use
        self.tasks = {t.id: t for t in tasks}
        if eval_tasks:
            self.eval_tasks = {t.id: t for t in eval_tasks}
            # Merge for lookup during rollout
            self.all_tasks = {**self.tasks, **self.eval_tasks}
        else:
            self.all_tasks = self.tasks
            
        # Create datasets
        dataset = Dataset.from_dict({
            "question": [t.prompt for t in tasks],
            "task": [t.id for t in tasks],
            "answer": [t.metadata.get("answer", "") for t in tasks],
            "info": [{"metadata": t.metadata} for t in tasks],
        })
        
        eval_dataset = None
        if eval_tasks:
            eval_dataset = Dataset.from_dict({
                "question": [t.prompt for t in eval_tasks],
                "task": [t.id for t in eval_tasks],
                "answer": [t.metadata.get("answer", "") for t in eval_tasks],
                "info": [{"metadata": t.metadata} for t in eval_tasks],
            })
        
        # Create rubric that gets reward from state (set during rollout)
        def hud_reward_func(completion, **kwargs) -> float:
            state = kwargs.get('state')
            assert state, "State not provided to reward function"
            assert 'reward' in state, "Reward not found in state"
            return state['reward']
        
        # XML parser for tool extraction
        self.tool_parser = vf.XMLParser(["think", "tool"])
        
        combined_rubric = vf.Rubric(
            funcs=[hud_reward_func, self.tool_parser.get_format_reward_func()],
            weights=[0.9, 0.1]
        )

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=self.config['system_prompt'],
            rubric=combined_rubric,
            **kwargs
        )
        
        # Track active cleanup tasks
        self.active_cleanups = set()
    
    async def _cleanup_env(self, env):
        """Background cleanup of HUD environment."""
        try:
            await env.close()
            self.logger.debug("Background environment cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during background cleanup: {e}")
    
    async def wait_for_cleanups(self):
        """Wait for all active cleanup tasks to complete."""
        if self.active_cleanups:
            self.logger.info(f"Waiting for {len(self.active_cleanups)} cleanup tasks...")
            await asyncio.gather(*self.active_cleanups, return_exceptions=True)
            self.logger.info("All cleanup tasks completed")

    
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
        self.logger.debug(f"Starting rollout for task: {task} with model: {model}")

        is_completed = False
        state = {
            'prompt': prompt,
            'completion': [],
            'answer': answer,
            'task': task,
            'info': info,
            'responses': [],
            'timing': {
                'env_creation_time': 0.0,
                'model_response_times': [],
                'env_step_times': [],
                'total_rollout_time': 0.0,
                'num_turns': 0
            },
            'error': None,  # Track any errors that occur
            'error_step': None  # Track where the error occurred
        }
        rollout_start_time = time.time()

        assert isinstance(prompt, list)
        completion: List[ChatMessage] = []
        conversation = deepcopy(prompt)
        turn = 1

        current_task = self.all_tasks.get(task)
        assert current_task, f"Task '{task}' not found in task list"
        
        env_creation_start = time.time()
        
        hud_env = None
        try:
            hud_env = await gym.make(current_task) # type: ignore
            obs, _ = await hud_env.reset()
            env_creation_time = time.time() - env_creation_start
            state['timing']['env_creation_time'] = env_creation_time
            self.logger.debug(f"HUD environment created in {env_creation_time:.3f}s")
            
            assert obs, "Environment reset did not return a valid observation"
            
            
            initial_message: ChatMessage = {
                "role": "user",
                "content": obs.text  # type: ignore
            }
            conversation.append(initial_message)
            completion.append(initial_message)
            self.logger.debug("Added initial observation")
            
            while not is_completed and turn < self.max_turns:
                self.logger.debug(f"Rollout turn {turn}")

                # Get model response
                self.logger.debug("Calling get_model_response...")
                model_start = time.time()
                response = await self.get_model_response(
                    prompt=conversation,
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                    message_type='chat'
                )
                state['responses'].append(response)
                model_duration = time.time() - model_start
                state['timing']['model_response_times'].append(model_duration)
                self.logger.debug(f"Model response received in {model_duration:.3f}s")

                assert isinstance(response, ChatCompletion)
                response_text = response.choices[0].message.content
                if not response_text:
                    raise ValueError("Model returned empty response")
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text
                }
                conversation.append(response_message)
                completion.append(response_message)
                self.logger.debug("Appended assistant message to conversation")
                
                # Extract tool from XML tags and convert to action
                parsed = self.tool_parser.parse(response_text)
                if not (hasattr(parsed, 'tool') and parsed.tool):
                    raise ValueError("No tool found in model response")
                action = self.adapter.convert(parsed.tool)
                actions = [action]
                
                # First action must be a screenshot fetch
                if turn == 1:
                    if not isinstance(action, ScreenshotFetch):
                        raise ValueError(f"First action must be a screenshot fetch, got {type(action).__name__}")

                # Check if task is complete
                if isinstance(actions[0], ResponseAction):
                    # Evaluate and get reward
                    eval_result = await hud_env.evaluate()
                    self.logger.info(f"Evaluation result: {eval_result}")
                    assert 'grade' in eval_result, "Evaluation result missing 'grade' field"
                    reward = float(eval_result['grade'])

                    state['reward'] = reward
                    self.logger.info(f"Task {task} completed with reward: {reward}")
                    
                    # Add final environment message
                    final_message: ChatMessage = {
                        "role": "user",
                        "content": f"Task completed. Env reward: {reward}"
                    }
                    conversation.append(final_message)
                    completion.append(final_message)
                    
                    is_completed = True
                    break
                
                # Step the environment
                step_start = time.time()
                next_obs, _, _, _ = await hud_env.step(actions)
                step_duration = time.time() - step_start
                state['timing']['env_step_times'].append(step_duration)
                self.logger.debug(f"Environment step complete in {step_duration:.3f}s.")
                
                turn += 1
                
                # Continue with next observation
                if next_obs.screenshot:
                    env_message: ChatMessage = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": next_obs.text if next_obs.text else "Continuing..."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{next_obs.screenshot}"}}
                        ]
                    }
                else:
                    self.logger.warning("No screenshot in observation, using text only")
                    env_message: ChatMessage = {"role": "user", "content": next_obs.text if next_obs.text else "Continuing..."}
                
                conversation.append(env_message)
                completion.append(env_message)
                self.logger.debug(f"Appended observation (has screenshot: {bool(next_obs.screenshot)})")
            
            if not is_completed and turn >= self.max_turns:
                state['reward'] = 0.0
                self.logger.warning(f"Task {task} reached max_turns ({self.max_turns}) without completion")
            
            state['timing']['num_turns'] = turn
            state['timing']['total_rollout_time'] = time.time() - rollout_start_time
            self.logger.debug(f"Rollout finished. Returning completion with {len(completion)} messages and final state.")
            return completion, state
        
        except Exception as e:
            self.logger.error(f"Error during rollout: {e}")
            state['error'] = str(e)
            state['error_step'] = f"turn_{turn}"
            
        finally:
            if hud_env:
                # Create async cleanup task
                cleanup_task = asyncio.create_task(self._cleanup_env(hud_env))
                self.active_cleanups.add(cleanup_task)
                cleanup_task.add_done_callback(
                    lambda t: self.active_cleanups.discard(t)
                )
                self.logger.debug("Environment cleanup started asynchronously")
            
            if 'reward' not in state:
                state["reward"] = 0.0
            state['timing']['total_rollout_time'] = time.time() - rollout_start_time
            return completion, state
    
    def compute_timing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregate timing statistics from evaluation results."""
        all_timings = [state['timing'] for state in results['state'] if 'timing' in state]
        
        if not all_timings:
            return {}
        
        stats = {
            'avg_env_creation_time': sum(t['env_creation_time'] for t in all_timings) / len(all_timings),
            'avg_total_rollout_time': sum(t['total_rollout_time'] for t in all_timings) / len(all_timings),
            'avg_num_turns': sum(t['num_turns'] for t in all_timings) / len(all_timings),
        }
        
        # Model response times
        all_model_times = []
        for t in all_timings:
            all_model_times.extend(t['model_response_times'])
        if all_model_times:
            stats['avg_model_response_time'] = sum(all_model_times) / len(all_model_times)
            stats['total_model_calls'] = len(all_model_times)
        
        # Environment step times
        all_step_times = []
        for t in all_timings:
            all_step_times.extend(t['env_step_times'])
        if all_step_times:
            stats['avg_env_step_time'] = sum(all_step_times) / len(all_step_times)
            stats['total_env_steps'] = len(all_step_times)
        
        return stats
    

if __name__ == "__main__":
    from openai import OpenAI
    import os
    import logging
    logging.getLogger("verifiers").setLevel(logging.INFO)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key) 

    # Load Gmail tasks from taskset.json
    taskset_path = "/Users/jaideepchawla/dev/hud/hud-gmail-environment/taskset.json"
    num_tasks = -1
    
    print(f"Loading tasks from {taskset_path}")
    with open(taskset_path, 'r') as f:
        taskset_data = json.load(f)
    
    # Convert the first num_tasks to Task objects
    tasks = []
    for task_dict in taskset_data['tasks'][:num_tasks]:
        # Convert gym config from remote to local
        gym_config = task_dict["gym"].copy()
        gym_config["location"] = "local"
        gym_config["image_or_build_context"] = "gmail"  # Use local gmail build context
        
        # Create a task dict in the format HUD expects
        task_data = {
            "id": task_dict["metadata"]["id"],
            "prompt": task_dict["prompt"],
            "gym": gym_config,  # Use modified gym config
            "setup": task_dict["setup"],
            "evaluate": task_dict["evaluate"],
            "metadata": task_dict["metadata"],
            "description": task_dict["description"]
        }
        
        # Add optional fields if present
        if task_dict.get("system_prompt"):
            task_data["system_prompt"] = task_dict["system_prompt"]
        if task_dict.get("config"):
            task_data["config"] = task_dict["config"]
        if task_dict.get("sensitive_data"):
            task_data["sensitive_data"] = task_dict["sensitive_data"]
            
        tasks.append(Task.from_dict(task_data))
    
    print(f"Loaded {len(tasks)} tasks:")
    for i, task in enumerate(tasks):
        print(f"  {i+1}. {task.id}: {task.prompt[:60]}...")

    # Create HUDGym instance
    hudgym = HUDGym(
        tasks=tasks[:20],
        eval_tasks=tasks[:20],
        max_turns=30,
    )
    
    results = hudgym.evaluate(
        client=client,
        model="gpt-4.1-mini",
        max_concurrent=8,
        num_examples=num_tasks
    )

    # Format and print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Summary statistics
    num_tasks = len(results['task'])
    avg_reward = sum(results['reward']) / num_tasks if num_tasks > 0 else 0.0
    print(f"\nTasks evaluated: {num_tasks}")
    print(f"Average reward: {avg_reward:.3f}")
    
    # Display timing statistics
    print("\nTiming Statistics:")
    print("-"*40)
    timing_stats = hudgym.compute_timing_stats(results)
    
    if timing_stats:
        print(f"Average environment creation time: {timing_stats['avg_env_creation_time']:.3f}s")
        print(f"Average total rollout time: {timing_stats['avg_total_rollout_time']:.3f}s")
        print(f"Average number of turns: {timing_stats['avg_num_turns']:.1f}")
        
        if 'avg_model_response_time' in timing_stats:
            print(f"\nModel Statistics:")
            print(f"  Average response time: {timing_stats['avg_model_response_time']:.3f}s")
            print(f"  Total model calls: {timing_stats['total_model_calls']}")
        
        if 'avg_env_step_time' in timing_stats:
            print(f"\nEnvironment Step Statistics:")
            print(f"  Average step time: {timing_stats['avg_env_step_time']:.3f}s")
            print(f"  Total environment steps: {timing_stats['total_env_steps']}")
    
    print("\n")

    # Make dataset from results
    try:
        dataset = hudgym.make_dataset(
            results, 
            push_to_hub=False,
            state_columns=["timing", "error", "error_step"],
            extra_columns=["hud_reward_func", "format_reward_func"]
        )
        print(f"\nDataset created successfully with {len(dataset)} examples!")
        print(f"Columns: {dataset.column_names}")
        
        # Show first example's timing data
        if len(dataset) > 0 and "timing" in dataset.column_names:
            first_timing = dataset[0]["timing"]
            print(f"\nFirst example timing data:")
            print(f"  Total rollout time: {first_timing['total_rollout_time']:.3f}s")
            print(f"  Number of turns: {first_timing['num_turns']}")
            if first_timing['model_response_times']:
                avg_model_time = sum(first_timing['model_response_times']) / len(first_timing['model_response_times'])
                print(f"  Avg model response time: {avg_model_time:.3f}s")

        # Save dataset to disk
        # dataset_path = Path("hudgym_results_dataset")
        # dataset_path.mkdir(exist_ok=True)
        # dataset.save_to_disk(dataset_path)
        # print(f"\nDataset saved to {dataset_path}")
        
    except Exception as e:
        print(f"\nError creating dataset: {e}")
        import traceback
        traceback.print_exc()