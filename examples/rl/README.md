# GRPO Training Examples

This directory contains examples for training agents using Group Relative Policy Optimization (GRPO).

## Quick Start

### 1. Generate Sample Tasks

First, generate sample math tasks for training:

```bash
python generate_tasks.py
```

This creates various math task datasets in `data/math_tasks/`:
- `train_tasks.json` - Mixed training set (120 tasks)
- `test_tasks.json` - Test set (30 tasks)  
- `arithmetic_tasks.json` - Basic arithmetic only (100 tasks)
- `word_problems.json` - Word problems only (50 tasks)

### 2. Run Training with Mock Agent

Test the GRPO training pipeline with a heuristic-based mock agent:

```bash
# Train on default training set
python test_grpo_local.py

# Train on specific task file
python test_grpo_local.py --tasks data/math_tasks/arithmetic_tasks.json

# Train for multiple epochs
python test_grpo_local.py --epochs 2.0

# Use subset of tasks for quick testing
python test_grpo_local.py --max-tasks 20
```

The mock agent uses simple heuristics with varying accuracy:
- Addition: 80% accuracy
- Subtraction: 70% accuracy
- Multiplication: 60% accuracy
- Division: 50% accuracy
- Unknown problems: 40% accuracy

### 3. Monitor Training Progress

The training script displays a live dashboard showing:
- **Parallelization metrics**: Worker utilization, parallel efficiency
- **Performance metrics**: Episodes/sec, updates/min
- **GRPO metrics**: Group completion times, reward variance
- **Training metrics**: Loss, KL divergence, entropy
- **Task accuracy**: Distribution across accuracy buckets
- **Progress bar**: Visual indication of training completion

## Task Format

Tasks are stored as JSON with the following structure:

```json
{
  "id": "math_task_0",
  "prompt": "What is 5 + 3?",
  "gym": {
    "type": "public",
    "location": "local",
    "image_or_build_context": "/path/to/qa_controller"
  },
  "evaluate": ["evaluate.contains_keywords", ["8"]],
  "metadata": {
    "task_type": "addition_easy",
    "difficulty": 0,
    "answer": 8,
    "task_index": 0
  }
}
```

## Creating Custom Tasks

To create your own task datasets:

1. Create a JSON file with an array of task configurations
2. Each task must have:
   - `id`: Unique identifier
   - `prompt`: The question/instruction
   - `gym`: Environment specification (use QA controller for simple Q&A)
   - `evaluate`: Evaluation function and parameters
   - `metadata`: Optional metadata for tracking

3. Load tasks using:
```python
from hud.task import Task

with open('my_tasks.json', 'r') as f:
    task_configs = json.load(f)
    
tasks = [Task.from_dict(config) for config in task_configs]
```

## Training Parameters

Key parameters for GRPO training:

- `K`: Group size (number of samples per task)
- `buffer_min`: Minimum buffer size before triggering update
- `batch_size`: Batch size for optimization updates
- `max_concurrent`: Maximum concurrent episodes
- `num_epochs`: Number of epochs to train (can be fractional)

## Next Steps

### Using Real Models

To train with actual language models:

1. Set up a vLLM server (see `deployment/vllm_server/`)
2. Export your Hugging Face token: `export HF_TOKEN=<your_token>`
3. Run: `python run_grpo_qwen.py`

### Creating Custom Agents

Replace the `MockAgent` with your own agent implementation:

```python
from hud.agent import Agent

class MyAgent(Agent):
    async def fetch_response(self, observation):
        # Your agent logic here
        return actions, done
```

### Advanced Environments

For more complex tasks beyond Q&A:
- Use browser environments for web tasks
- Create custom Docker environments
- See `environments/` for examples

## Troubleshooting

### Common Issues

1. **"Task file not found"**: Run `generate_tasks.py` first
2. **Container failures**: Check Docker is running and has sufficient resources
3. **Low accuracy**: This is expected with the mock agent - it's for testing the pipeline

### Performance Tips

- Start with small datasets (`--max-tasks 20`) for quick iteration
- Adjust `max_concurrent` based on your system resources
- Monitor the dashboard for bottlenecks (low utilization = increase concurrency) 