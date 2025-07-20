import verifiers as vf
import json
import os
from hudgym import HUDGym, BasicAdapter
from hud import Task

"""
inference:
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model Qwen/Qwen3-0.6B

training:
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 --config-file verifiers/configs/zero3.yaml hud_env/train.py
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
train_tasks_file = os.path.join(script_dir, "gsm8k_tasks", "gsm8k_train.json")
eval_tasks_file = os.path.join(script_dir, "gsm8k_tasks", "gsm8k_test.json")
with open(train_tasks_file, 'r') as f:
    train_tasks_data = json.load(f)

with open(eval_tasks_file, 'r') as f:
    eval_tasks_data = json.load(f)

# Create all tasks
tasks = [Task.from_dict(task_data) for task_data in train_tasks_data]
eval_tasks = [Task.from_dict(task_data) for task_data in eval_tasks_data]

EVAL_TASKS = 100

# Create HUDGym instance
vf_env = HUDGym(
    tasks=tasks,
    eval_tasks=eval_tasks[:EVAL_TASKS],
    adapter=BasicAdapter(),
)

model_name = "willcb/Qwen3-0.6B"
run_name = "gsm8k-grpo_" + model_name.split("/")[-1].lower()

model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args = vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size = 12
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 2048
training_args.max_seq_len = 2048
training_args.eval_strategy = "steps"
training_args.eval_steps = 10
training_args.save_strategy = "steps"
training_args.save_steps = 100
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 10
training_args.max_concurrent = 96

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)
trainer.train()