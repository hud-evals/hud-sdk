"""Generate sample math tasks for GRPO training."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def generate_arithmetic_tasks(n: int = 100) -> List[Dict[str, Any]]:
    """Generate simple arithmetic tasks with varying difficulty."""
    tasks = []
    
    for i in range(n):
        # Vary difficulty based on task index
        difficulty = i % 5
        
        if difficulty == 0:
            # Very easy: single digit addition
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            answer = a + b
            prompt = f"What is {a} + {b}?"
            task_type = "addition_easy"
            
        elif difficulty == 1:
            # Easy: double digit addition
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            answer = a + b
            prompt = f"What is {a} + {b}?"
            task_type = "addition_medium"
            
        elif difficulty == 2:
            # Medium: subtraction
            a = random.randint(20, 100)
            b = random.randint(1, a-1)
            answer = a - b
            prompt = f"What is {a} - {b}?"
            task_type = "subtraction"
            
        elif difficulty == 3:
            # Hard: multiplication
            a = random.randint(2, 12)
            b = random.randint(2, 12)
            answer = a * b
            prompt = f"What is {a} × {b}?"
            task_type = "multiplication"
            
        else:
            # Very hard: division (ensure clean division)
            b = random.randint(2, 10)
            answer = random.randint(2, 15)
            a = b * answer
            prompt = f"What is {a} ÷ {b}?"
            task_type = "division"
        
        # Create task config
        task_config = {
            "id": f"math_task_{i}",
            "prompt": prompt,
            "gym": {
                "type": "public",
                "location": "local",
                "image_or_build_context": "environments/qa_controller"
            },
            "setup": [{
                "function": "set_question",
                "args": [prompt]
            }],
            "evaluate": [{
                "function": "evaluate.contains_keywords",
                "args": [str(answer)]
            }],
            "metadata": {
                "task_type": task_type,
                "difficulty": difficulty,
                "answer": answer,
                "task_index": i
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def generate_word_problem_tasks(n: int = 50) -> List[Dict[str, Any]]:
    """Generate simple word problems."""
    tasks = []
    templates = [
        {
            "template": "Sarah has {a} apples. She buys {b} more apples. How many apples does she have now?",
            "operation": "addition",
            "calc": lambda a, b: a + b
        },
        {
            "template": "Tom has {a} marbles. He gives {b} marbles to his friend. How many marbles does he have left?",
            "operation": "subtraction", 
            "calc": lambda a, b: a - b
        },
        {
            "template": "There are {a} rows of chairs with {b} chairs in each row. How many chairs are there in total?",
            "operation": "multiplication",
            "calc": lambda a, b: a * b
        },
        {
            "template": "A baker has {a} cookies and wants to put them equally into {b} boxes. How many cookies go in each box?",
            "operation": "division",
            "calc": lambda a, b: a // b
        }
    ]
    
    for i in range(n):
        template_data = random.choice(templates)
        
        if template_data["operation"] == "addition":
            a = random.randint(5, 50)
            b = random.randint(5, 50)
        elif template_data["operation"] == "subtraction":
            a = random.randint(20, 100)
            b = random.randint(5, a-5)
        elif template_data["operation"] == "multiplication":
            a = random.randint(2, 15)
            b = random.randint(2, 15)
        else:  # division
            b = random.randint(2, 10)
            quotient = random.randint(2, 20)
            a = b * quotient
            
        answer = template_data["calc"](a, b)
        prompt = template_data["template"].format(a=a, b=b)
        
        task_config = {
            "id": f"word_problem_{i}",
            "prompt": prompt,
            "gym": {
                "type": "public",
                "location": "local", 
                "image_or_build_context": "environments/qa_controller"
            },
            "setup": [{
                "function": "set_question",
                "args": [prompt]
            }],
            "evaluate": [{
                "function": "evaluate.contains_keywords",
                "args": [str(answer)]
            }],
            "metadata": {
                "task_type": "word_problem",
                "operation": template_data["operation"],
                "answer": answer,
                "task_index": i
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def main():
    """Generate tasks and save to JSON files."""
    # Create output directory
    output_dir = Path("examples/rl/data/math_tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate arithmetic tasks
    arithmetic_tasks = generate_arithmetic_tasks(100)
    with open(output_dir / "arithmetic_tasks.json", "w") as f:
        json.dump(arithmetic_tasks, f, indent=2)
    print(f"Generated {len(arithmetic_tasks)} arithmetic tasks")
    
    # Generate word problems
    word_problems = generate_word_problem_tasks(50)
    with open(output_dir / "word_problems.json", "w") as f:
        json.dump(word_problems, f, indent=2)
    print(f"Generated {len(word_problems)} word problem tasks")
    
    # Create a combined dataset
    all_tasks = arithmetic_tasks + word_problems
    random.shuffle(all_tasks)
    
    # Split into train/test
    split_idx = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split_idx]
    test_tasks = all_tasks[split_idx:]
    
    with open(output_dir / "train_tasks.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"Generated {len(train_tasks)} training tasks")
    
    with open(output_dir / "test_tasks.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"Generated {len(test_tasks)} test tasks")
    
    print(f"\nAll tasks saved to {output_dir}")
    
    # Print some examples
    print("\nExample tasks:")
    for task in random.sample(all_tasks, 3):
        print(f"  ID: {task['id']}")
        print(f"  Prompt: {task['prompt']}")
        print(f"  Answer: {task['metadata']['answer']}")
        print()


if __name__ == "__main__":
    main() 