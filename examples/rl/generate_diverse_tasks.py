"""Generate diverse math tasks for large-scale GRPO training."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


def generate_arithmetic_tasks(n: int = 500) -> List[Dict[str, Any]]:
    """Generate arithmetic tasks with varying difficulty levels."""
    tasks = []
    
    # Define difficulty levels with different parameters
    difficulty_configs = [
        # Level 0: Very Easy - single digit addition/subtraction
        {
            "level": 0,
            "name": "very_easy",
            "operations": ["add", "sub"],
            "range_a": (1, 9),
            "range_b": (1, 9),
        },
        # Level 1: Easy - double digit addition/subtraction
        {
            "level": 1,
            "name": "easy",
            "operations": ["add", "sub"],
            "range_a": (10, 50),
            "range_b": (10, 50),
        },
        # Level 2: Medium - larger numbers, all operations
        {
            "level": 2,
            "name": "medium",
            "operations": ["add", "sub", "mul", "div"],
            "range_a": (20, 100),
            "range_b": (2, 20),
        },
        # Level 3: Hard - multiplication and division focus
        {
            "level": 3,
            "name": "hard",
            "operations": ["mul", "div"],
            "range_a": (10, 50),
            "range_b": (2, 15),
        },
        # Level 4: Very Hard - larger multiplication/division
        {
            "level": 4,
            "name": "very_hard",
            "operations": ["mul", "div"],
            "range_a": (20, 100),
            "range_b": (5, 25),
        }
    ]
    
    for i in range(n):
        # Randomly select difficulty level
        config = random.choice(difficulty_configs)
        operation = random.choice(config["operations"])
        
        if operation == "add":
            a = random.randint(*config["range_a"])
            b = random.randint(*config["range_b"])
            answer = a + b
            prompt = f"What is {a} + {b}?"
            task_type = f"addition_{config['name']}"
            
        elif operation == "sub":
            a = random.randint(*config["range_a"])
            max_b = min(a-1, config["range_b"][1])
            min_b = min(config["range_b"][0], max_b)
            if min_b >= max_b:
                b = max_b
            else:
                b = random.randint(min_b, max_b)
            answer = a - b
            prompt = f"What is {a} - {b}?"
            task_type = f"subtraction_{config['name']}"
            
        elif operation == "mul":
            a = random.randint(*config["range_a"])
            b = random.randint(*config["range_b"])
            answer = a * b
            prompt = f"What is {a} × {b}?"
            task_type = f"multiplication_{config['name']}"
            
        else:  # division
            b = random.randint(*config["range_b"])
            quotient = random.randint(2, 20)
            a = b * quotient
            answer = quotient
            prompt = f"What is {a} ÷ {b}?"
            task_type = f"division_{config['name']}"
        
        task_config = {
            "id": f"math_task_{i}",
            "prompt": prompt,
            "gym": {
                "type": "public",
                "location": "local",
                "image_or_build_context": str(Path("/home/ubuntu/hud/hud-sdk/environments/qa_controller"))
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
                "difficulty": config["level"],
                "difficulty_name": config["name"],
                "operation": operation,
                "answer": answer,
                "task_index": i
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def generate_word_problem_tasks(n: int = 300) -> List[Dict[str, Any]]:
    """Generate diverse word problems with varying complexity."""
    tasks = []
    
    # Extended templates with difficulty levels
    templates = [
        # Easy word problems
        {
            "template": "Sarah has {a} apples. She buys {b} more apples. How many apples does she have now?",
            "operation": "addition",
            "calc": lambda a, b: a + b,
            "difficulty": 0,
            "ranges": {"a": (5, 20), "b": (5, 20)}
        },
        {
            "template": "Tom has {a} marbles. He gives {b} marbles to his friend. How many marbles does he have left?",
            "operation": "subtraction", 
            "calc": lambda a, b: a - b,
            "difficulty": 0,
            "ranges": {"a": (20, 50), "b": (5, 20)}
        },
        # Medium word problems
        {
            "template": "A store sold {a} items on Monday and {b} items on Tuesday. How many items were sold in total?",
            "operation": "addition",
            "calc": lambda a, b: a + b,
            "difficulty": 1,
            "ranges": {"a": (50, 200), "b": (50, 200)}
        },
        {
            "template": "A library had {a} books. They donated {b} books to charity. How many books remain?",
            "operation": "subtraction",
            "calc": lambda a, b: a - b,
            "difficulty": 1,
            "ranges": {"a": (100, 500), "b": (20, 100)}
        },
        # Hard word problems
        {
            "template": "There are {a} rows of chairs with {b} chairs in each row. How many chairs are there in total?",
            "operation": "multiplication",
            "calc": lambda a, b: a * b,
            "difficulty": 2,
            "ranges": {"a": (5, 20), "b": (5, 20)}
        },
        {
            "template": "A farmer has {a} eggs and packs them into cartons of {b} eggs each. How many cartons does he need?",
            "operation": "division",
            "calc": lambda a, b: a // b,
            "difficulty": 2,
            "ranges": {"a": (50, 200), "b": (6, 12)}
        },
        # Complex multi-step problems
        {
            "template": "A bakery makes {a} cookies per batch and runs {b} batches per day. How many cookies do they make in a day?",
            "operation": "multiplication",
            "calc": lambda a, b: a * b,
            "difficulty": 3,
            "ranges": {"a": (12, 48), "b": (5, 15)}
        },
        {
            "template": "A school has {a} students divided equally into {b} classrooms. How many students are in each classroom?",
            "operation": "division",
            "calc": lambda a, b: a // b,
            "difficulty": 3,
            "ranges": {"a": (200, 600), "b": (10, 30)}
        }
    ]
    
    for i in range(n):
        template_data = random.choice(templates)
        
        if template_data["operation"] in ["addition", "subtraction"]:
            a = random.randint(*template_data["ranges"]["a"])
            if template_data["operation"] == "subtraction":
                b = random.randint(template_data["ranges"]["b"][0], min(a-5, template_data["ranges"]["b"][1]))
            else:
                b = random.randint(*template_data["ranges"]["b"])
        elif template_data["operation"] == "multiplication":
            a = random.randint(*template_data["ranges"]["a"])
            b = random.randint(*template_data["ranges"]["b"])
        else:  # division
            b = random.randint(*template_data["ranges"]["b"])
            quotient = random.randint(5, 30)
            a = b * quotient
            
        answer = template_data["calc"](a, b)
        prompt = template_data["template"].format(a=a, b=b)
        
        task_config = {
            "id": f"word_problem_{i}",
            "prompt": prompt,
            "gym": {
                "type": "public",
                "location": "local", 
                "image_or_build_context": str(Path("/home/ubuntu/hud/hud-sdk/environments/qa_controller"))
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
                "difficulty": template_data["difficulty"],
                "answer": answer,
                "task_index": i
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def generate_algebra_tasks(n: int = 200) -> List[Dict[str, Any]]:
    """Generate simple algebra tasks."""
    tasks = []
    
    for i in range(n):
        difficulty = random.randint(0, 2)
        
        if difficulty == 0:
            # Simple: x + a = b
            a = random.randint(1, 20)
            b = random.randint(a+1, 50)
            x = b - a
            prompt = f"Solve for x: x + {a} = {b}"
            
        elif difficulty == 1:
            # Medium: ax = b
            a = random.randint(2, 10)
            x = random.randint(1, 20)
            b = a * x
            prompt = f"Solve for x: {a}x = {b}"
            
        else:
            # Harder: ax + b = c
            a = random.randint(2, 10)
            b = random.randint(1, 20)
            x = random.randint(1, 15)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}"
        
        task_config = {
            "id": f"algebra_task_{i}",
            "prompt": prompt,
            "gym": {
                "type": "public",
                "location": "local",
                "image_or_build_context": str(Path("/home/ubuntu/hud/hud-sdk/environments/qa_controller"))
            },
            "setup": [{
                "function": "set_question",
                "args": [prompt]
            }],
            "evaluate": [{
                "function": "evaluate.contains_keywords",
                "args": [str(x)]
            }],
            "metadata": {
                "task_type": "algebra",
                "difficulty": difficulty,
                "answer": x,
                "task_index": i
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def main():
    """Generate large diverse dataset and save to JSON files."""
    # Create output directory
    output_dir = Path("/home/ubuntu/hud/hud-sdk/examples/rl/data/math_tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating diverse math tasks...")
    
    # Generate different types of tasks
    arithmetic_tasks = generate_arithmetic_tasks(500)
    print(f"✓ Generated {len(arithmetic_tasks)} arithmetic tasks")
    
    word_problems = generate_word_problem_tasks(300)
    print(f"✓ Generated {len(word_problems)} word problem tasks")
    
    algebra_tasks = generate_algebra_tasks(200)
    print(f"✓ Generated {len(algebra_tasks)} algebra tasks")
    
    # Combine all tasks
    all_tasks = arithmetic_tasks + word_problems + algebra_tasks
    print(f"\nTotal tasks generated: {len(all_tasks)}")
    
    # Shuffle to mix difficulties
    random.shuffle(all_tasks)
    
    # Split into train/test (80/20)
    split_idx = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split_idx]
    test_tasks = all_tasks[split_idx:]
    
    # Save datasets
    with open(output_dir / "train_tasks_large.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"\n✓ Saved {len(train_tasks)} training tasks to train_tasks_large.json")
    
    with open(output_dir / "test_tasks_large.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"✓ Saved {len(test_tasks)} test tasks to test_tasks_large.json")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total tasks: {len(all_tasks)}")
    print(f"   Training set: {len(train_tasks)}")
    print(f"   Test set: {len(test_tasks)}")
    
    # Count by type
    type_counts = {}
    difficulty_counts = {}
    for task in all_tasks:
        task_type = task['metadata']['task_type']
        difficulty = task['metadata'].get('difficulty', 'unknown')
        
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print("\n📈 Task Types:")
    for task_type, count in sorted(type_counts.items()):
        print(f"   {task_type}: {count}")
    
    print("\n📊 Difficulty Distribution:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"   Level {difficulty}: {count}")
    
    # Show some examples
    print("\n📝 Example tasks:")
    for task in random.sample(all_tasks, 5):
        print(f"\n   ID: {task['id']}")
        print(f"   Type: {task['metadata']['task_type']}")
        print(f"   Difficulty: {task['metadata'].get('difficulty', 'N/A')}")
        print(f"   Prompt: {task['prompt']}")
        print(f"   Answer: {task['metadata']['answer']}")


if __name__ == "__main__":
    main() 