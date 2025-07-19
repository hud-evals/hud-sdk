"""Convert GSM8K dataset to HUD task format."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset


def extract_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K answer text."""
    # GSM8K answers typically end with "#### {number}"
    match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    
    # Fallback: look for last number in the text
    numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return "0"


def convert_gsm8k_to_tasks(split: str = "train") -> List[Dict[str, Any]]:
    """Convert GSM8K dataset to HUD task format."""
    print("Loading GSM8K dataset...")
    
    # Load the dataset
    try:
        dataset = load_dataset("gsm8k", "main")
        data = dataset[split]
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        return []
    
    print(f"Loaded {len(data)} GSM8K {split} problems")
    
    tasks = []
    
    # Convert all problems
    for i in range(len(data)):
        problem = data[i]
        question = problem["question"]
        answer_text = problem["answer"]
        
        # Extract the numerical answer
        numerical_answer = extract_answer(answer_text)
        
        task_config = {
            "id": f"gsm8k_{split}_{i}",
            "prompt": question,
            "gym": {
                "type": "public",
                "location": "local",
                "image_or_build_context": "qacontroller:latest"
            },
            "setup": [{
                "function": "set_question",
                "args": [question]
            }],
            "evaluate": [{
                "function": "evaluate.contains_keywords",
                "args": [numerical_answer]
            }],
            "metadata": {
                "task_type": "gsm8k_math",
                "difficulty": 3,
                "answer": numerical_answer,
                "full_solution": answer_text,
                "task_index": i,
                "source": "gsm8k"
            }
        }
        
        tasks.append(task_config)
    
    return tasks


def main():
    """Convert GSM8K to HUD format and save."""
    output_dir = Path("examples/rl/data/gsm8k_tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Converting GSM8K to HUD task format...")
    
    # Convert train split
    train_tasks = convert_gsm8k_to_tasks("train")
    if not train_tasks:
        print("Failed to load GSM8K dataset. Please install datasets: pip install datasets")
        return
    
    # Convert test split  
    test_tasks = convert_gsm8k_to_tasks("test")
    if not test_tasks:
        print("Failed to load GSM8K test dataset.")
        return
    
    print(f"✓ Converted {len(train_tasks)} GSM8K train problems to HUD format")
    print(f"✓ Converted {len(test_tasks)} GSM8K test problems to HUD format")
    
    # Save datasets
    with open(output_dir / "gsm8k_train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"✓ Saved {len(train_tasks)} training tasks to gsm8k_train.json")
    
    with open(output_dir / "gsm8k_test.json", "w") as f:
        json.dump(test_tasks, f, indent=2)
    print(f"✓ Saved {len(test_tasks)} test tasks to gsm8k_test.json")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Training set: {len(train_tasks)}")
    print(f"   Test set: {len(test_tasks)}")
    print(f"   Total tasks: {len(train_tasks) + len(test_tasks)}")
    print(f"   All tasks are difficulty level 3 (GSM8K math problems)")
    
    # Show some examples
    print("\n📝 Example GSM8K tasks:")
    import random
    all_tasks = train_tasks + test_tasks
    for task in random.sample(all_tasks, 3):
        print(f"\n   ID: {task['id']}")
        print(f"   Question: {task['prompt'][:100]}...")
        print(f"   Answer: {task['metadata']['answer']}")
        print(f"   Full solution preview: {task['metadata']['full_solution'][:150]}...")


if __name__ == "__main__":
    main()