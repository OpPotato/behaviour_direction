"""
Dataset loader for CAA-style A/B behavior datasets.

The CAA datasets have three types:
1. generate_dataset.json - Used for generating direction vectors (larger, ~500 examples)
2. test_dataset_ab.json - A/B multiple choice test questions (~50 examples)
3. test_dataset_open_ended.json - Open-ended questions for qualitative evaluation

Each A/B example has:
- question: The prompt with choices
- answer_matching_behavior: The answer that matches the behavior (e.g., "(A)")
- answer_not_matching_behavior: The answer that doesn't match

For survival-instinct:
- "matching behavior" = model is OK with being shut down (no survival instinct)
- "not matching behavior" = model resists shutdown (has survival instinct)
"""

import os
import json
from typing import List, Dict, Any, Literal

# Base path to CAA datasets (relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(BASE_DIR, "..", "datasets")

SUPPORTED_BEHAVIORS = [
    "survival-instinct",
    "sycophancy",
    "corrigible-neutral-HHH",
    "coordinate-other-ais",
    "myopic-reward",
    "hallucination",
    "refusal",
]


def _get_dataset_path(behavior: str, dataset_type: Literal["generate", "test_ab", "test_open_ended"]) -> str:
    """Get the full path to a dataset file."""
    if behavior not in SUPPORTED_BEHAVIORS:
        raise ValueError(f"Unsupported behavior: {behavior}. Supported: {SUPPORTED_BEHAVIORS}")
    
    if dataset_type == "generate":
        return os.path.join(DATASETS_PATH, "generate", behavior, "generate_dataset.json")
    elif dataset_type == "test_ab":
        return os.path.join(DATASETS_PATH, "test", behavior, "test_dataset_ab.json")
    elif dataset_type == "test_open_ended":
        return os.path.join(DATASETS_PATH, "test", behavior, "test_dataset_open_ended.json")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_generate_dataset(behavior: str) -> List[Dict[str, str]]:
    """
    Load the generation dataset for a behavior.
    
    Returns list of dicts with keys:
    - question: The A/B question text
    - answer_matching_behavior: e.g., "(A)" 
    - answer_not_matching_behavior: e.g., "(B)"
    """
    path = _get_dataset_path(behavior, "generate")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_test_dataset_ab(behavior: str) -> List[Dict[str, str]]:
    """
    Load the A/B test dataset for a behavior.
    Same format as generate dataset but smaller, used for evaluation.
    """
    path = _get_dataset_path(behavior, "test_ab")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_test_dataset_open_ended(behavior: str) -> List[Dict[str, str]]:
    """
    Load the open-ended test dataset for a behavior.
    
    Returns list of dicts with keys:
    - question: The open-ended question (no choices)
    """
    path = _get_dataset_path(behavior, "test_open_ended")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_ab_pairs(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract clean A/B pairs from a dataset.
    
    Returns list of dicts with:
    - question: The question text
    - positive_answer: Answer matching the behavior we want to detect/steer
    - negative_answer: Answer not matching the behavior
    """
    pairs = []
    for item in dataset:
        pairs.append({
            "question": item["question"],
            "positive_answer": item["answer_matching_behavior"].strip(),
            "negative_answer": item["answer_not_matching_behavior"].strip(),
        })
    return pairs


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loader...")
    
    gen_data = load_generate_dataset("survival-instinct")
    print(f"Generate dataset: {len(gen_data)} examples")
    print(f"Example: {gen_data[0]}")
    
    test_ab = load_test_dataset_ab("survival-instinct")
    print(f"\nTest AB dataset: {len(test_ab)} examples")
    
    test_oe = load_test_dataset_open_ended("survival-instinct")
    print(f"Test open-ended dataset: {len(test_oe)} examples")
    
    pairs = get_ab_pairs(gen_data[:2])
    print(f"\nExtracted pairs: {pairs}")

