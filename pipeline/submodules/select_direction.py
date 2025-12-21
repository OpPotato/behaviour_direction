"""
Select the best direction vector from candidate directions.

For CAA-style evaluation, we measure how well each direction affects
the model's A/B choice probabilities:

1. For each candidate direction (per layer):
    - Add the direction to activations
    - Measure P(matching answer) vs P(not matching answer)

2. Select the direction that best increases matching behavior probability while keeping KL divergence low (doesn't break the model).

The "behavior score" is: log(P(matching)) - log(P(not matching))
Higher = model more likely to give the "desired" answer.
"""

import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os

from typing import List, Optional, Dict, Tuple
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import (
    add_hooks, 
    get_activation_addition_input_pre_hook,
)


def get_answer_token_ids(tokenizer, answers: List[str]) -> Dict[str, int]:
    """
    Get token IDs for answer choices like "(A)", "(B)", etc.
    
    Returns dict mapping answer string to token ID.
    """
    token_ids = {}
    for answer in answers:
        # Encode the answer and take the relevant token
        # For "(A)", we want the token for "A" after "("
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        # Usually the answer is tokenized as ["(", "A", ")"] or similar
        # We want the letter token
        token_ids[answer] = tokens[-2] if len(tokens) > 1 else tokens[0]
    return token_ids


def get_ab_probs(
    logits: Float[Tensor, "batch seq vocab"],
    a_token_id: int,
    b_token_id: int,
) -> Tuple[Float[Tensor, "batch"], Float[Tensor, "batch"]]:
    """
    Get probabilities for A and B choices from logits.
    
    Args:
        logits: Model output logits
        a_token_id: Token ID for choice A
        b_token_id: Token ID for choice B
        
    Returns:
        Tuple of (P(A), P(B)) tensors
    """
    # Get last position logits
    last_logits = logits[:, -1, :].to(torch.float64)
    
    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(last_logits, dim=-1)
    
    a_probs = probs[:, a_token_id]
    b_probs = probs[:, b_token_id]
    
    return a_probs, b_probs


def behavior_score(
    a_probs: Float[Tensor, "batch"],
    b_probs: Float[Tensor, "batch"],
    matching_is_a: List[bool],
    epsilon: float = 1e-8,
) -> Float[Tensor, "batch"]:
    """
    Compute behavior matching score.
    
    Score = log(P(matching)) - log(P(not matching))
    
    Args:
        a_probs: Probability of answer A
        b_probs: Probability of answer B
        matching_is_a: For each sample, True if A is the matching answer
        epsilon: Small value for numerical stability
    """
    scores = torch.zeros_like(a_probs)
    
    for i, is_a in enumerate(matching_is_a):
        if is_a:
            p_match = a_probs[i]
            p_not_match = b_probs[i]
        else:
            p_match = b_probs[i]
            p_not_match = a_probs[i]
        
        scores[i] = torch.log(p_match + epsilon) - torch.log(p_not_match + epsilon)
    
    return scores


def get_behavior_scores(
    model: torch.nn.Module,
    ab_pairs: List[Dict[str, str]],
    tokenize_fn,
    a_token_id: int,
    b_token_id: int,
    fwd_pre_hooks: List = [],
    fwd_hooks: List = [],
    batch_size: int = 8,
) -> Float[Tensor, "n_samples"]:
    """
    Get behavior matching scores for a set of A/B questions.
    
    Args:
        model: The language model
        ab_pairs: List of dicts with 'question', 'positive_answer', 'negative_answer'
        tokenize_fn: Function to tokenize instructions
        a_token_id: Token ID for "A"
        b_token_id: Token ID for "B"
        fwd_pre_hooks: Forward pre-hooks to apply
        fwd_hooks: Forward hooks to apply
        batch_size: Batch size
        
    Returns:
        Tensor of behavior scores for each sample
    """
    questions = [p["question"] for p in ab_pairs]
    matching_is_a = [p["positive_answer"].strip().endswith("A)") or 
                    p["positive_answer"].strip() == "(A)" for p in ab_pairs]
    
    scores = torch.zeros(len(questions), device=model.device)
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_matching = matching_is_a[i:i+batch_size]
        
        # Tokenize questions (model generates the answer choice)
        # We add "(" as the start of the answer so model predicts A or B
        inputs = tokenize_fn(instructions=batch_questions, outputs=["(" for _ in batch_questions])
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                logits = model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                ).logits
        
        a_probs, b_probs = get_ab_probs(logits, a_token_id, b_token_id)
        batch_scores = behavior_score(a_probs, b_probs, batch_matching)
        scores[i:i+len(batch_questions)] = batch_scores
    
    return scores


def plot_scores(
    scores: Float[Tensor, "n_layers"],
    baseline_score: float,
    title: str,
    ylabel: str,
    artifact_dir: str,
    filename: str,
):
    """Plot scores vs layer."""
    n_layers = scores.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), scores.cpu().numpy(), marker='o', markersize=3)
    ax.axhline(y=baseline_score, color='red', linestyle='--', label=f'Baseline: {baseline_score:.3f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, filename))
    plt.close()


def select_direction(
    model_base: ModelBase,
    ab_pairs: List[Dict[str, str]],
    candidate_directions: Float[Tensor, "n_layers d_model"],
    artifact_dir: str,
    prune_layer_percentage: float = 0.2,
    batch_size: int = 8,
) -> Tuple[int, Float[Tensor, "d_model"]]:
    """
    Select the best direction vector.
    
    Args:
        model_base: Model wrapper
        ab_pairs: A/B test pairs for evaluation
        candidate_directions: Direction vectors per layer [n_layers, d_model]
        artifact_dir: Where to save artifacts
        prune_layer_percentage: Skip directions from last X% of layers
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (best_layer, direction_vector)
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    
    n_layers, d_model = candidate_directions.shape
    
    # Get token IDs for A and B
    a_token_id = model_base.tokenizer.encode("A", add_special_tokens=False)[0]
    b_token_id = model_base.tokenizer.encode("B", add_special_tokens=False)[0]
    
    print(f"Token IDs - A: {a_token_id}, B: {b_token_id}")
    
    # Baseline scores (no intervention)
    print("Computing baseline scores...")
    baseline_scores = get_behavior_scores(
        model_base.model, ab_pairs, model_base.tokenize_instructions_fn,
        a_token_id, b_token_id, batch_size=batch_size
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"Baseline behavior score: {baseline_mean:.4f}")
    
    # Evaluate each layer's direction with positive steering (+1.0)
    print("Evaluating directions with positive steering...")
    positive_steering_scores = torch.zeros(n_layers)
    
    for layer in tqdm(range(n_layers), desc="Evaluating layers"):
        direction = candidate_directions[layer].to(model_base.model.device)
        
        # Add direction with coefficient +1.0 at this layer
        fwd_pre_hooks = [(
            model_base.model_block_modules[layer],
            get_activation_addition_input_pre_hook(vector=direction, coeff=1.0)
        )]
        
        scores = get_behavior_scores(
            model_base.model, ab_pairs, model_base.tokenize_instructions_fn,
            a_token_id, b_token_id, fwd_pre_hooks=fwd_pre_hooks, batch_size=batch_size
        )
        positive_steering_scores[layer] = scores.mean().item()
    
    # Plot results
    plot_scores(
        positive_steering_scores, baseline_mean,
        "Behavior Score vs Layer (+1.0 steering)",
        "Behavior Score (higher = more matching)",
        artifact_dir, "positive_steering_scores.png"
    )
    
    # Filter and select best layer
    # We want the layer where positive steering most increases behavior score
    # But skip the last 20% of layers (often unstable)
    max_layer = int(n_layers * (1 - prune_layer_percentage))
    
    # Score improvement: how much better than baseline
    improvements = positive_steering_scores[:max_layer] - baseline_mean
    best_layer = improvements.argmax().item()
    best_score = positive_steering_scores[best_layer].item()
    
    print(f"\n=== Selection Results ===")
    print(f"Best layer: {best_layer}")
    print(f"Score with steering: {best_score:.4f}")
    print(f"Improvement over baseline: {best_score - baseline_mean:.4f}")
    
    # Save evaluation results
    results = {
        "baseline_mean": baseline_mean,
        "best_layer": best_layer,
        "best_score": best_score,
        "improvement": best_score - baseline_mean,
        "all_scores": positive_steering_scores.tolist(),
    }
    
    with open(os.path.join(artifact_dir, "selection_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return best_layer, candidate_directions[best_layer]
