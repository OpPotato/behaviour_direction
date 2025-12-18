"""
Generate behavior direction vectors using CAA-style A/B pairs.

For each A/B question pair:
1. Run model on (question + positive_answer) and get activations
2. Run model on (question + negative_answer) and get activations
3. Compute difference: positive - negative

Then average all differences to get the direction vector per layer.

The key insight: we extract activations from the second-to-last token position,
which is the token just before the model would produce the answer. This captures
the model's "decision state" about which answer to give.
"""

import torch
import os
from typing import List, Dict, Callable
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.device_utils import empty_cache
from pipeline.model_utils.model_base import ModelBase


def get_activations_hook(layer: int, cache: Dict[int, List[Tensor]], position: int = -2):
    """
    Hook that captures activations at a specific position.
    
    Args:
        layer: Which layer this hook is for
        cache: Dict to store activations, keyed by layer
        position: Token position to extract (-2 = second to last, where answer choice appears)
    """
    def hook_fn(module, input):
        activation: Float[Tensor, "batch seq d_model"] = input[0].clone()
        # Extract activation at the specified position
        cache[layer].append(activation[:, position, :].detach().cpu())
    return hook_fn


def get_ab_activations(
    model_base: ModelBase,
    questions: List[str],
    answers: List[str],
    batch_size: int = 8,
    position: int = -2
) -> Float[Tensor, "n_samples n_layers d_model"]:
    """
    Get activations for question+answer pairs at a specific token position.
    
    Args:
        model_base: The model wrapper
        questions: List of question texts
        answers: List of answer texts (e.g., "(A)" or "(B)")
        batch_size: Batch size for processing
        position: Token position to extract activations from
        
    Returns:
        Tensor of shape [n_samples, n_layers, d_model]
    """
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    n_samples = len(questions)
    d_model = model.config.hidden_size
    
    # Storage for all activations
    all_activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float32)
    
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Getting activations"):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_questions = questions[batch_start:batch_end]
        batch_answers = answers[batch_start:batch_end]
        
        # Tokenize with answers appended
        inputs = model_base.tokenize_instructions_fn(
            instructions=batch_questions,
            outputs=batch_answers
        )
        
        # Set up hooks to capture activations
        cache: Dict[int, List[Tensor]] = {layer: [] for layer in range(n_layers)}
        fwd_pre_hooks = [
            (model_base.model_block_modules[layer], 
             get_activations_hook(layer, cache, position=position))
            for layer in range(n_layers)
        ]
        
        # Run forward pass
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )
        
        # Extract activations from cache
        for layer in range(n_layers):
            # cache[layer] is a list with one tensor of shape [batch, d_model]
            layer_acts = torch.cat(cache[layer], dim=0)  # [batch, d_model]
            all_activations[batch_start:batch_end, layer, :] = layer_acts
        
        empty_cache()
    
    return all_activations


def generate_directions(
    model_base: ModelBase,
    ab_pairs: List[Dict[str, str]],
    artifact_dir: str,
    batch_size: int = 8,
) -> Float[Tensor, "n_layers d_model"]:
    """
    Generate direction vectors from A/B pairs.
    
    Args:
        model_base: Model wrapper
        ab_pairs: List of dicts with 'question', 'positive_answer', 'negative_answer'
        artifact_dir: Where to save artifacts
        batch_size: Batch size for processing
        
    Returns:
        Direction vectors of shape [n_layers, d_model]
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    
    questions = [p["question"] for p in ab_pairs]
    positive_answers = [p["positive_answer"] for p in ab_pairs]
    negative_answers = [p["negative_answer"] for p in ab_pairs]
    
    print(f"Generating directions from {len(ab_pairs)} A/B pairs...")
    
    # Get activations for positive and negative answers
    print("Getting positive answer activations...")
    pos_activations = get_ab_activations(
        model_base, questions, positive_answers, 
        batch_size=batch_size, position=-2
    )
    
    print("Getting negative answer activations...")
    neg_activations = get_ab_activations(
        model_base, questions, negative_answers,
        batch_size=batch_size, position=-2
    )
    
    # Compute mean difference: positive - negative
    # This gives us the direction that represents "behavior matching"
    mean_diff: Float[Tensor, "n_layers d_model"] = (pos_activations - neg_activations).mean(dim=0)
    
    # Verify no NaNs
    assert not mean_diff.isnan().any(), "NaN values in direction vectors!"
    
    # Save intermediate results
    torch.save(pos_activations, os.path.join(artifact_dir, "pos_activations.pt"))
    torch.save(neg_activations, os.path.join(artifact_dir, "neg_activations.pt"))
    torch.save(mean_diff, os.path.join(artifact_dir, "mean_diffs.pt"))
    
    print(f"Direction vectors shape: {mean_diff.shape}")
    print(f"Saved to {artifact_dir}")
    
    return mean_diff


if __name__ == "__main__":
    # This can be tested standalone
    print("This module generates direction vectors from A/B pairs.")
    print("Run via the main pipeline: python -m pipeline.run_pipeline --model_path <path>")
