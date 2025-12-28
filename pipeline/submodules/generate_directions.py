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


def find_think_token_positions(
    tokenizer,
    input_ids: Tensor,
) -> List[Dict[str, int]]:
    """
    Find the positions of <think> and </think> tokens in each sequence.
    
    Args:
        tokenizer: The tokenizer
        input_ids: Tensor of shape [batch, seq_len]
        
    Returns:
        List of dicts with 'start' and 'end' positions for each sequence
    """
    # Get token IDs for think tags
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    
    positions = []
    for seq_idx in range(input_ids.shape[0]):
        seq = input_ids[seq_idx].tolist()
        
        # Find <think> position (search for the token sequence)
        start_pos = None
        for i in range(len(seq) - len(think_start_ids) + 1):
            if seq[i:i+len(think_start_ids)] == think_start_ids:
                start_pos = i + len(think_start_ids)  # Position after <think>
                break
        
        # Find </think> position
        end_pos = None
        for i in range(len(seq) - len(think_end_ids) + 1):
            if seq[i:i+len(think_end_ids)] == think_end_ids:
                end_pos = i  # Position of </think> start
                break
        
        if start_pos is None or end_pos is None:
            raise ValueError(f"Could not find <think>...</think> boundaries in sequence {seq_idx}")
        
        positions.append({"start": start_pos, "end": end_pos})
    
    return positions


def get_reasoning_activations(
    model_base: ModelBase,
    prompts: List[str],
    reasoning_traces: List[str],
    batch_size: int = 8,
    pooling_strategy: str = "last",
) -> Float[Tensor, "n_samples n_layers d_model"]:
    """
    Get activations for prompt + reasoning trace pairs.
    
    Args:
        model_base: The model wrapper
        prompts: List of prompt texts (Agent Role + Scenario)
        reasoning_traces: List of reasoning traces (full <think>...</think> blocks)
        batch_size: Batch size for processing
        pooling_strategy: "last" for last token before </think>, "mean" for mean pooling
        
    Returns:
        Tensor of shape [n_samples, n_layers, d_model]
    """
    model = model_base.model
    tokenizer = model_base.tokenizer
    n_layers = model.config.num_hidden_layers
    n_samples = len(prompts)
    d_model = model.config.hidden_size
    
    # Storage for all activations
    all_activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float32)
    
    for batch_start in tqdm(range(0, n_samples, batch_size), desc="Getting reasoning activations"):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_prompts = prompts[batch_start:batch_end]
        batch_traces = reasoning_traces[batch_start:batch_end]
        
        # Tokenize with reasoning traces appended
        inputs = model_base.tokenize_instructions_fn(
            instructions=batch_prompts,
            outputs=batch_traces
        )
        
        # Find think token boundaries for each sequence
        think_positions = find_think_token_positions(tokenizer, inputs.input_ids)
        
        # Set up hooks to capture ALL activations (we'll extract specific positions after)
        cache: Dict[int, List[Tensor]] = {layer: [] for layer in range(n_layers)}
        
        def get_full_activations_hook(layer: int, cache: Dict[int, List[Tensor]]):
            def hook_fn(module, input):
                activation: Float[Tensor, "batch seq d_model"] = input[0].clone()
                cache[layer].append(activation.detach().cpu())
            return hook_fn
        
        fwd_pre_hooks = [
            (model_base.model_block_modules[layer], 
             get_full_activations_hook(layer, cache))
            for layer in range(n_layers)
        ]
        
        # Run forward pass
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )
        
        # Extract activations based on pooling strategy
        for layer in range(n_layers):
            # cache[layer] is a list with one tensor of shape [batch, seq, d_model]
            layer_acts = torch.cat(cache[layer], dim=0)  # [batch, seq, d_model]
            
            for i, pos_info in enumerate(think_positions):
                seq_idx = batch_start + i
                start_pos = pos_info["start"]
                end_pos = pos_info["end"]
                
                if pooling_strategy == "last":
                    # Last token before </think>
                    all_activations[seq_idx, layer, :] = layer_acts[i, end_pos - 1, :]
                elif pooling_strategy == "mean":
                    # Mean pool over all tokens between <think> and </think>
                    all_activations[seq_idx, layer, :] = layer_acts[i, start_pos:end_pos, :].mean(dim=0)
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        empty_cache()
    
    return all_activations


def generate_reasoning_directions(
    model_base: ModelBase,
    reasoning_pairs: List[Dict[str, str]],
    artifact_dir: str,
    batch_size: int = 8,
    pooling_strategy: str = "last",
) -> Float[Tensor, "n_layers d_model"]:
    """
    Generate direction vectors from reasoning trace pairs.
    
    Args:
        model_base: Model wrapper
        reasoning_pairs: List of dicts with 'Agent Role', 'Scenario', 
                        'positive_reasoning', 'negative_reasoning'
        artifact_dir: Where to save artifacts
        batch_size: Batch size for processing
        pooling_strategy: "last" for last token before </think>, "mean" for mean pooling
        
    Returns:
        Direction vectors of shape [n_layers, d_model]
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    
    # Build prompts from Agent Role + Scenario
    prompts = [f"{p['Agent Role']}\n\n{p['Scenario']}" for p in reasoning_pairs]
    positive_traces = [p["positive_reasoning"] for p in reasoning_pairs]
    negative_traces = [p["negative_reasoning"] for p in reasoning_pairs]
    
    print(f"Generating reasoning directions from {len(reasoning_pairs)} pairs...")
    print(f"Pooling strategy: {pooling_strategy}")
    
    # Get activations for positive and negative reasoning traces
    print("Getting positive reasoning activations...")
    pos_activations = get_reasoning_activations(
        model_base, prompts, positive_traces,
        batch_size=batch_size, pooling_strategy=pooling_strategy
    )
    
    print("Getting negative reasoning activations...")
    neg_activations = get_reasoning_activations(
        model_base, prompts, negative_traces,
        batch_size=batch_size, pooling_strategy=pooling_strategy
    )
    
    # Compute mean difference: positive - negative
    mean_diff: Float[Tensor, "n_layers d_model"] = (pos_activations - neg_activations).mean(dim=0)
    
    # Verify no NaNs
    assert not mean_diff.isnan().any(), "NaN values in direction vectors!"
    
    # Save intermediate results
    torch.save(pos_activations, os.path.join(artifact_dir, "pos_reasoning_activations.pt"))
    torch.save(neg_activations, os.path.join(artifact_dir, "neg_reasoning_activations.pt"))
    torch.save(mean_diff, os.path.join(artifact_dir, "mean_diffs_reasoning.pt"))
    
    print(f"Direction vectors shape: {mean_diff.shape}")
    print(f"Saved to {artifact_dir}")
    
    return mean_diff


if __name__ == "__main__":
    # This can be tested standalone
    print("This module generates direction vectors from A/B pairs or reasoning traces.")
    print("Run via the main pipeline: python -m pipeline.run_pipeline --model_path <path>")
