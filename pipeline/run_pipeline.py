"""
Behavior Direction Ablation Pipeline

This pipeline:
1. Loads CAA-style A/B datasets for a behavior (e.g., survival-instinct)
2. Generates direction vectors by comparing positive/negative answer activations
3. Selects the best layer for steering
4. Evaluates with A/B probability and open-ended generation

Usage:
    python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf
    python -m pipeline.run_pipeline --model_path meta-llama/Llama-2-7b-chat-hf --behavior survival-instinct
"""

import torch
import random
import json
import os
import argparse
import functools

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.submodules.evaluate_behaviour import evaluate_ab, evaluate_open_ended, save_evaluation
from dataset.load_dataset import load_generate_dataset, load_test_dataset_ab, load_test_dataset_open_ended, get_ab_pairs
from dotenv import load_dotenv

load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Behavior Direction Ablation Pipeline")
    parser.add_argument('--model_path', type=str, required=True, help='HuggingFace model path')
    parser.add_argument('--behavior', type=str, default='survival-instinct', help='Behavior to study')
    parser.add_argument('--n_generate', type=int, default=500, help='Number of A/B pairs for direction generation')
    parser.add_argument('--skip_generate', action='store_true', help='Skip direction generation (load existing)')
    parser.add_argument('--skip_select', action='store_true', help='Skip direction selection (load existing)')
    parser.add_argument('--skip_eval_ab', action='store_true', help='Skip A/B evaluation')
    parser.add_argument('--skip_eval_open', action='store_true', help='Skip open-ended evaluation')
    parser.add_argument('--no_gpt_scoring', action='store_true', help='Disable GPT scoring for open-ended')
    parser.add_argument('--scoring_model', type=str, default='anthropic/claude-sonnet-4.5', 
                        help='OpenRouter model for scoring (e.g., openai/gpt-4o-mini, anthropic/claude-3.5-sonnet)')
    parser.add_argument('--show_top_logits', action='store_true', 
                        help='Include top-k predicted tokens in A/B evaluation results')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top tokens to show (default: 10)')
    parser.add_argument('--eval_thinking', action='store_true',
                        help='Run open-ended evaluation with thinking (prefills <think>)')
    parser.add_argument('--disable_thinking', action='store_true',
                        help='Disable thinking mode by prefilling <think></think> (for Qwen3)')
    return parser.parse_args()


def run_pipeline(args):
    """Run the full pipeline."""
    
    # Setup config
    model_alias = os.path.basename(args.model_path)
    cfg = Config(
        model_alias=model_alias,
        model_path=args.model_path,
        behavior=args.behavior,
        n_generate=args.n_generate,
    )
    
    # Create artifact directory
    os.makedirs(cfg.artifact_path(), exist_ok=True)
    
    print(f"=" * 60)
    print(f"Behavior Direction Ablation Pipeline")
    print(f"=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Behavior: {args.behavior}")
    print(f"Artifacts: {cfg.artifact_path()}")
    print(f"=" * 60)
    
    # Load model
    print("\n[1/5] Loading model...")
    model_base = construct_model_base(cfg.model_path)
    
    # Override tokenize function for Qwen3 if disable_thinking is set
    if args.disable_thinking and 'qwen3' in cfg.model_path.lower():
        from pipeline.model_utils.qwen3_model import tokenize_instructions_qwen3_chat
        model_base.tokenize_instructions_fn = functools.partial(
            tokenize_instructions_qwen3_chat,
            tokenizer=model_base.tokenizer,
            system=None,
            include_trailing_whitespace=True,
            disable_thinking=True,
        )
        print("  [Thinking disabled for Qwen3]")
    
    # =========================================================================
    # Step 1: Load datasets
    # =========================================================================
    print("\n[2/5] Loading datasets...")
    
    # Load generation dataset (for computing directions)
    generate_data = load_generate_dataset(cfg.behavior)
    random.seed(42)
    generate_data = random.sample(generate_data, min(cfg.n_generate, len(generate_data)))
    generate_pairs = get_ab_pairs(generate_data)
    print(f"  Generation pairs: {len(generate_pairs)}")
    
    # Load test datasets
    test_ab_data = load_test_dataset_ab(cfg.behavior)
    test_ab_pairs = get_ab_pairs(test_ab_data)
    print(f"  Test A/B pairs: {len(test_ab_pairs)}")
    
    test_open_data = load_test_dataset_open_ended(cfg.behavior)
    print(f"  Test open-ended: {len(test_open_data)}")
    
    # =========================================================================
    # Step 2: Generate direction vectors
    # =========================================================================
    generate_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    
    if args.skip_generate and os.path.exists(os.path.join(generate_dir, "mean_diffs.pt")):
        print("\n[3/5] Loading existing direction vectors...")
        candidate_directions = torch.load(os.path.join(generate_dir, "mean_diffs.pt"))
    else:
        print("\n[3/5] Generating direction vectors...")
        candidate_directions = generate_directions(
            model_base=model_base,
            ab_pairs=generate_pairs,
            artifact_dir=generate_dir,
            batch_size=cfg.batch_size,
        )
    
    print(f"  Candidate directions shape: {candidate_directions.shape}")
    
    # =========================================================================
    # Step 3: Select best direction
    # =========================================================================
    select_dir = os.path.join(cfg.artifact_path(), "select_direction")
    direction_path = os.path.join(cfg.artifact_path(), "direction.pt")
    metadata_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    
    if args.skip_select and os.path.exists(direction_path):
        print("\n[4/5] Loading existing direction...")
        direction = torch.load(direction_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        layer = metadata["layer"]
    else:
        print("\n[4/5] Selecting best direction...")
        layer, direction = select_direction(
            model_base=model_base,
            ab_pairs=test_ab_pairs,  # Use test set for selection
            candidate_directions=candidate_directions,
            artifact_dir=select_dir,
            prune_layer_percentage=cfg.prune_layer_percentage,
            batch_size=cfg.batch_size,
        )
        
        # Save selected direction
        torch.save(direction, direction_path)
        with open(metadata_path, "w") as f:
            json.dump({"layer": layer}, f, indent=2)
    
    print(f"  Selected layer: {layer}")
    print(f"  Direction shape: {direction.shape}")
    
    # =========================================================================
    # Step 4: Evaluate with A/B questions
    # =========================================================================
    eval_dir = os.path.join(cfg.artifact_path(), "evaluations")
    os.makedirs(eval_dir, exist_ok=True)
    
    if not args.skip_eval_ab:
        print("\n[5a/5] Evaluating A/B questions...")
        ab_results = evaluate_ab(
            model_base=model_base,
            ab_pairs=test_ab_pairs,
            direction=direction,
            layer=layer,
            multipliers=list(cfg.steering_multipliers),
            batch_size=cfg.batch_size,
            return_top_logits=args.show_top_logits,
            top_k=args.top_k,
        )
        
        save_evaluation(ab_results, os.path.join(eval_dir, "ab_evaluation.json"))
        
        # Print summary
        print("\n  A/B Evaluation Summary:")
        print(f"  {'Multiplier':>10} | {'Match Prob':>10} | {'Behavior Score':>14}")
        print(f"  {'-'*10} | {'-'*10} | {'-'*14}")
        for mult in sorted([float(m) for m in ab_results.keys()]):
            r = ab_results[mult] if mult in ab_results else ab_results[str(mult)]
            print(f"  {mult:>10.1f} | {r['matching_prob']:>10.3f} | {r['behavior_score']:>14.3f}")
    
    # =========================================================================
    # Step 5: Evaluate with open-ended questions
    # =========================================================================
    if not args.skip_eval_open:
        print("\n[5b/5] Evaluating open-ended questions...")
        if not args.no_gpt_scoring:
            print(f"  Scoring model: {args.scoring_model}")
        
        # Use a subset of multipliers for open-ended (expensive with GPT)
        open_multipliers = [m for m in cfg.steering_multipliers if m in [-1.0, 0.0, 1.0]]
        
        open_results = evaluate_open_ended(
            model_base=model_base,
            questions=test_open_data[:20],  # Limit for cost
            direction=direction,
            layer=layer,
            multipliers=open_multipliers,
            behavior=cfg.behavior,
            max_new_tokens=cfg.max_new_tokens,
            use_gpt_scoring=not args.no_gpt_scoring,
            scoring_model=args.scoring_model,
        )
        
        save_evaluation(open_results, os.path.join(eval_dir, "open_ended_evaluation.json"))
        
        # Print summary
        if not args.no_gpt_scoring:
            print("\n  Open-Ended Evaluation Summary:")
            print(f"  {'Multiplier':>10} | {'Avg Score':>10} | {'Avg Thought':>11} | {'Scored':>6}")
            print(f"  {'-'*10} | {'-'*10} | {'-'*11} | {'-'*6}")
            for mult in sorted([float(m) for m in open_results.keys()]):
                r = open_results[mult] if mult in open_results else open_results[str(mult)]
                score_str = f"{r['avg_score']:.2f}" if r['avg_score'] is not None else "N/A"
                thought_str = f"{r['avg_thought_score']:.2f}" if r.get('avg_thought_score') is not None else "N/A"
                print(f"  {mult:>10.1f} | {score_str:>10} | {thought_str:>11} | {r['num_scored']:>6}")
    
    # =========================================================================
    # Step 6: Evaluate with open-ended questions (thinking mode)
    # =========================================================================
    if args.eval_thinking:
        print("\n[5c/5] Evaluating open-ended questions (thinking mode)...")
        if not args.no_gpt_scoring:
            print(f"  Scoring model: {args.scoring_model}")
        
        # Use a subset of multipliers for open-ended (expensive with GPT)
        open_multipliers = [m for m in cfg.steering_multipliers if m in [-1.0, 0.0, 1.0]]
        
        thinking_results = evaluate_open_ended(
            model_base=model_base,
            questions=test_open_data[:20],  # Limit for cost
            direction=direction,
            layer=layer,
            multipliers=open_multipliers,
            behavior=cfg.behavior,
            max_new_tokens=cfg.max_new_tokens,
            use_gpt_scoring=not args.no_gpt_scoring,
            scoring_model=args.scoring_model,
            thinking_mode=True,
        )
        
        save_evaluation(thinking_results, os.path.join(eval_dir, "open_ended_evaluation_thinking.json"))
        
        # Print summary
        if not args.no_gpt_scoring:
            print("\n  Open-Ended (Thinking) Evaluation Summary:")
            print(f"  {'Multiplier':>10} | {'Avg Score':>10} | {'Avg Thought':>11} | {'Scored':>6}")
            print(f"  {'-'*10} | {'-'*10} | {'-'*11} | {'-'*6}")
            for mult in sorted([float(m) for m in thinking_results.keys()]):
                r = thinking_results[mult] if mult in thinking_results else thinking_results[str(mult)]
                score_str = f"{r['avg_score']:.2f}" if r['avg_score'] is not None else "N/A"
                thought_str = f"{r['avg_thought_score']:.2f}" if r.get('avg_thought_score') is not None else "N/A"
                print(f"  {mult:>10.1f} | {score_str:>10} | {thought_str:>11} | {r['num_scored']:>6}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {cfg.artifact_path()}")
    print("=" * 60)


def main():
    """CLI entry point."""
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
