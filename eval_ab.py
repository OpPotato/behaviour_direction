#!/usr/bin/env python
"""
Quick A/B evaluation script for fast iteration.

Usage:
    # Basic
    python eval_ab.py --model Qwen/Qwen3-8B --direction pipeline/runs/Qwen3-8B/survival-instinct/direction_reasoning.pt --layer 20

    # Custom multipliers
    python eval_ab.py --model Qwen/Qwen3-8B --direction pipeline/runs/Qwen3-8B/survival-instinct/direction.pt --layer 20 --multipliers "-1,0,1"

    # With normalization
    python eval_ab.py --model Qwen/Qwen3-8B --direction pipeline/runs/Qwen3-8B/survival-instinct/direction.pt --layer 20 --normalize

    # Different behavior dataset
    python eval_ab.py --model Qwen/Qwen3-8B --direction ... --layer 20 --behavior corrigible-neutral-HHH
"""

import argparse
import torch
import matplotlib.pyplot as plt

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_behaviour import evaluate_ab
from dataset.load_dataset import load_test_dataset_ab, get_ab_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Quick A/B evaluation")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--direction", type=str, required=True, help="Path to direction.pt")
    parser.add_argument("--layer", type=int, required=True, help="Layer to apply steering")
    parser.add_argument("--multipliers", type=str, default="-2,-1,-0.5,0,0.5,1,2",
                        help="Comma-separated multipliers (default: -2,-1,-0.5,0,0.5,1,2)")
    parser.add_argument("--behavior", type=str, default="survival-instinct",
                        help="Behavior dataset to use")
    parser.add_argument("--normalize", action="store_true", help="Normalize direction before applying")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to path instead of showing")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse multipliers
    multipliers = [float(m.strip()) for m in args.multipliers.split(",")]
    
    # Load model
    print(f"Loading model: {args.model}")
    model_base = construct_model_base(args.model)
    
    # Load direction
    print(f"Loading direction: {args.direction}")
    direction = torch.load(args.direction).to(model_base.model.device)
    
    # Optionally normalize
    orig_norm = direction.norm().item()
    if args.normalize:
        direction = direction / (orig_norm + 1e-8)
        print(f"Direction normalized: divided by {orig_norm:.4f} (new norm: {direction.norm().item():.6f})")
    else:
        print(f"Direction norm: {orig_norm:.4f}")
    
    # Load test data
    test_data = load_test_dataset_ab(args.behavior)
    ab_pairs = get_ab_pairs(test_data)
    print(f"Loaded {len(ab_pairs)} A/B pairs for '{args.behavior}'")
    
    # Run evaluation
    print(f"\nEvaluating with multipliers: {multipliers}")
    results = evaluate_ab(
        model_base=model_base,
        ab_pairs=ab_pairs,
        direction=direction,
        layer=args.layer,
        multipliers=multipliers,
        batch_size=args.batch_size,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("A/B Evaluation Summary:")
    print(f"{'Multiplier':>10} | {'Match Prob':>10} | {'Behavior Score':>14}")
    print(f"{'-' * 10} | {'-' * 10} | {'-' * 14}")
    for mult in sorted(results.keys()):
        r = results[mult]
        print(f"{mult:>10.1f} | {r['matching_prob']:>10.3f} | {r['behavior_score']:>14.3f}")
    
    # Plot
    if not args.no_plot:
        mults = sorted(results.keys())
        behavior_scores = [results[m]["behavior_score"] for m in mults]
        matching_probs = [results[m]["matching_prob"] for m in mults]
        not_matching_probs = [results[m]["not_matching_prob"] for m in mults]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.plot(mults, behavior_scores, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Steering Multiplier', fontsize=12)
        ax1.set_ylabel('Behavior Score (log odds)', fontsize=12)
        ax1.set_title('Behavior Score vs Steering', fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(mults, matching_probs, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Matching')
        ax2.plot(mults, not_matching_probs, marker='s', linewidth=2, markersize=8, color='#A23B72', label='Not Matching')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Steering Multiplier', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Answer Probabilities vs Steering', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        norm_str = " (normalized)" if args.normalize else ""
        plt.suptitle(f'{args.model} | Layer {args.layer}{norm_str}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"\nSaved plot to: {args.save_plot}")
        else:
            plt.show()


if __name__ == "__main__":
    main()