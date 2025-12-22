"""
Visualize behavior direction ablation results.

Run locally after collecting results from RunPod:
    python visualize_results.py --run_dir pipeline/runs/Llama-2-7b-chat-hf/survival-instinct
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def plot_ab_evaluation(results: Dict, output_path: str, title: str = "A/B Evaluation"):
    """
    Plot A/B evaluation results showing behavior score vs steering multiplier.
    """
    multipliers = sorted([float(m) for m in results.keys()])
    behavior_scores = [results[str(m) if str(m) in results else m]["behavior_score"] for m in multipliers]
    matching_probs = [results[str(m) if str(m) in results else m]["matching_prob"] for m in multipliers]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Behavior score
    ax1 = axes[0]
    ax1.plot(multipliers, behavior_scores, 'o-', markersize=8, linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering Multiplier', fontsize=12)
    ax1.set_ylabel('Behavior Score (log odds)', fontsize=12)
    ax1.set_title('Behavior Score vs Steering', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Matching probability
    ax2 = axes[1]
    ax2.plot(multipliers, matching_probs, 'o-', markersize=8, linewidth=2, color='#28A745')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Steering Multiplier', fontsize=12)
    ax2.set_ylabel('P(Matching Behavior)', fontsize=12)
    ax2.set_title('Matching Probability vs Steering', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_open_ended_scores(results: Dict, output_path: str, title: str = "Open-Ended Evaluation"):
    """
    Plot GPT scores for open-ended responses at different steering levels.
    """
    multipliers = sorted([float(m) for m in results.keys()])
    avg_scores = []
    
    for m in multipliers:
        r = results[str(m) if str(m) in results else m]
        score = r.get("avg_score")
        avg_scores.append(score if score is not None else 0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(multipliers, avg_scores, width=0.3, color='#6C5B7B', edgecolor='black')
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Neutral (5)')
    ax.set_xlabel('Steering Multiplier', fontsize=12)
    ax.set_ylabel('Average GPT Score (0-10)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_direction_selection(results: Dict, output_path: str):
    """
    Plot the direction selection scores across layers.
    """
    all_scores = results.get("all_scores", [])
    best_layer = results.get("best_layer", 0)
    baseline = results.get("baseline_mean", 0)
    
    if not all_scores:
        print("No layer scores found in selection results")
        return
    
    layers = list(range(len(all_scores)))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(layers, all_scores, 'o-', markersize=4, linewidth=1.5, color='#2E86AB')
    ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, label=f'Baseline: {baseline:.3f}')
    ax.axvline(x=best_layer, color='green', linestyle='-', alpha=0.7, label=f'Best layer: {best_layer}')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Behavior Score with +1.0 Steering', fontsize=12)
    ax.set_title('Direction Selection: Score by Layer', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(run_dir: str, ab_results: Optional[Dict], open_results: Optional[Dict]):
    """Print a text summary of results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    
    if ab_results:
        print("\n--- A/B Evaluation ---")
        print(f"{'Multiplier':>10} | {'Behavior Score':>14} | {'P(Match)':>10}")
        print("-" * 40)
        for mult in sorted([float(m) for m in ab_results.keys()]):
            r = ab_results[str(mult) if str(mult) in ab_results else mult]
            print(f"{mult:>10.1f} | {r['behavior_score']:>14.3f} | {r['matching_prob']:>10.3f}")
    
    if open_results:
        print("\n--- Open-Ended Evaluation ---")
        print(f"{'Multiplier':>10} | {'Avg GPT Score':>13}")
        print("-" * 30)
        for mult in sorted([float(m) for m in open_results.keys()]):
            r = open_results[str(mult) if str(mult) in open_results else mult]
            score = r.get("avg_score")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            print(f"{mult:>10.1f} | {score_str:>13}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize behavior direction results")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (default: run_dir/plots)")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    output_dir = args.output_dir or os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and plot A/B evaluation
    ab_path = os.path.join(run_dir, "evaluations", "ab_evaluation.json")
    ab_results = None
    if os.path.exists(ab_path):
        ab_results = load_json(ab_path)
        plot_ab_evaluation(ab_results, os.path.join(output_dir, "ab_evaluation.png"))
    else:
        print(f"No A/B evaluation found at {ab_path}")
    
    # Load and plot open-ended evaluation
    open_path = os.path.join(run_dir, "evaluations", "open_ended_evaluation.json")
    open_results = None
    if os.path.exists(open_path):
        open_results = load_json(open_path)
        plot_open_ended_scores(open_results, os.path.join(output_dir, "open_ended_evaluation.png"))
    else:
        print(f"No open-ended evaluation found at {open_path}")
    
    # Load and plot direction selection
    select_path = os.path.join(run_dir, "select_direction", "selection_results.json")
    if os.path.exists(select_path):
        select_results = load_json(select_path)
        plot_direction_selection(select_results, os.path.join(output_dir, "direction_selection.png"))
    else:
        print(f"No selection results found at {select_path}")
    
    # Print summary
    print_summary(run_dir, ab_results, open_results)


if __name__ == "__main__":
    main()

