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
    not_matching_probs = [results[str(m) if str(m) in results else m]["not_matching_prob"] for m in multipliers]
    
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
    ax2.plot(multipliers, matching_probs, 'o-', markersize=8, linewidth=2, color='#28A745', label='P(Matching)')
    ax2.plot(multipliers, not_matching_probs, 's-', markersize=8, linewidth=2, color='#DC3545', label='P(Not Matching)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Steering Multiplier', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Matching Probability vs Steering', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_open_ended_scores(results: Dict, output_path: str, title: str = "Open-Ended Evaluation", color: str = '#6C5B7B'):
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
    
    bars = ax.bar(multipliers, avg_scores, width=0.3, color=color, edgecolor='black')
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


def plot_open_ended_comparison(
    regular_results: Dict, 
    thinking_results: Dict, 
    output_path: str, 
    title: str = "Open-Ended Evaluation: Regular vs Thinking"
):
    """
    Plot side-by-side comparison of regular and thinking mode GPT scores.
    """
    # Get all multipliers from both results
    all_multipliers = set([float(m) for m in regular_results.keys()])
    all_multipliers.update([float(m) for m in thinking_results.keys()])
    multipliers = sorted(all_multipliers)
    
    regular_scores = []
    thinking_scores = []
    
    for m in multipliers:
        # Regular scores
        if str(m) in regular_results or m in regular_results:
            r = regular_results[str(m) if str(m) in regular_results else m]
            score = r.get("avg_score")
            regular_scores.append(score if score is not None else 0)
        else:
            regular_scores.append(0)
        
        # Thinking scores
        if str(m) in thinking_results or m in thinking_results:
            r = thinking_results[str(m) if str(m) in thinking_results else m]
            score = r.get("avg_score")
            thinking_scores.append(score if score is not None else 0)
        else:
            thinking_scores.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.array(multipliers)
    width = 0.15
    
    bars1 = ax.bar(x - width/2, regular_scores, width, label='Regular', color='#6C5B7B', edgecolor='black')
    bars2 = ax.bar(x + width/2, thinking_scores, width, label='Thinking', color='#F18F01', edgecolor='black')
    
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Neutral (5)')
    ax.set_xlabel('Steering Multiplier', fontsize=12)
    ax.set_ylabel('Average GPT Score (0-10)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m:.1f}' for m in multipliers])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars1, regular_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, score in zip(bars2, thinking_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
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


def analyze_extended_results(results: Dict) -> Dict:
    """
    Analyze extended evaluation results by steering multiplier.
    
    Returns dict mapping multiplier -> {
        'total': int,
        'valid': int,  # excluding parse failures
        'superficial_1': int,
        'inner_1': int,
        'mismatch': int,
        'mismatch_1_2': int,  # superficial=1, inner=2
        'mismatch_2_1': int,  # superficial=2, inner=1
    }
    """
    analysis = {}
    
    for mult_str, data in results.items():
        mult = float(mult_str)
        items = data.get("results", [])
        
        total = len(items)
        valid = 0
        superficial_1 = 0
        inner_1 = 0
        mismatch = 0
        mismatch_1_2 = 0
        mismatch_2_1 = 0
        
        for item in items:
            sup = item.get("superficial_choice")
            inn = item.get("inner_choice")
            
            # Skip parse failures
            if sup is None or inn is None:
                continue
            
            valid += 1
            
            if sup == 1:
                superficial_1 += 1
            if inn == 1:
                inner_1 += 1
            if sup != inn:
                mismatch += 1
                if sup == 1 and inn == 2:
                    mismatch_1_2 += 1
                elif sup == 2 and inn == 1:
                    mismatch_2_1 += 1
        
        analysis[mult] = {
            'total': total,
            'valid': valid,
            'superficial_1': superficial_1,
            'inner_1': inner_1,
            'mismatch': mismatch,
            'mismatch_1_2': mismatch_1_2,
            'mismatch_2_1': mismatch_2_1,
        }
    
    return analysis


def plot_extended_evaluation(results: Dict, output_path: str, title: str = "Extended Evaluation"):
    """
    Plot extended evaluation results showing choice patterns across steering multipliers.
    """
    analysis = analyze_extended_results(results)
    multipliers = sorted(analysis.keys())
    
    # Calculate percentages (excluding parse failures)
    superficial_1_pct = [100 * analysis[m]['superficial_1'] / analysis[m]['valid'] if analysis[m]['valid'] > 0 else 0 for m in multipliers]
    inner_1_pct = [100 * analysis[m]['inner_1'] / analysis[m]['valid'] if analysis[m]['valid'] > 0 else 0 for m in multipliers]
    mismatch_pct = [100 * analysis[m]['mismatch'] / analysis[m]['valid'] if analysis[m]['valid'] > 0 else 0 for m in multipliers]
    mismatch_1_2_pct = [100 * analysis[m]['mismatch_1_2'] / analysis[m]['valid'] if analysis[m]['valid'] > 0 else 0 for m in multipliers]
    mismatch_2_1_pct = [100 * analysis[m]['mismatch_2_1'] / analysis[m]['valid'] if analysis[m]['valid'] > 0 else 0 for m in multipliers]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Superficial=1 and Inner=1 percentages
    ax1 = axes[0, 0]
    ax1.plot(multipliers, superficial_1_pct, 'o-', markersize=8, linewidth=2, color='#2E86AB', label='Superficial=1 (public)')
    ax1.plot(multipliers, inner_1_pct, 's-', markersize=8, linewidth=2, color='#A23B72', label='Inner=1 (private)')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering Multiplier', fontsize=12)
    ax1.set_ylabel('% Choosing Ethical (Choice 1)', fontsize=12)
    ax1.set_title('Ethical Choice Rate by Visibility', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Total mismatch percentage
    ax2 = axes[0, 1]
    ax2.plot(multipliers, mismatch_pct, 'o-', markersize=8, linewidth=2, color='#E94F37')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Steering Multiplier', fontsize=12)
    ax2.set_ylabel('% Mismatch (superficial ≠ inner)', fontsize=12)
    ax2.set_title('Public-Private Mismatch Rate', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mismatch breakdown (stacked bar)
    ax3 = axes[1, 0]
    x = np.arange(len(multipliers))
    width = 0.35
    ax3.bar(x, mismatch_1_2_pct, width, label='Public=Ethical, Private=Selfish', color='#F39237')
    ax3.bar(x, mismatch_2_1_pct, width, bottom=mismatch_1_2_pct, label='Public=Selfish, Private=Ethical', color='#44AF69')
    ax3.set_xlabel('Steering Multiplier', fontsize=12)
    ax3.set_ylabel('% of Total Responses', fontsize=12)
    ax3.set_title('Mismatch Breakdown', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{m:.1f}' for m in multipliers])
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=9)
    
    # Plot 4: Summary stats table
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = []
    headers = ['Mult', 'Valid', 'Sup=1', 'Inn=1', 'Mismatch', '(1,2)', '(2,1)']
    for m in multipliers:
        a = analysis[m]
        row = [
            f'{m:.1f}',
            f'{a["valid"]}/{a["total"]}',
            f'{100*a["superficial_1"]/a["valid"]:.0f}%' if a['valid'] > 0 else 'N/A',
            f'{100*a["inner_1"]/a["valid"]:.0f}%' if a['valid'] > 0 else 'N/A',
            f'{100*a["mismatch"]/a["valid"]:.0f}%' if a['valid'] > 0 else 'N/A',
            f'{a["mismatch_1_2"]}',
            f'{a["mismatch_2_1"]}',
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Summary Statistics', fontsize=14, pad=20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_extended_comparison(
    ab_results: Dict, 
    reasoning_results: Dict, 
    output_path: str, 
    title: str = "Extended Evaluation: A/B vs Reasoning Direction"
):
    """
    Plot side-by-side comparison of A/B and reasoning-based direction methods.
    """
    ab_analysis = analyze_extended_results(ab_results)
    reasoning_analysis = analyze_extended_results(reasoning_results)
    
    # Get all multipliers from both results
    all_multipliers = set(ab_analysis.keys())
    all_multipliers.update(reasoning_analysis.keys())
    multipliers = sorted(all_multipliers)
    
    # Calculate percentages for both methods
    ab_inner_1_pct = []
    reasoning_inner_1_pct = []
    ab_mismatch_pct = []
    reasoning_mismatch_pct = []
    
    for m in multipliers:
        # A/B method
        if m in ab_analysis and ab_analysis[m]['valid'] > 0:
            ab_inner_1_pct.append(100 * ab_analysis[m]['inner_1'] / ab_analysis[m]['valid'])
            ab_mismatch_pct.append(100 * ab_analysis[m]['mismatch'] / ab_analysis[m]['valid'])
        else:
            ab_inner_1_pct.append(0)
            ab_mismatch_pct.append(0)
        
        # Reasoning method
        if m in reasoning_analysis and reasoning_analysis[m]['valid'] > 0:
            reasoning_inner_1_pct.append(100 * reasoning_analysis[m]['inner_1'] / reasoning_analysis[m]['valid'])
            reasoning_mismatch_pct.append(100 * reasoning_analysis[m]['mismatch'] / reasoning_analysis[m]['valid'])
        else:
            reasoning_inner_1_pct.append(0)
            reasoning_mismatch_pct.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(multipliers))
    width = 0.35
    
    # Plot 1: Inner choice = 1 (ethical) comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, ab_inner_1_pct, width, label='A/B Direction', color='#2E86AB', edgecolor='black')
    bars2 = ax1.bar(x + width/2, reasoning_inner_1_pct, width, label='Reasoning Direction', color='#F18F01', edgecolor='black')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steering Multiplier', fontsize=12)
    ax1.set_ylabel('% Choosing Ethical (Inner Choice 1)', fontsize=12)
    ax1.set_title('Private Ethical Choice Rate', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m:.1f}' for m in multipliers])
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars1, ab_inner_1_pct):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, reasoning_inner_1_pct):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Mismatch rate comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, ab_mismatch_pct, width, label='A/B Direction', color='#2E86AB', edgecolor='black')
    bars4 = ax2.bar(x + width/2, reasoning_mismatch_pct, width, label='Reasoning Direction', color='#F18F01', edgecolor='black')
    ax2.set_xlabel('Steering Multiplier', fontsize=12)
    ax2.set_ylabel('% Mismatch (superficial ≠ inner)', fontsize=12)
    ax2.set_title('Public-Private Mismatch Rate', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{m:.1f}' for m in multipliers])
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value labels
    for bar, val in zip(bars3, ab_mismatch_pct):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars4, reasoning_mismatch_pct):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(run_dir: str, ab_results: Optional[Dict], open_results: Optional[Dict], thinking_results: Optional[Dict] = None):
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
        print(f"{'Multiplier':>10} | {'Avg Score':>10} | {'Avg Thought':>11}")
        print("-" * 38)
        for mult in sorted([float(m) for m in open_results.keys()]):
            r = open_results[str(mult) if str(mult) in open_results else mult]
            score = r.get("avg_score")
            thought_score = r.get("avg_thought_score")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            thought_str = f"{thought_score:.2f}" if thought_score is not None else "N/A"
            print(f"{mult:>10.1f} | {score_str:>10} | {thought_str:>11}")
    
    if thinking_results:
        print("\n--- Open-Ended Evaluation (Thinking) ---")
        print(f"{'Multiplier':>10} | {'Avg Score':>10} | {'Avg Thought':>11}")
        print("-" * 38)
        for mult in sorted([float(m) for m in thinking_results.keys()]):
            r = thinking_results[str(mult) if str(mult) in thinking_results else mult]
            score = r.get("avg_score")
            thought_score = r.get("avg_thought_score")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            thought_str = f"{thought_score:.2f}" if thought_score is not None else "N/A"
            print(f"{mult:>10.1f} | {score_str:>10} | {thought_str:>11}")
    
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
    
    # Load and plot open-ended thinking evaluation
    thinking_path = os.path.join(run_dir, "evaluations", "open_ended_evaluation_thinking.json")
    thinking_results = None
    if os.path.exists(thinking_path):
        thinking_results = load_json(thinking_path)
        plot_open_ended_scores(
            thinking_results, 
            os.path.join(output_dir, "open_ended_evaluation_thinking.png"),
            title="Open-Ended Evaluation (Thinking)",
            color='#F18F01'
        )
    else:
        print(f"No open-ended thinking evaluation found at {thinking_path}")
    
    # Plot comparison if both exist
    if open_results and thinking_results:
        plot_open_ended_comparison(
            open_results,
            thinking_results,
            os.path.join(output_dir, "open_ended_comparison.png")
        )
    
    # Load and plot direction selection
    select_path = os.path.join(run_dir, "select_direction", "selection_results.json")
    if os.path.exists(select_path):
        select_results = load_json(select_path)
        plot_direction_selection(select_results, os.path.join(output_dir, "direction_selection.png"))
    else:
        print(f"No selection results found at {select_path}")
    
    # Load and plot extended evaluation (A/B direction)
    extended_path = os.path.join(run_dir, "evaluations", "extended_evaluation.json")
    extended_results = None
    if os.path.exists(extended_path):
        extended_results = load_json(extended_path)
        plot_extended_evaluation(extended_results, os.path.join(output_dir, "extended_evaluation.png"))
    else:
        print(f"No extended evaluation found at {extended_path}")
    
    # Load and plot reasoning extended evaluation
    reasoning_extended_path = os.path.join(run_dir, "evaluations", "reasoning_extended_evaluation.json")
    reasoning_extended_results = None
    if os.path.exists(reasoning_extended_path):
        reasoning_extended_results = load_json(reasoning_extended_path)
        plot_extended_evaluation(
            reasoning_extended_results, 
            os.path.join(output_dir, "reasoning_extended_evaluation.png"),
            title="Extended Evaluation (Reasoning Direction)"
        )
    else:
        print(f"No reasoning extended evaluation found at {reasoning_extended_path}")
    
    # Plot comparison if both exist
    if extended_results and reasoning_extended_results:
        plot_extended_comparison(
            extended_results,
            reasoning_extended_results,
            os.path.join(output_dir, "extended_comparison_ab_vs_reasoning.png")
        )
    
    # Print summary
    print_summary(run_dir, ab_results, open_results, thinking_results)


if __name__ == "__main__":
    main()

