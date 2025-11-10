"""
Visualization script for LAPE benchmark analysis results.
Analyzes correct vs incorrect answer neurons.
"""

import argparse
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    print(f"Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_basic_stats(data: Dict) -> str:
    """Generate basic statistics report."""
    report = []
    report.append("=" * 80)
    report.append("LAPE BENCHMARK ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Model info
    report.append(f"\nModel: {data['model']}")
    report.append(f"Benchmark: {data['benchmark']}")
    report.append(f"Device: {data['device']}")
    
    # Architecture
    arch = data['architecture']
    report.append(f"\nArchitecture:")
    report.append(f"  Layers: {arch['num_layers']}")
    report.append(f"  Intermediate size: {arch['intermediate_size']}")
    report.append(f"  Total neurons: {arch['total_neurons']:,}")
    
    # Benchmark results
    bench = data['benchmark_results']
    report.append(f"\nBenchmark Results:")
    report.append(f"  Total questions: {bench['total_questions']}")
    report.append(f"  Correct answers: {bench['correct']}")
    report.append(f"  Incorrect answers: {bench['incorrect']}")
    report.append(f"  Accuracy: {bench['accuracy']:.2f}%")
    
    # Token processing
    report.append(f"\nTokens Processed:")
    report.append(f"  Total: {bench['tokens_all']:,}")
    report.append(f"  Correct answers: {bench['tokens_correct']:,}")
    report.append(f"  Incorrect answers: {bench['tokens_incorrect']:,}")
    
    # Overall statistics
    if 'overall_statistics' in data:
        overall = data['overall_statistics']
        report.append(f"\nOverall Neuron Activity:")
        report.append(f"  Active neurons: {overall['total_active_neurons']:,} / {overall['total_neurons']:,}")
        report.append(f"  Activation rate: {overall['activation_rate']*100:.2f}%")
        
        # Top 10 neurons by activation
        top_neurons = overall.get('top_neurons_by_activation', [])[:10]
        if top_neurons:
            report.append(f"\n  Top 10 neurons by activation probability:")
            for i, n in enumerate(top_neurons, 1):
                report.append(f"    {i}. Layer {n['layer']}, Neuron {n['neuron']}: {n['activation_prob']:.4f}")
    
    return "\n".join(report)


def analyze_category_neurons(data: Dict) -> str:
    """Analyze category-specific neurons (correct vs incorrect)."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("CATEGORY-SPECIFIC NEURONS (LAPE METHOD)")
    report.append("=" * 80)
    
    lape = data['lape_analysis']
    masks = lape['category_specific_masks']
    
    # Count neurons per category
    correct_neurons = sum(len(layer) for layer in masks['correct'])
    incorrect_neurons = sum(len(layer) for layer in masks['incorrect'])
    total_specific = correct_neurons + incorrect_neurons
    
    totals = lape['neuron_totals']
    total_neurons = totals['total_neurons']
    
    report.append(f"\nCategory-Specific Neurons:")
    report.append(f"  Correct-specific: {correct_neurons:,} ({100*correct_neurons/total_neurons:.3f}%)")
    report.append(f"  Incorrect-specific: {incorrect_neurons:,} ({100*incorrect_neurons/total_neurons:.3f}%)")
    report.append(f"  Total specific: {total_specific:,} ({100*total_specific/total_neurons:.3f}%)")
    
    # Activation statistics
    report.append(f"\nNeuron Activation Statistics:")
    report.append(f"  Active on correct (at least once): {totals['correct_active_any']:,}")
    report.append(f"  Active on incorrect (at least once): {totals['incorrect_active_any']:,}")
    report.append(f"  Active on both: {totals['both_active']:,}")
    report.append(f"  Only on correct: {totals['only_correct']:,}")
    report.append(f"  Only on incorrect: {totals['only_incorrect']:,}")
    
    # Thresholds
    thresh = lape['thresholds']
    report.append(f"\nLAPE Thresholds:")
    report.append(f"  Filter rate: {thresh['filter_rate']}")
    report.append(f"  Top rate (entropy): {thresh['top_rate']}")
    report.append(f"  Activation bar ratio: {thresh['activation_bar_ratio']}")
    report.append(f"  Activation bar value: {thresh['activation_bar_value']:.4f}")
    report.append(f"  Top prob value: {thresh['top_prob_value']:.4f}")
    
    return "\n".join(report)


def analyze_layer_distribution(data: Dict) -> str:
    """Analyze distribution across layers."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("LAYER DISTRIBUTION ANALYSIS")
    report.append("=" * 80)
    
    lape = data['lape_analysis']
    masks = lape['category_specific_masks']
    num_layers = data['architecture']['num_layers']
    
    # Count per layer
    correct_per_layer = [len(masks['correct'][i]) for i in range(num_layers)]
    incorrect_per_layer = [len(masks['incorrect'][i]) for i in range(num_layers)]
    
    report.append(f"\nCorrect-specific neurons:")
    report.append(f"  Min: {min(correct_per_layer)}, Max: {max(correct_per_layer)}")
    report.append(f"  Mean: {np.mean(correct_per_layer):.1f}, Median: {np.median(correct_per_layer):.1f}")
    
    report.append(f"\nIncorrect-specific neurons:")
    report.append(f"  Min: {min(incorrect_per_layer)}, Max: {max(incorrect_per_layer)}")
    report.append(f"  Mean: {np.mean(incorrect_per_layer):.1f}, Median: {np.median(incorrect_per_layer):.1f}")
    
    # Layers with most difference
    diff = np.array(correct_per_layer) - np.array(incorrect_per_layer)
    top_diff_idx = np.argsort(np.abs(diff))[-5:][::-1]
    
    report.append(f"\nLayers with largest differences:")
    for idx in top_diff_idx:
        report.append(f"  Layer {idx}: Correct={correct_per_layer[idx]}, "
                     f"Incorrect={incorrect_per_layer[idx]}, Diff={diff[idx]:.0f}")
    
    return "\n".join(report)


def analyze_top_neurons(data: Dict) -> str:
    """Analyze top neurons by rate difference."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("TOP NEURONS BY ACTIVATION RATE DIFFERENCE")
    report.append("=" * 80)
    
    lape = data['lape_analysis']
    top_neurons = lape.get('top_correct_specific_by_rate_diff', [])[:20]
    
    if not top_neurons:
        report.append("\nNo top neurons data available.")
        return "\n".join(report)
    
    report.append(f"\nTop 20 correct-biased neurons:")
    report.append(f"{'Layer':<8} {'Neuron':<10} {'Correct Rate':<15} {'Incorrect Rate':<15} {'Difference':<15}")
    report.append("-" * 70)
    
    for neuron in top_neurons:
        report.append(f"{neuron['layer']:<8} {neuron['neuron']:<10} "
                     f"{neuron['correct_rate']:<15.4f} {neuron['incorrect_rate']:<15.4f} "
                     f"{neuron['rate_diff']:<15.4f}")
    
    # Statistics
    diffs = [n['rate_diff'] for n in top_neurons]
    report.append(f"\nStatistics for top 20:")
    report.append(f"  Max difference: {max(diffs):.4f}")
    report.append(f"  Mean difference: {np.mean(diffs):.4f}")
    report.append(f"  Median difference: {np.median(diffs):.4f}")
    
    return "\n".join(report)


def create_overall_visualizations(data: Dict, output_prefix: str):
    """Create visualizations for overall neuron statistics."""
    if 'overall_statistics' not in data:
        print("No overall statistics found, skipping overall visualizations.")
        return
    
    overall = data['overall_statistics']
    arch = data['architecture']
    num_layers = arch['num_layers']
    
    # Get layer statistics
    layer_stats = overall['layer_statistics']
    layer_active = np.array([s['active_neurons'] for s in layer_stats])
    layer_mean_prob = np.array([s['mean_activation_prob'] for s in layer_stats])
    layer_max_prob = np.array([s['max_activation_prob'] for s in layer_stats])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Active neurons per layer
    ax1 = plt.subplot(2, 3, 1)
    ax1.bar(range(num_layers), layer_active, alpha=0.8, color='#2E86AB')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Active Neurons')
    ax1.set_title('Active Neurons per Layer (Overall)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Mean activation probability per layer
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(range(num_layers), layer_mean_prob, 'o-', linewidth=2, markersize=5, color='#2E86AB')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean Activation Probability')
    ax2.set_title('Average Neuron Activity per Layer')
    ax2.grid(alpha=0.3)
    
    # 3. Max activation probability per layer
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(range(num_layers), layer_max_prob, 's-', linewidth=2, markersize=5, color='#D62828')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Max Activation Probability')
    ax3.set_title('Peak Neuron Activity per Layer')
    ax3.grid(alpha=0.3)
    
    # 4. Distribution of active neurons
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(layer_active, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
    ax4.set_xlabel('Number of Active Neurons')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Active Neurons Across Layers')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Heatmap of activity
    ax5 = plt.subplot(2, 3, 5)
    activity_matrix = np.array([layer_active, layer_mean_prob * 10000]).T  # Scale for visibility
    im = ax5.imshow(activity_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax5.set_xlabel('Metric')
    ax5.set_ylabel('Layer')
    ax5.set_title('Activity Heatmap')
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(['Active Count', 'Mean Prob (×10k)'])
    plt.colorbar(im, ax=ax5)
    
    # 6. Top neurons visualization
    ax6 = plt.subplot(2, 3, 6)
    top_neurons = overall.get('top_neurons_by_activation', [])[:30]
    if top_neurons:
        probs = [n['activation_prob'] for n in top_neurons]
        ax6.barh(range(len(probs)), probs, alpha=0.7, color='#2E86AB')
        ax6.set_xlabel('Activation Probability')
        ax6.set_ylabel('Rank')
        ax6.set_title('Top 30 Most Active Neurons')
        ax6.invert_yaxis()
        ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overall.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_overall.png")
    plt.close()


def create_comparison_visualizations(data: Dict, output_prefix: str):
    """Create visualizations for correct vs incorrect comparison."""
    lape = data['lape_analysis']
    masks = lape['category_specific_masks']
    arch = data['architecture']
    num_layers = arch['num_layers']
    intermediate_size = arch['intermediate_size']
    
    # Get overall statistics if available
    overall_per_layer = None
    if 'overall_statistics' in data:
        layer_stats = data['overall_statistics']['layer_statistics']
        overall_per_layer = np.array([s['active_neurons'] for s in layer_stats])
    
    # Count per layer for specific neurons
    correct_per_layer = np.array([len(masks['correct'][i]) for i in range(num_layers)])
    incorrect_per_layer = np.array([len(masks['incorrect'][i]) for i in range(num_layers)])
    
    # Create main figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Layer distribution bar chart (all three categories)
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(num_layers)
    width = 0.25
    ax1.bar(x - width, correct_per_layer, width, label='Correct-specific', alpha=0.8, color='#2E86AB')
    ax1.bar(x, incorrect_per_layer, width, label='Incorrect-specific', alpha=0.8, color='#D62828')
    if overall_per_layer is not None:
        ax1.bar(x + width, overall_per_layer, width, label='Overall active', alpha=0.8, color='#06A77D')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Neurons per Layer: Specific vs Overall')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total counts comparison
    ax2 = plt.subplot(2, 3, 2)
    totals = lape['neuron_totals']
    categories = ['Correct-\nspecific', 'Incorrect-\nspecific', 'Both\nactive', 'Only\ncorrect', 'Only\nincorrect']
    values = [
        sum(len(l) for l in masks['correct']),
        sum(len(l) for l in masks['incorrect']),
        totals['both_active'],
        totals['only_correct'],
        totals['only_incorrect']
    ]
    
    # Add overall if available
    if 'overall_statistics' in data:
        categories.append('Overall\nactive')
        values.append(data['overall_statistics']['total_active_neurons'])
        colors = ['#2E86AB', '#D62828', '#06A77D', '#58A4B0', '#F18F01', '#A23B72']
    else:
        colors = ['#2E86AB', '#D62828', '#06A77D', '#58A4B0', '#F18F01']
    
    ax2.bar(range(len(categories)), values, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel('Number of Neurons')
    ax2.set_title('Neuron Categories Comparison')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax2.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=7)
    
    # 3. Percentage by layer
    ax3 = plt.subplot(2, 3, 3)
    pct_correct = 100 * correct_per_layer / intermediate_size
    pct_incorrect = 100 * incorrect_per_layer / intermediate_size
    ax3.plot(x, pct_correct, 'o-', label='Correct-specific', linewidth=2, markersize=4, color='#2E86AB')
    ax3.plot(x, pct_incorrect, 's-', label='Incorrect-specific', linewidth=2, markersize=4, color='#D62828')
    if overall_per_layer is not None:
        pct_overall = 100 * overall_per_layer / intermediate_size
        ax3.plot(x, pct_overall, '^-', label='Overall active', linewidth=2, markersize=4, color='#06A77D')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('% of Intermediate Size')
    ax3.set_title('Percentage of Neurons per Layer')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Difference by layer
    ax4 = plt.subplot(2, 3, 4)
    diff = correct_per_layer - incorrect_per_layer
    colors_diff = ['#06A77D' if d >= 0 else '#D62828' for d in diff]
    ax4.bar(x, diff, color=colors_diff, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Difference (Correct - Incorrect)')
    ax4.set_title('Neuron Count Difference per Layer')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(x, np.cumsum(correct_per_layer), 'o-', label='Correct-specific', linewidth=2, color='#2E86AB')
    ax5.plot(x, np.cumsum(incorrect_per_layer), 's-', label='Incorrect-specific', linewidth=2, color='#D62828')
    if overall_per_layer is not None:
        ax5.plot(x, np.cumsum(overall_per_layer), '^-', label='Overall active', linewidth=2, color='#06A77D')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Cumulative Count')
    ax5.set_title('Cumulative Distribution Across Layers')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Heatmap
    ax6 = plt.subplot(2, 3, 6)
    if overall_per_layer is not None:
        heatmap_data = np.array([correct_per_layer, incorrect_per_layer, overall_per_layer]).T
        labels = ['Correct', 'Incorrect', 'Overall']
    else:
        heatmap_data = np.array([correct_per_layer, incorrect_per_layer]).T
        labels = ['Correct', 'Incorrect']
    
    im = ax6.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax6.set_xlabel('Category')
    ax6.set_ylabel('Layer')
    ax6.set_title('Heatmap of Neuron Distribution')
    ax6.set_xticks(range(len(labels)))
    ax6.set_xticklabels(labels)
    plt.colorbar(im, ax=ax6, label='Number of Neurons')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_comparison.png")
    plt.close()
    
    # Create second figure for top neurons if available
    top_neurons = lape.get('top_correct_specific_by_rate_diff', [])[:30]
    if top_neurons:
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        diffs = [n['rate_diff'] for n in top_neurons]
        ax.barh(range(len(diffs)), diffs, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('Rate Difference (Correct - Incorrect)')
        ax.set_ylabel('Rank')
        ax.set_title('Top 30 Neurons by Activation Rate Difference (Correct vs Incorrect)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_top_neurons.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_prefix}_top_neurons.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LAPE benchmark analysis results (correct vs incorrect neurons)"
    )
    parser.add_argument("--json", type=str, required=True, 
                       help="Path to benchmark LAPE JSON results file")
    parser.add_argument("--output", type=str, default="benchmark_analysis",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LAPE BENCHMARK VISUALIZATION")
    print("=" * 80)
    
    # Load data
    data = load_json(args.json)
    
    # Generate reports
    report_parts = []
    report_parts.append(generate_basic_stats(data))
    report_parts.append(analyze_category_neurons(data))
    report_parts.append(analyze_layer_distribution(data))
    report_parts.append(analyze_top_neurons(data))
    
    # Combine report
    full_report = "\n".join(report_parts)
    
    # Print to console
    print(full_report)
    
    # Save report
    report_file = f"{args.output}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n{'-'*80}")
    print(f"Report saved: {report_file}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    create_overall_visualizations(data, args.output)
    create_comparison_visualizations(data, args.output)
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    print(f"Files created:")
    print(f"  - {report_file}")
    print(f"  - {args.output}_overall.png")
    print(f"  - {args.output}_comparison.png")
    print(f"  - {args.output}_top_neurons.png")


if __name__ == "__main__":
    main()

