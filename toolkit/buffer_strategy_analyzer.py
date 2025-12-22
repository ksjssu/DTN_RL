#!/usr/bin/env python3
"""
Buffer Strategy Analyzer
Analyzes optimal strategies for different buffer sizes in DTN networks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_buffer_strategies():
    """Analyze different buffer size performance patterns"""

    # Load episode rewards for different buffer sizes
    buffer_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    results = {}

    for buf_size in buffer_sizes:
        file_path = f"reports_drl_train/buf{buf_size:02d}/drl_episode_rewards.txt"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                delivery_rates = []
                for line in lines:
                    if 'avg_delivery_rate' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'avg_delivery_rate':
                                rate = float(parts[i+1])
                                delivery_rates.append(rate)
                                break

                if delivery_rates:
                    results[buf_size] = {
                        'mean': np.mean(delivery_rates),
                        'std': np.std(delivery_rates),
                        'max': np.max(delivery_rates),
                        'min': np.min(delivery_rates),
                        'episodes': len(delivery_rates),
                        'rates': delivery_rates
                    }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return results

def plot_buffer_performance(results):
    """Plot buffer size vs performance"""

    if not results:
        print("No data available for plotting")
        return

    # Extract data for plotting
    buffer_sizes = sorted(results.keys())
    means = [results[buf]['mean'] for buf in buffer_sizes]
    stds = [results[buf]['std'] for buf in buffer_sizes]
    maxs = [results[buf]['max'] for buf in buffer_sizes]
    mins = [results[buf]['min'] for buf in buffer_sizes]

    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Mean performance with error bars
    ax1.errorbar(buffer_sizes, means, yerr=stds, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Buffer Size (MB)')
    ax1.set_ylabel('Average Delivery Rate')
    ax1.set_title('Average Performance by Buffer Size')
    ax1.grid(True, alpha=0.3)

    # Highlight problematic middle range
    ax1.axvspan(10, 25, alpha=0.2, color='red', label='Problematic Zone')
    ax1.legend()

    # 2. Performance stability (coefficient of variation)
    cv = [std/mean if mean > 0 else 0 for mean, std in zip(means, stds)]
    ax2.bar(buffer_sizes, cv, alpha=0.7, color='orange')
    ax2.set_xlabel('Buffer Size (MB)')
    ax2.set_ylabel('Coefficient of Variation (std/mean)')
    ax2.set_title('Performance Stability')
    ax2.grid(True, alpha=0.3)

    # 3. Performance range (max - min)
    ranges = [max_val - min_val for max_val, min_val in zip(maxs, mins)]
    ax3.bar(buffer_sizes, ranges, alpha=0.7, color='purple')
    ax3.set_xlabel('Buffer Size (MB)')
    ax3.set_ylabel('Performance Range (Max - Min)')
    ax3.set_title('Performance Variability')
    ax3.grid(True, alpha=0.3)

    # 4. Learning progress for problematic buffers
    for buf_size in [10, 15, 20]:
        if buf_size in results and len(results[buf_size]['rates']) > 20:
            rates = results[buf_size]['rates'][-30:]  # Last 30 episodes
            ax4.plot(range(len(rates)), rates, label=f'{buf_size}M', marker='o', markersize=3)

    ax4.set_xlabel('Episode (Last 30)')
    ax4.set_ylabel('Delivery Rate')
    ax4.set_title('Recent Learning Progress (Problematic Buffers)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/buffer_strategy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results

def identify_optimal_strategies(results):
    """Identify what makes certain buffer sizes successful"""

    print("=== BUFFER STRATEGY ANALYSIS ===\n")

    if not results:
        print("No data available for analysis")
        return

    # Sort by performance
    sorted_buffers = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)

    print("PERFORMANCE RANKING:")
    for i, (buf_size, data) in enumerate(sorted_buffers[:5]):
        print(f"{i+1}. {buf_size}M: {data['mean']:.3f} ± {data['std']:.3f}")

    print(f"\nWORST PERFORMERS:")
    for i, (buf_size, data) in enumerate(sorted_buffers[-3:]):
        print(f"{buf_size}M: {data['mean']:.3f} ± {data['std']:.3f}")

    # Identify problematic zone
    middle_buffers = [10, 15, 20, 25]
    middle_performance = [results[buf]['mean'] for buf in middle_buffers if buf in results]

    if middle_performance:
        avg_middle = np.mean(middle_performance)
        print(f"\nMIDDLE BUFFER ZONE (10M-25M):")
        print(f"   Average performance: {avg_middle:.3f}")

        # Compare with extremes
        small_buffers = [5]
        large_buffers = [40, 45, 50]

        small_perf = [results[buf]['mean'] for buf in small_buffers if buf in results]
        large_perf = [results[buf]['mean'] for buf in large_buffers if buf in results]

        if small_perf:
            print(f"   Small buffers (5M): {np.mean(small_perf):.3f}")
        if large_perf:
            print(f"   Large buffers (40M+): {np.mean(large_perf):.3f}")

    # Strategy recommendations
    print(f"\nSTRATEGY RECOMMENDATIONS:")
    print(f"   Small buffers (5M-10M): Selective forwarding, minimal relay")
    print(f"   Medium buffers (15M-25M): Hybrid strategy needed")
    print(f"   Large buffers (30M+): Hub role, active relay")

def main():
    """Main analysis function"""
    print("DTN Buffer Strategy Analyzer")
    print("=" * 40)

    # Analyze buffer strategies
    results = analyze_buffer_strategies()

    if results:
        # Plot performance
        plot_buffer_performance(results)

        # Identify strategies
        identify_optimal_strategies(results)

        print(f"\n✅ Analysis complete! Check reports/buffer_strategy_analysis.png")
    else:
        print("❌ No data found. Run some training first!")

if __name__ == "__main__":
    main()