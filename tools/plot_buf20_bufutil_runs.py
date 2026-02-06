#!/usr/bin/env python3
"""
Plot global average buffer utilization over time for multiple runs
of drl_train_buf20_100k_rng1 experiment.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_rlstate_file(filepath):
    """Parse RLStateReport file and extract time vs global_avg_buf data."""
    time_data = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 19:
                continue

            try:
                time = int(parts[0])
                # global_avg_buf is at index 18 (0-based)
                global_avg_buf = float(parts[18])
                time_data[time].append(global_avg_buf)
            except (ValueError, IndexError):
                continue

    # Average values for each time point (multiple hosts per time)
    times = sorted(time_data.keys())
    avg_buf_utils = [np.mean(time_data[t]) for t in times]

    return times, avg_buf_utils

def main():
    # Support both WSL and Windows paths
    import platform
    if platform.system() == 'Windows':
        base_dir = r"C:\dtn\DTN_RL\reports_drl_train\buf20"
    else:
        base_dir = "/mnt/c/dtn/DTN_RL/reports_drl_train/buf20"
    runs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    plt.figure(figsize=(14, 8))

    # Color map for different runs
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))

    for idx, run in enumerate(runs):
        filename = f"drl_train_buf20_100k_rng1_run{run}_RLStateReport.txt"
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        print(f"Processing run {run}...")
        times, avg_buf_utils = parse_rlstate_file(filepath)

        if times:
            # Convert time to hours for better readability
            times_hours = [t / 3600 for t in times]
            plt.plot(times_hours, avg_buf_utils, label=f'Run {run}',
                    color=colors[idx], linewidth=1.5, alpha=0.8)

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Global Average Buffer Utilization', fontsize=12)
    plt.title('Buffer Utilization Over Time - DRL Train buf20 100k rng1\n(Runs 1, 5, 10, 15, 20, 25, 30, 35, 40, 45)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1)

    # Save the plot
    output_path = os.path.join(base_dir, "buf20_rng1_bufutil_runs_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    # Also save as SVG
    output_svg = os.path.join(base_dir, "buf20_rng1_bufutil_runs_comparison.svg")
    plt.savefig(output_svg, format='svg')
    print(f"SVG saved to: {output_svg}")

    plt.show()

if __name__ == "__main__":
    main()
