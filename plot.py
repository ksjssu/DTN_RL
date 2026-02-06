#!/usr/bin/env python3
"""Plot global buffer utilization over time for runs 11-12."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_rlstate_file(filepath):
    """Parse RLStateReport file and extract time and global_avg_buf."""
    times = []
    buf_vals = []

    with open(filepath, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# time'):
                header = line[2:].split()
                continue
            if line.startswith('#'):
                continue

            parts = line.split()
            if header and len(parts) >= len(header):
                try:
                    time_idx = header.index('time')
                    buf_idx = header.index('global_avg_buf')
                    t = float(parts[time_idx])
                    buf = float(parts[buf_idx])
                    times.append(t)
                    buf_vals.append(buf)
                except (ValueError, IndexError):
                    continue

    return np.array(times), np.array(buf_vals)

def main():
    base_dir = "/mnt/c/Users/Public/workspace/git/DTN_RL/reports_drl_train/buf20"
    runs = [64, 65]
    print(f"Runs to plot: {runs}")

    plt.figure(figsize=(14, 8))

    colors = ['blue', 'red']

    for i, run in enumerate(runs):
        filepath = os.path.join(base_dir, f"drl_train_buf20_100k_rng1_run{run}_RLStateReport.txt")

        if not os.path.exists(filepath):
            print(f"Warning: Could not find file for run {run}")
            continue

        times, buf_vals = parse_rlstate_file(filepath)

        if len(times) == 0:
            print(f"Warning: No data in file for run {run}")
            continue

        # Downsample for plotting if too many points
        if len(times) > 1000:
            step = len(times) // 1000
            times = times[::step]
            buf_vals = buf_vals[::step]

        plt.plot(times / 1000, buf_vals, label=f'Run {run}', color=colors[i], alpha=0.9, linewidth=1.8)

    plt.xlabel('Time (x1000 seconds)', fontsize=12)
    plt.ylabel('Global Average Buffer Utilization', fontsize=12)
    plt.title(f'Global Buffer Utilization Over Time (buf20, runs 11 & 12)', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path = "/mnt/c/Users/Public/workspace/git/DTN_RL/buf20_runs_11_12.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
