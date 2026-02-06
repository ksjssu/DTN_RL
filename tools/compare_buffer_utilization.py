#!/usr/bin/env python3
"""
Compare buffer utilization between two runs
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_rlstate_file(filepath):
    """Parse RLStateReport file and extract time and global_avg_buf"""
    times = []
    buf_utils = []

    with open(filepath, 'r') as f:
        # Find header line
        header = None
        for line in f:
            if line.startswith('# time type'):
                header = line[2:].strip().split()
                break

        if header is None:
            print(f"Error: No header found in {filepath}")
            return times, buf_utils

        # Find global_avg_buf column index
        try:
            buf_idx = header.index('global_avg_buf')
        except ValueError:
            print(f"Error: global_avg_buf column not found in {filepath}")
            return times, buf_utils

        # Parse data lines
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split()
            if len(parts) <= buf_idx:
                continue

            try:
                time = float(parts[0])
                buf_util = float(parts[buf_idx])
                times.append(time)
                buf_utils.append(buf_util)
            except (ValueError, IndexError):
                continue

    return times, buf_utils

def aggregate_by_interval(times, values, interval=200, max_time=57600):
    """Aggregate values by time interval"""
    bins = defaultdict(list)

    for t, v in zip(times, values):
        if t > max_time:
            break
        bin_idx = int(t // interval)
        bins[bin_idx].append(v)

    # Calculate averages
    time_points = []
    avg_values = []

    for bin_idx in sorted(bins.keys()):
        time_point = (bin_idx + 0.5) * interval  # Middle of interval
        avg_value = np.mean(bins[bin_idx])
        time_points.append(time_point)
        avg_values.append(avg_value)

    return time_points, avg_values

def main():
    file1 = '/mnt/c/dtn/DTN_RL/reports_drl_train/buf20/drl_train_buf20_100k_rng1_run1_RLStateReport.txt'
    file2 = '/mnt/c/dtn/DTN_RL/reports_drl_train/buf20/drl_train_buf20_100k_rng1_run44_RLStateReport.txt'
    file3 = '/mnt/c/dtn/DTN_RL/reports_drl_train/buf20/drl_train_buf20_100k_rng1_run75_RLStateReport.txt'
    file4 = '/mnt/c/dtn/DTN_RL/reports_drl_train/buf20/drl_train_buf20_100k_rng1_run95_RLStateReport.txt'

    print("Parsing run1...")
    times1, buf_utils1 = parse_rlstate_file(file1)
    print(f"  Found {len(times1)} data points")

    print("Parsing run44...")
    times2, buf_utils2 = parse_rlstate_file(file2)
    print(f"  Found {len(times2)} data points")

    print("Parsing run75...")
    times3, buf_utils3 = parse_rlstate_file(file3)
    print(f"  Found {len(times3)} data points")

    print("Parsing run95...")
    times4, buf_utils4 = parse_rlstate_file(file4)
    print(f"  Found {len(times4)} data points")

    print("\nAggregating data by 200s intervals...")
    interval = 200
    max_time = 57600

    time_points1, avg_buf1 = aggregate_by_interval(times1, buf_utils1, interval, max_time)
    time_points2, avg_buf2 = aggregate_by_interval(times2, buf_utils2, interval, max_time)
    time_points3, avg_buf3 = aggregate_by_interval(times3, buf_utils3, interval, max_time)
    time_points4, avg_buf4 = aggregate_by_interval(times4, buf_utils4, interval, max_time)

    print(f"  Run1: {len(time_points1)} intervals")
    print(f"  Run44: {len(time_points2)} intervals")
    print(f"  Run75: {len(time_points3)} intervals")
    print(f"  Run95: {len(time_points4)} intervals")

    # Plot
    plt.figure(figsize=(14, 7))

    plt.plot(time_points1, avg_buf1, label='Run 1', linewidth=1.5, alpha=0.8)
    plt.plot(time_points2, avg_buf2, label='Run 44', linewidth=1.5, alpha=0.8)
    plt.plot(time_points3, avg_buf3, label='Run 75', linewidth=1.5, alpha=0.8)
    plt.plot(time_points4, avg_buf4, label='Run 95', linewidth=1.5, alpha=0.8)

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Average Buffer Utilization', fontsize=12)
    plt.title('Buffer Utilization Comparison: Run1 vs Run44 vs Run75 vs Run95 (200s intervals)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Set x-axis limits
    plt.xlim(0, max_time)

    # Add some statistics
    if len(avg_buf1) > 0 and len(avg_buf2) > 0 and len(avg_buf3) > 0 and len(avg_buf4) > 0:
        mean1 = np.mean(avg_buf1)
        mean2 = np.mean(avg_buf2)
        mean3 = np.mean(avg_buf3)
        mean4 = np.mean(avg_buf4)
        std1 = np.std(avg_buf1)
        std2 = np.std(avg_buf2)
        std3 = np.std(avg_buf3)
        std4 = np.std(avg_buf4)

        stats_text = f"Run1: mean={mean1:.4f}, std={std1:.4f}\n"
        stats_text += f"Run44: mean={mean2:.4f}, std={std2:.4f}\n"
        stats_text += f"Run75: mean={mean3:.4f}, std={std3:.4f}\n"
        stats_text += f"Run95: mean={mean4:.4f}, std={std4:.4f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save figure
    output_file = '/mnt/c/dtn/DTN_RL/reports_plots/buffer_util_run1_vs_run44_vs_run75_vs_run95.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")

    plt.show()

if __name__ == '__main__':
    main()
