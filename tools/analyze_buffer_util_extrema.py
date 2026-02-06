#!/usr/bin/env python3
"""
Analyze buffer utilization extrema and slopes for run1
"""

import sys
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

    return np.array(time_points), np.array(avg_values)

def find_extrema(time_points, values, window=3):
    """Find local maxima and minima using a sliding window"""
    maxima = []
    minima = []

    for i in range(window, len(values) - window):
        # Check if it's a local maximum
        is_max = all(values[i] >= values[j] for j in range(i-window, i+window+1) if j != i)
        if is_max and values[i] > values[i-1] and values[i] > values[i+1]:
            maxima.append((time_points[i], values[i], i))

        # Check if it's a local minimum
        is_min = all(values[i] <= values[j] for j in range(i-window, i+window+1) if j != i)
        if is_min and values[i] < values[i-1] and values[i] < values[i+1]:
            minima.append((time_points[i], values[i], i))

    return maxima, minima

def calculate_slopes(time_points, values):
    """Calculate slopes between consecutive points"""
    slopes = []
    for i in range(len(values) - 1):
        dt = time_points[i+1] - time_points[i]
        dv = values[i+1] - values[i]
        if dt > 0:
            slope = dv / dt
            slopes.append((time_points[i], time_points[i+1], slope))

    return slopes

def main():
    file1 = '/mnt/c/dtn/DTN_RL/reports_drl_train/buf20/drl_train_buf20_100k_rng1_run1_RLStateReport.txt'

    print("=== Run1 Buffer Utilization Analysis ===\n")
    print("Parsing run1...")
    times1, buf_utils1 = parse_rlstate_file(file1)
    print(f"  Found {len(times1)} data points")

    print("\nAggregating data by 200s intervals...")
    interval = 200
    max_time = 57600

    time_points1, avg_buf1 = aggregate_by_interval(times1, buf_utils1, interval, max_time)
    print(f"  Run1: {len(time_points1)} intervals")

    # Basic statistics
    print(f"\n=== Basic Statistics ===")
    print(f"Overall Mean: {np.mean(avg_buf1):.6f}")
    print(f"Overall Std:  {np.std(avg_buf1):.6f}")
    print(f"Global Max:   {np.max(avg_buf1):.6f} at t={time_points1[np.argmax(avg_buf1)]:.0f}s")
    print(f"Global Min:   {np.min(avg_buf1):.6f} at t={time_points1[np.argmin(avg_buf1)]:.0f}s")

    # Find local extrema
    print(f"\n=== Local Extrema (window=3) ===")
    maxima, minima = find_extrema(time_points1, avg_buf1, window=3)

    print(f"\nLocal Maxima (count={len(maxima)}):")
    for i, (t, v, idx) in enumerate(maxima[:10], 1):  # Show top 10
        print(f"  {i}. t={t:7.0f}s, value={v:.6f}, index={idx}")
    if len(maxima) > 10:
        print(f"  ... and {len(maxima)-10} more")

    print(f"\nLocal Minima (count={len(minima)}):")
    for i, (t, v, idx) in enumerate(minima[:10], 1):  # Show top 10
        print(f"  {i}. t={t:7.0f}s, value={v:.6f}, index={idx}")
    if len(minima) > 10:
        print(f"  ... and {len(minima)-10} more")

    # Calculate slopes
    print(f"\n=== Slope Analysis ===")
    slopes = calculate_slopes(time_points1, avg_buf1)

    slope_values = [s[2] for s in slopes]
    print(f"Total segments: {len(slopes)}")
    print(f"Mean slope: {np.mean(slope_values):.9f}")
    print(f"Std slope:  {np.std(slope_values):.9f}")

    # Find steepest slopes
    slopes_sorted = sorted(slopes, key=lambda x: abs(x[2]), reverse=True)

    print(f"\nSteepest positive slopes (increasing):")
    positive_slopes = [s for s in slopes_sorted if s[2] > 0][:10]
    for i, (t1, t2, slope) in enumerate(positive_slopes, 1):
        print(f"  {i}. [{t1:7.0f}s → {t2:7.0f}s]: slope={slope:.9f} ({slope*100:.6f}%/s)")

    print(f"\nSteepest negative slopes (decreasing):")
    negative_slopes = [s for s in slopes_sorted if s[2] < 0][:10]
    for i, (t1, t2, slope) in enumerate(negative_slopes, 1):
        print(f"  {i}. [{t1:7.0f}s → {t2:7.0f}s]: slope={slope:.9f} ({slope*100:.6f}%/s)")

    # Identify major trend changes
    print(f"\n=== Major Trend Changes ===")
    major_changes = []
    for i in range(1, len(slopes)):
        prev_slope = slopes[i-1][2]
        curr_slope = slopes[i][2]
        if abs(curr_slope - prev_slope) > 0.0001:  # Significant change threshold
            major_changes.append((slopes[i][0], prev_slope, curr_slope, curr_slope - prev_slope))

    major_changes_sorted = sorted(major_changes, key=lambda x: abs(x[3]), reverse=True)[:10]
    print(f"Top 10 slope changes:")
    for i, (t, prev_s, curr_s, delta) in enumerate(major_changes_sorted, 1):
        print(f"  {i}. t={t:7.0f}s: {prev_s:.9f} → {curr_s:.9f} (Δ={delta:.9f})")

    # Phase analysis
    print(f"\n=== Phase Analysis ===")
    phases = [
        ("Early (0-10000s)", 0, 10000),
        ("Mid-Early (10000-20000s)", 10000, 20000),
        ("Mid (20000-30000s)", 20000, 30000),
        ("Crisis (30000-35000s)", 30000, 35000),
        ("Recovery (35000-45000s)", 35000, 45000),
        ("Late (45000-57600s)", 45000, 57600)
    ]

    for phase_name, t_start, t_end in phases:
        mask = (time_points1 >= t_start) & (time_points1 < t_end)
        phase_values = avg_buf1[mask]
        phase_times = time_points1[mask]

        if len(phase_values) > 0:
            # Calculate linear regression slope for the phase
            if len(phase_values) > 1:
                coeffs = np.polyfit(phase_times, phase_values, 1)
                phase_slope = coeffs[0]
                print(f"{phase_name:25s}: mean={np.mean(phase_values):.6f}, slope={phase_slope:.9f}")
            else:
                print(f"{phase_name:25s}: mean={np.mean(phase_values):.6f}, slope=N/A")

if __name__ == '__main__':
    main()
