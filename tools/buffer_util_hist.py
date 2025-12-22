#!/usr/bin/env python3
"""Summarize buffer utilization distribution from RLStateReport by time windows."""
import argparse
from collections import defaultdict

BUCKETS = [(i / 20.0, (i + 1) / 20.0) for i in range(20)]
BUCKETS.append((1.0, 1.0))  # exactly 100%


def bucket_index(value: float) -> int:
    if value >= 1.0:
        return len(BUCKETS) - 1
    return min(int(value * 20), len(BUCKETS) - 2)


def summarize(path: str, window: int):
    windows = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                t = float(parts[0])
                buf_util = float(parts[9])
            except ValueError:
                continue
            w = int(t // window)
            idx = bucket_index(max(0.0, min(1.0, buf_util)))
            windows[w][idx] += 1
            totals[w] += 1

    bucket_labels = [f"{int(low*100):02d}-{int(high*100):02d}%" if i < len(BUCKETS) - 1 else "100%"
                     for i, (low, high) in enumerate(BUCKETS)]

    for w in sorted(windows.keys()):
        start = w * window
        end = start + window
        total = totals[w]
        print(f"Window {start}-{end}s (samples={total})")
        for idx, label in enumerate(bucket_labels):
            cnt = windows[w].get(idx, 0)
            pct = (cnt / total * 100.0) if total else 0.0
            print(f"  {label:>8}: {cnt:6d} ({pct:5.2f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to RLStateReport.txt")
    parser.add_argument("--window", type=int, default=3600, help="Window size in seconds")
    args = parser.parse_args()
    summarize(args.file, args.window)

if __name__ == "__main__":
    main()
