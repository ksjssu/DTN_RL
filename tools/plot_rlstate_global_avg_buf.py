#!/usr/bin/env python3
"""
Plot global_avg_buf over time for RLStateReport runs.
Example:
  .venv/bin/python tools/plot_rlstate_global_avg_buf.py --runs 1-42,59-71 --rainbow \
    --out reports_plots/rlstate_global_avg_buf_runs1_42_59_71.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def parse_runs(spec):
    runs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            runs.extend(range(start, end + 1))
        else:
            runs.append(int(part))
    return sorted(set(runs))


def read_series(path, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return read_series_from_tsv(cache_path)

    times = []
    values = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 19:
                continue
            t = parts[0]
            if t in seen:
                continue
            seen.add(t)
            try:
                times.append(float(t))
                values.append(float(parts[18]))
            except ValueError:
                continue

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as out:
            for t, v in zip(times, values):
                out.write(f"{t}\t{v}\n")

    return times, values


def read_series_from_tsv(path):
    times = []
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            try:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
            except ValueError:
                continue
    return times, values


def main():
    parser = argparse.ArgumentParser(description="Plot global_avg_buf from RLStateReport runs.")
    parser.add_argument(
        "--runs",
        required=True,
        help="Run list, e.g. '1-42,59-71,99,100'",
    )
    parser.add_argument(
        "--base-dir",
        default="reports_drl_train/buf20",
        help="Directory containing rlstate_run*.txt files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="reports_plots/rlstate_tmp",
        help="Directory for cached TSV files (optional).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--rainbow",
        action="store_true",
        help="Use rainbow color order across runs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional horizontal threshold line (e.g., 0.8).",
    )
    args = parser.parse_args()

    run_ids = parse_runs(args.runs)
    if not run_ids:
        print("No runs specified.", file=sys.stderr)
        return 2

    series = {}
    for run_id in run_ids:
        path = os.path.join(args.base_dir, f"rlstate_run{run_id}.txt")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        cache_path = None
        if args.cache_dir:
            cache_path = os.path.join(
                args.cache_dir, f"rlstate_run{run_id}_global_avg_buf.tsv"
            )
        times, values = read_series(path, cache_path=cache_path)
        if times:
            series[run_id] = (times, values)

    if not series:
        print("No data found.", file=sys.stderr)
        return 2

    plt.figure(figsize=(12, 7))
    colors = None
    if args.rainbow:
        colors = cm.rainbow([i / max(1, len(run_ids) - 1) for i in range(len(run_ids))])

    for idx, run_id in enumerate(run_ids):
        if run_id not in series:
            continue
        times, values = series[run_id]
        color = colors[idx] if colors is not None else None
        plt.plot(
            times,
            values,
            linewidth=1.2,
            alpha=0.85,
            color=color,
            label=f"run{run_id}",
        )

    if args.threshold is not None:
        plt.axhline(
            args.threshold,
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=f"threshold {args.threshold}",
        )

    plt.title("Global Avg Buffer Utilization Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Global Avg Buffer Utilization")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=6, fontsize=7)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
