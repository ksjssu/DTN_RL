#!/usr/bin/env python3
"""
Plot cumulative delivery success rates per buffer size.
Generates PNG files in reports_plots/, with one plot per buffer containing curves for
Prophet (delta=0) and heuristics delta 0.1 / 0.2 / 0.3.
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CONFIGS = {
    "prophet": ("Prophet", "#1f77b4"),
    "heuristic_d010": ("Delta 0.1", "#ff7f0e"),
    "heuristic_d020": ("Delta 0.2", "#2ca02c"),
    "heuristic_d030": ("Delta 0.3", "#d62728"),
}


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "agg_success_rate" in df.columns:
        df = df.rename(columns={"agg_success_rate": "agg_rate"})
    return df


def plot_buffer(buffer_label: str, args):
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for prefix, (label, color) in CONFIGS.items():
        summary_name = f"{prefix}_buf{buffer_label}_summary.csv"
        path = Path(args.summary_root) / summary_name
        if not path.exists():
            print(f"Missing summary: {path}")
            continue
        df = load_summary(path)
        if df.empty:
            continue
        plt.plot(df["end"], df["agg_rate"], label=label, color=color, marker="o", markersize=4, linewidth=1.5)

    plt.title(f"Cumulative Delivery Success Rate (Buffer {int(buffer_label)}M)")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"delivery_rate_buf{buffer_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot delivery success rates by buffer size")
    parser.add_argument("--summary-root", default="reports_summary", help="Directory containing *_summary.csv")
    parser.add_argument("--out-root", default="reports_plots", help="Directory for output PNG plots")
    args = parser.parse_args()

    for buf in range(5, 51, 5):
        plot_buffer(f"{buf:02d}", args)


if __name__ == "__main__":
    main()

