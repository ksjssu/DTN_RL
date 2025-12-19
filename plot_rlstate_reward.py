#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_rlstate(path: Path) -> pd.DataFrame:
    columns = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("# time "):
                columns = line[2:].strip().split()
                break
    if columns is None:
        raise RuntimeError("Could not find header line beginning with '# time'.")

    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        names=columns,
        header=None,
    )
    if df.empty:
        raise RuntimeError("RLState report contained no data rows.")
    return df


def plot_reward(df: pd.DataFrame, out_path: Path, title: str) -> None:
    step_reward = df.groupby("time")["reward"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(step_reward["time"], step_reward["reward"], label="Reward per step", linewidth=1)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Reward (sum per time step)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot step-wise reward from RLState report.")
    parser.add_argument("rlstate_path", type=Path, help="Path to *_RLStateReport.txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path for PNG output (default: same directory with _reward_plot.png)",
    )
    args = parser.parse_args()

    df = load_rlstate(args.rlstate_path)
    out_path = args.output or args.rlstate_path.with_name(
        args.rlstate_path.stem + "_reward_plot.png"
    )
    title = f"Reward trajectory - {args.rlstate_path.stem}"
    plot_reward(df, out_path, title)


if __name__ == "__main__":
    main()
