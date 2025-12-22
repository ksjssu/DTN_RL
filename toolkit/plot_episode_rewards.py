#!/usr/bin/env python3
"""
Plot PPO rewards and delivered counts for a given training buffer directory.

Usage:
    python toolkit/plot_episode_rewards.py --dir reports_drl_train/buf100

Requirements: matplotlib (already bundled in repo's venv requirements)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RX = re.compile(
    r"episode\s+(?P<ep>\d+)\s+reward\s+(?P<del>\d+)"
    r"(?:\s+ppo_reward\s+(?P<ppo>[-\d\.]+))?\s+avg_delivery_rate\s+(?P<adr>[-\d\.]+)"
)


def parse_rewards(path: Path):
    episodes = []
    delivered = []
    ppo_rewards = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = LINE_RX.search(line)
            if not m:
                continue
            episodes.append(int(m.group("ep")))
            delivered.append(float(m.group("del")))
            ppo_val = m.group("ppo")
            ppo_rewards.append(float(ppo_val) if ppo_val is not None else None)
    return episodes, delivered, ppo_rewards


def main():
    parser = argparse.ArgumentParser(description="Plot DRL training episode rewards")
    parser.add_argument(
        "--dir",
        default="reports_drl_train/buf100",
        help="Directory containing drl_episode_rewards.txt",
    )
    parser.add_argument(
        "--show-delivered",
        action="store_true",
        help="Overlay delivered counts in addition to PPO reward",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the figure instead of displaying it",
    )
    args = parser.parse_args()

    rewards_file = Path(args.dir) / "drl_episode_rewards.txt"
    if not rewards_file.exists():
        parser.error(f"{rewards_file} not found")

    episodes, delivered, ppo_rewards = parse_rewards(rewards_file)
    if not episodes:
        parser.error(f"No reward entries parsed from {rewards_file}")

    fig, ax = plt.subplots(figsize=(8, 4))

    if any(r is not None for r in ppo_rewards):
        ax.plot(
            episodes,
            [r if r is not None else float("nan") for r in ppo_rewards],
            marker="o",
            label="PPO reward",
        )
    else:
        ax.plot(episodes, delivered, marker="o", label="Delivered (legacy reward)")

    if args.show_delivered and any(d is not None for d in delivered):
        ax.plot(episodes, delivered, linestyle="--", alpha=0.5, label="Delivered")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"Episode rewards ({args.dir})")
    ax.legend()
    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
