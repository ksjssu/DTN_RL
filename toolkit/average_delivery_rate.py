#!/usr/bin/env python3
"""
Compute the mean of average_delivery_rate fields from *_EpisodeReward.txt files.

Usage:
    python toolkit/average_delivery_rate.py --dir reports_drl_eval/buf60
"""

import argparse
from pathlib import Path


def extract_delivery_rates(path: Path):
    rates = []
    for file in path.glob("*_EpisodeReward.txt"):
        try:
            for line in file.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    idx = parts.index("average_delivery_rate")
                except ValueError:
                    continue
                if idx + 1 < len(parts):
                    try:
                        rates.append(float(parts[idx + 1]))
                    except ValueError:
                        continue
        except FileNotFoundError:
            continue
    return rates


def main():
    parser = argparse.ArgumentParser(description="Average delivery success rate from EpisodeReward logs")
    parser.add_argument("--dir", required=True, help="Directory containing *_EpisodeReward.txt files")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        parser.error(f"{directory} is not a directory")

    rates = extract_delivery_rates(directory)
    if not rates:
        print(f"No average_delivery_rate values found in {directory}")
        return

    avg = sum(rates) / len(rates)
    print(f"Samples: {len(rates)}")
    print(f"Average delivery rate: {avg:.6f}")


if __name__ == "__main__":
    main()
