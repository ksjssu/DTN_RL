#!/usr/bin/env python3
"""
Traverse a reports directory tree and compute average delivery rates for each leaf folder.

Example:
    python toolkit/average_delivery_rate_tree.py --root reports_drl_eval
"""

import argparse
from pathlib import Path
from average_delivery_rate import extract_delivery_rates


def main():
    parser = argparse.ArgumentParser(description="Average delivery rate per subdirectory")
    parser.add_argument("--root", required=True, help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    any_found = False
    for subdir in sorted(p for p in root.rglob("*") if p.is_dir()):
        rates = extract_delivery_rates(subdir)
        if not rates:
            continue
        avg = sum(rates) / len(rates)
        print(f"{subdir}: samples={len(rates)} avg={avg:.6f}")
        any_found = True

    if not any_found:
        print(f"No *_EpisodeReward.txt files with average_delivery_rate under {root}")


if __name__ == "__main__":
    main()
