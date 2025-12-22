#!/usr/bin/env python3
"""
Compute average delivery success rates for heuristic configs over rng ranges.

Example:
    python toolkit/average_delivery_rate_configs.py --root reports \
        --configs heuristic_d010 heuristic_d020 heuristic_d030 \
        --buffers 60 70 80 90 100 --rng-start 1 --rng-end 50
"""

import argparse
import re
from pathlib import Path
from statistics import mean

RNG_RE = re.compile(r"rng(\d+)")


def extract_rng(path: Path):
    match = RNG_RE.search(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def final_delivery_rate(report_path: Path):
    last_value = None
    with report_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                last_value = float(parts[-1])
            except (ValueError, IndexError):
                continue
    return last_value


def main():
    parser = argparse.ArgumentParser(description="Average delivery rates for heuristic configs")
    parser.add_argument("--root", default="reports", help="Root directory containing heuristic report folders")
    parser.add_argument("--configs", nargs="+", default=["heuristic_d010", "heuristic_d020", "heuristic_d030"])
    parser.add_argument("--buffers", nargs="+", type=int, default=[60, 70, 80, 90, 100])
    parser.add_argument("--rng-start", type=int, default=1, help="Inclusive RNG start")
    parser.add_argument("--rng-end", type=int, default=50, help="Inclusive RNG end")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        parser.error(f"{root} does not exist")

    for cfg in args.configs:
        for buf in args.buffers:
            dir_path = root / f"{cfg}_buf{buf}"
            if not dir_path.is_dir():
                print(f"{cfg} buf{buf}: directory missing ({dir_path})")
                continue
            rates = []
            for report in sorted(dir_path.glob("*_DeliveryRatePerIntervalReport.txt")):
                rng = extract_rng(report)
                if rng is None or rng < args.rng_start or rng > args.rng_end:
                    continue
                value = final_delivery_rate(report)
                if value is not None:
                    rates.append(value)
            if not rates:
                print(f"{cfg} buf{buf}: samples=0 avg=N/A")
            else:
                print(f"{cfg} buf{buf}: samples={len(rates)} avg={mean(rates):.6f}")


if __name__ == "__main__":
    main()
