#!/usr/bin/env python3
"""
Aggregate DeliveryRatePerIntervalReport outputs across seeds.
Outputs CSV per scenario folder under reports/, containing cumulative averages per time bin.
"""

import argparse
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path


def parse_report(path: Path):
    bins = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            start = int(float(parts[0]))
            end = int(float(parts[1]))
            try:
                created = float(parts[2])
                delivered = float(parts[3])
                rate = float(parts[4])
            except ValueError:
                continue
            bins.append((start, end, created, delivered, rate))
    return bins


def aggregate_scenario(dir_path: Path):
    created_sum = defaultdict(float)
    delivered_sum = defaultdict(float)
    rate_sum = defaultdict(float)
    count = defaultdict(int)
    bins_order = OrderedDict()

    for report_path in sorted(dir_path.glob('*_DeliveryRatePerIntervalReport.txt')):
        report_bins = parse_report(report_path)
        for start, end, created, delivered, rate in report_bins:
            key = (start, end)
            if key not in bins_order:
                bins_order[key] = None
            created_sum[key] += created
            delivered_sum[key] += delivered
            if not (rate != rate):  # filter NaN
                rate_sum[key] += rate
                count[key] += 1

    rows = []
    for key in bins_order.keys():
        created_total = created_sum.get(key, 0.0)
        delivered_total = delivered_sum.get(key, 0.0)
        agg_rate = float('nan')
        if created_total > 0:
            agg_rate = delivered_total / created_total
        avg_rate = float('nan')
        if count.get(key, 0) > 0:
            avg_rate = rate_sum[key] / count[key]
        rows.append((key[0], key[1], created_total, delivered_total, agg_rate, avg_rate, count.get(key, 0)))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Aggregate DeliveryRatePerIntervalReport files per scenario directory.')
    parser.add_argument('--reports-root', default='reports', help='Root directory containing scenario subfolders.')
    parser.add_argument('--out-root', default='reports_summary', help='Directory to write aggregated CSV files.')
    args = parser.parse_args()

    reports_root = Path(args.reports_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scenario_dirs = sorted([p for p in reports_root.iterdir() if p.is_dir()])
    if not scenario_dirs:
        print('No scenario directories found under', reports_root)
        return

    for scenario_dir in scenario_dirs:
        rows = aggregate_scenario(scenario_dir)
        if not rows:
            continue
        out_path = out_root / f'{scenario_dir.name}_summary.csv'
        with out_path.open('w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['start', 'end', 'total_created', 'total_delivered', 'agg_success_rate', 'avg_success_rate', 'samples'])
            for row in rows:
                writer.writerow(row)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
