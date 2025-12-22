#!/usr/bin/env python3
"""
Aggregate delivery_prob from MessageStatsReport across rng seeds,
grouped by (bufferSize, tDup, delta). Expects scenario name format:
  grid_buf{buf}_t{tDup}_d{delta}_rng{seed}
and files under reports/:
  reports/{scenario}_MessageStatsReport.txt

Outputs CSV to stdout and writes summary to reports/grid_summary.csv
"""
import os
import re
import glob
import statistics as stats

REPORTS_DIR = os.path.join(os.getcwd(), 'reports')
PATTERN = re.compile(r'^grid_buf(?P<buf>[^_]+)_t(?P<tdup>\d+)_d(?P<delta>[0-9.]+)_rng(?P<rng>\d+)$')

def parse_scenario_name(name: str):
    m = PATTERN.match(name)
    if not m:
        return None
    buf = m.group('buf')
    tdup = int(m.group('tdup'))
    delta = float(m.group('delta'))
    rng = int(m.group('rng'))
    return buf, tdup, delta, rng

def read_delivery_prob(path: str):
    delivery = None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('delivery_prob:'):
                    try:
                        delivery = float(line.split(':', 1)[1].strip())
                    except Exception:
                        pass
        return delivery
    except Exception:
        return None

def main():
    # Recursively collect all *_MessageStatsReport.txt under reports/
    files = []
    for root, _, fnames in os.walk(REPORTS_DIR):
        for fn in fnames:
            if fn.endswith('_MessageStatsReport.txt'):
                files.append(os.path.join(root, fn))
    groups = {}  # (buf,tdup,delta) -> {rng: delivery_prob}
    for fp in files:
        base = os.path.basename(fp)
        scenario = base.replace('_MessageStatsReport.txt', '')
        parsed = parse_scenario_name(scenario)
        if not parsed:
            continue
        buf, tdup, delta, rng = parsed
        val = read_delivery_prob(fp)
        if val is None:
            continue
        key = (buf, tdup, delta)
        if key not in groups:
            groups[key] = {}
        groups[key][rng] = val

    rows = []
    for (buf, tdup, delta), seeds in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
        vals = list(seeds.values())
        n = len(vals)
        avg = stats.mean(vals) if n else float('nan')
        sd = stats.pstdev(vals) if n > 1 else 0.0
        rows.append((buf, tdup, delta, n, avg, sd))

    # Output CSV
    out_lines = ["bufferSize,tDup,delta,n,avg_delivery_prob,stddev"]
    for buf, tdup, delta, n, avg, sd in rows:
        out_lines.append(f"{buf},{tdup},{delta},{n},{avg:.6f},{sd:.6f}")
    csv_text = "\n".join(out_lines)
    print(csv_text)

    # Write to file
    out_path = os.path.join(REPORTS_DIR, 'grid_summary.csv')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(csv_text + "\n")
        print(f"\nSaved summary: {out_path}")
    except Exception as e:
        print(f"Failed to write summary: {e}")

if __name__ == '__main__':
    main()
