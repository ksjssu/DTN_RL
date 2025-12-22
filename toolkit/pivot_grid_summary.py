#!/usr/bin/env python3
"""
Read reports/grid_summary.csv and emit pivot tables of avg_delivery_prob
for each bufferSize. Rows=tDup, Columns=delta.
Outputs to reports/pivot_{buffer}.csv
"""
import csv
import os
from collections import defaultdict

REPORTS_DIR = os.path.join(os.getcwd(), 'reports')
SUMMARY = os.path.join(REPORTS_DIR, 'grid_summary.csv')

def read_summary(path):
    data = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                buf = row['bufferSize']
                tdup = int(row['tDup'])
                delta = float(row['delta'])
                avg = float(row['avg_delivery_prob'])
            except Exception:
                continue
            data.append((buf, tdup, delta, avg))
    return data

def pivot(data):
    # buf -> tdup -> delta -> avg
    table = defaultdict(lambda: defaultdict(dict))
    for buf, tdup, delta, avg in data:
        table[buf][tdup][delta] = avg
    return table

def write_pivots(table):
    for buf, rows in table.items():
        tdups = sorted(rows.keys())
        deltas = sorted({d for m in rows.values() for d in m.keys()})
        out_path = os.path.join(REPORTS_DIR, f'pivot_{buf}.csv')
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ['tDup'] + [str(d) for d in deltas]
            w.writerow(header)
            for t in tdups:
                line = [t]
                for d in deltas:
                    val = rows[t].get(d, '')
                    if val == '':
                        line.append('')
                    else:
                        line.append(f'{val:.6f}')
                w.writerow(line)
        print(f'Wrote {out_path}')

def main():
    if not os.path.exists(SUMMARY):
        print(f'Not found: {SUMMARY}\nRun aggregate_grid_results.py first.')
        return
    data = read_summary(SUMMARY)
    tbl = pivot(data)
    write_pivots(tbl)

if __name__ == '__main__':
    main()

