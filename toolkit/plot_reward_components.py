#!/usr/bin/env python3
"""
Plot per-step reward component totals logged by DRL server.

Usage examples:
  python toolkit/plot_reward_components.py --buffer 20
  python toolkit/plot_reward_components.py --file reports/reward_logs/step_reward_components_rmappo.csv
  python toolkit/plot_reward_components.py --buffer 20 --ma 50  # moving average window
"""
import argparse
import csv
import os


def read_components_csv(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    'episode': int(float(row.get('episode', 0) or 0)),
                    'time': float(row.get('sim_time', row.get('time', 0)) or 0),
                    'pbase': float(row.get('pbase', 0) or 0),
                    'pbase_max': float(row.get('pbase_max', 0) or 0),
                    'buffer_diff': float(row.get('buffer_diff', 0) or 0),
                    'success_delta': float(row.get('success_delta', 0) or 0),
                    'success_bonus': float(row.get('success_bonus', row.get('success_pos', 0)) or 0),
                    'success_penalty': float(row.get('success_penalty', row.get('success_neg', 0)) or 0),
                    'delivery': float(row.get('delivery', 0) or 0),
                    'heuristic': float(row.get('heuristic', 0) or 0),
                    'high_load_penalty': float(row.get('high_load_penalty', 0) or 0),
                    'mid_util_penalty': float(row.get('mid_util_penalty', 0) or 0),
                    'trend_penalty': float(row.get('trend_penalty', 0) or 0),
                    'relay_bonus': float(row.get('relay_bonus', 0) or 0),
                    'target_bonus': float(row.get('target_bonus', 0) or 0),
                    'total': float(row.get('total_hosts_reward', 0) or 0),
                })
            except Exception:
                continue
    return rows


def moving_average(xs, k):
    if k <= 1:
        return xs
    out = []
    s = 0.0
    buf = []
    for x in xs:
        buf.append(x)
        s += x
        if len(buf) > k:
            s -= buf.pop(0)
        out.append(s / len(buf))
    return out


def plot_components(path, ma_window=1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}. Install it or open CSV in your tool.")
        return

    rows = read_components_csv(path)
    if not rows:
        print(f"No data: {path}")
        return
    t = [r['time'] for r in rows]
    pbase = moving_average([r['pbase'] for r in rows], ma_window)
    pbase_max = moving_average([r.get('pbase_max', 0.0) for r in rows], ma_window)
    buf = moving_average([r['buffer_diff'] for r in rows], ma_window)
    sr = moving_average([r['success_delta'] for r in rows], ma_window)
    sr_pos = moving_average([r.get('success_bonus', 0.0) for r in rows], ma_window)
    sr_neg = moving_average([r.get('success_penalty', 0.0) for r in rows], ma_window)
    tot = moving_average([r['total'] for r in rows], ma_window)

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path)

    plt.figure(figsize=(9,4))
    plt.plot(t, pbase, label='pbase(neighbor)', lw=1)
    # Plot max-based pbase if present (non-zero)
    if any(abs(v) > 0 for v in pbase_max):
        plt.plot(t, pbase_max, label='pbase_max(global)', lw=1, ls='-.')
    plt.plot(t, buf, label='buffer_diff', lw=1)
    plt.plot(t, sr, label='success_delta(sum)', lw=1)
    # If detailed columns exist, plot them separately
    if any(abs(v) > 0 for v in sr_pos):
        plt.plot(t, sr_pos, label='success_bonus', lw=1, ls='--')
    if any(abs(v) > 0 for v in sr_neg):
        plt.plot(t, sr_neg, label='success_penalty', lw=1, ls=':')
    plt.xlabel('sim time (s)')
    plt.ylabel('reward per step (host-level total)')
    plt.title(f'Reward Components ({base})')
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{base}_components.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Optional: total vs sum of selected components sanity check
    try:
        ssum = [pbase[i] + buf[i] + sr[i] for i in range(len(tot))]
        plt.figure(figsize=(9,4))
        plt.plot(t, tot, label='total_hosts_reward', lw=1)
        plt.plot(t, ssum, label='selected_sum(pbase+buffer+sr)', lw=1)
        plt.xlabel('sim time (s)')
        plt.ylabel('reward per step')
        plt.title(f'Total vs Selected Components ({base})')
        plt.legend()
        plt.tight_layout()
        out_png2 = os.path.join(out_dir, f"{base}_total_vs_selected.png")
        plt.savefig(out_png2, dpi=150)
        plt.close()
        print(f"Wrote {out_png} and {out_png2}")
    except Exception:
        print(f"Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buffer', type=int, default=None, help='buffer size MB')
    ap.add_argument('--file', type=str, default=None, help='path to components CSV')
    ap.add_argument('--ma', type=int, default=1, help='moving average window')
    args = ap.parse_args()

    if args.file:
        path = args.file
    else:
        if args.buffer:
            path = os.path.join('reports','reward_logs', f'step_reward_components_buf{int(args.buffer)}M_rmappo.csv')
        else:
            path = os.path.join('reports','reward_logs', 'step_reward_components_rmappo.csv')
    plot_components(path, ma_window=max(1, int(args.ma)))


if __name__ == '__main__':
    main()
