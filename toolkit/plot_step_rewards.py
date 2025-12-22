#!/usr/bin/env python3
import csv
import os
import sys
import math

def read_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                # Support both step_rewards_bufXXM.csv (sim_time/total_reward) and drl_step_rewards.csv (reward_sum)
                reward_val = row.get('reward_sum')
                if reward_val is None:
                    reward_val = row.get('total_reward')
                delivered_val = row.get('delivered')
                if delivered_val is None:
                    delivered_val = row.get('delivered_per_step')
                rows.append({
                    'time': int(float(row.get('time', row.get('sim_time', 0)) or 0)),
                    'episode': int(float(row.get('episode', 0) or 0)),
                    'sim_id': row.get('sim_id', ''),
                    'buffer': int(float(row.get('buffer', 0) or 0)),
                    'states': int(float(row.get('states', 0) or 0)),
                    'actions': int(float(row.get('actions', 0) or 0)),
                    'reward_sum': float(reward_val or 0),
                    'delivered': float(delivered_val or 0),
                    'relayed': float(row.get('relayed', 0) or 0),
                    'drops': float(row.get('drops', 0) or 0),
                    'aborted': float(row.get('aborted', 0) or 0),
                })
            except Exception:
                continue
    return rows

def plot_file(path, episode_seconds=None):
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless env: render to file
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}. Install it or open CSV in your tool.")
        return
    rows = read_csv(path)
    if not rows:
        print(f"No data in {path}")
        return
    # derive step index per file
    step_idx = list(range(1, len(rows)+1))
    rew = [r['reward_sum'] for r in rows]
    rew_cum = []
    acc = 0.0
    for v in rew:
        acc += v
        rew_cum.append(acc)
    t = [r['time'] for r in rows]
    delivered = [r.get('delivered', 0) for r in rows]
    drops = [r.get('drops', 0) for r in rows]
    buf = rows[-1].get('buffer', 0)
    episodes = [r.get('episode', 0) for r in rows]
    has_episode = any(e > 0 for e in episodes)

    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.dirname(path)

    # Plot reward per step (by step index)
    plt.figure(figsize=(8,4))
    plt.plot(step_idx, rew, lw=1)
    plt.xlabel('step')
    plt.ylabel('reward (per step)')
    title = f"Step Reward ({base})" + (f" buf={buf}M" if buf else "")
    plt.title(title)
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"{base}_by_step.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot cumulative reward vs step
    plt.figure(figsize=(8,4))
    plt.plot(step_idx, rew_cum, lw=1)
    plt.xlabel('step')
    plt.ylabel('cumulative reward')
    plt.title(title + ' (cumulative)')
    plt.tight_layout()
    p1c = os.path.join(out_dir, f"{base}_cumulative_by_step.png")
    plt.savefig(p1c, dpi=150)
    plt.close()

    # Plot reward packed by episode (x ~ episode slot, intra-episode spread)
    try:
        if has_episode or episode_seconds:
            x_ep = []
            y_ep = []
            groups = {}
            for i, r in enumerate(rows):
                ep = r.get('episode', 0)
                if ep <= 0 and episode_seconds:
                    ep = int(math.floor(r.get('time', 0) / float(episode_seconds))) + 1
                if ep <= 0:
                    ep = 1
                groups.setdefault(ep, []).append((i, r['reward_sum']))
            for ep, vals in sorted(groups.items()):
                n = len(vals)
                if n <= 1:
                    xs = [ep]
                    ys = [vals[0][1]]
                else:
                    xs = [ep - 0.45 + 0.9 * (j / float(n-1)) for j in range(n)]
                    ys = [v for _, v in vals]
                x_ep.extend(xs)
                y_ep.extend(ys)
            plt.figure(figsize=(9,4))
            plt.plot(x_ep, y_ep, lw=0.8)
            plt.xlabel('episode')
            plt.ylabel('reward (per step)')
            plt.title(title + ' by episode (steps packed)')
            plt.tight_layout()
            p_ep = os.path.join(out_dir, f"{base}_by_episode_steps.png")
            plt.savefig(p_ep, dpi=150)
            plt.close()
    except Exception as e:
        print(f"Episode-packed plot failed for {path}: {e}")

    # Plot reward per step (by time)
    plt.figure(figsize=(8,4))
    plt.plot(t, rew, lw=1)
    plt.xlabel('sim time (s)')
    plt.ylabel('reward (per step)')
    plt.title(title + ' vs time')
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"{base}_by_time.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    # Optional: delivered vs drops for quick health check
    try:
        plt.figure(figsize=(8,4))
        plt.plot(t, delivered, label='delivered', lw=1)
        plt.plot(t, drops, label='drops', lw=1)
        plt.xlabel('sim time (s)')
        plt.ylabel('count per step')
        plt.title((f"Delivered/Drops ({base})"))
        plt.legend()
        plt.tight_layout()
        p3 = os.path.join(out_dir, f"{base}_del_vs_drop.png")
        plt.savefig(p3, dpi=150)
        plt.close()
        print(f"Wrote {p1}, {p2}, {p3}")
    except Exception:
        print(f"Wrote {p1}, {p2}")

def main():
    args = sys.argv[1:]
    paths = []
    episode_seconds = None
    i = 0
    while i < len(args):
        if args[i] == "--episode_seconds" and i+1 < len(args):
            try:
                episode_seconds = float(args[i+1])
            except Exception:
                pass
            i += 2
            continue
        paths.append(args[i])
        i += 1
    if not paths:
        # scan common locations
        roots = ['reports_drl_train', 'reports_drl_eval']
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dn, _, files in os.walk(root):
                for fn in files:
                    if fn == 'drl_step_rewards.csv':
                        paths.append(os.path.join(dn, fn))
    if not paths:
        print("No step reward CSVs provided/found.")
        return
    for p in paths:
        try:
            plot_file(p, episode_seconds=episode_seconds)
        except Exception as e:
            print(f"Failed plotting {p}: {e}")

if __name__ == '__main__':
    main()
