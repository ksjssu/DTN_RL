#!/usr/bin/env python3
import os
import re
import sys

PAT_SUM = re.compile(r"^episode\s+(?P<ep>\d+)\s+reward_sum\s+(?P<rew>[-\d\.]+)")
PAT_EP = re.compile(
    r"^episode\s+(?P<ep>\d+)\s+reward\s+(?P<rew>[-\d\.]+)\s+avg_delivery_rate\s+(?P<rate>[-\d\.]+)\s+delivered\s+(?P<del>\d+)\s+created\s+(?P<cre>\d+)\s+sim_id\s+(?P<sid>\S+)"
)

def parse_file(path):
    rows = []
    if not os.path.isfile(path):
        print(f"No file: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            m = PAT_EP.match(ln)
            if m:
                d = m.groupdict()
                rows.append({
                    "ep": int(d["ep"]),
                    "reward": float(d["rew"]),
                    "avg": float(d["rate"]),
                    "delivered": int(d["del"]),
                    "created": int(d["cre"]),
                    "sim_id": d.get("sid", "")
                })
                continue
            m = PAT_SUM.match(ln)
            if m:
                d = m.groupdict()
                rows.append({
                    "ep": int(d["ep"]),
                    "reward": float(d["rew"]),
                    "avg": None,
                    "delivered": None,
                    "created": None,
                    "sim_id": ""
                })
    rows.sort(key=lambda r: r["ep"])  # per-file, monotone epi
    return rows

def plot(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}. Install it or view TXT.")
        return
    if not rows:
        print("No data to plot")
        return
    os.makedirs(out_dir, exist_ok=True)
    eps = [r["ep"] for r in rows]
    rew = [r["reward"] for r in rows]
    plt.figure(figsize=(8,4))
    plt.plot(eps, rew, marker='o', ms=3, lw=1)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Episode Reward')
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'drl_episode_reward.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    # Optional: average delivery rate if present
    if any(r.get('avg') is not None for r in rows):
        eps2 = [r["ep"] for r in rows if r.get('avg') is not None]
        avg = [r["avg"] for r in rows if r.get('avg') is not None]
        plt.figure(figsize=(8,4))
        plt.plot(eps2, avg, marker='o', ms=3, lw=1)
        plt.xlabel('episode')
        plt.ylabel('avg_delivery_rate')
        plt.title('Average Delivery Rate per Episode')
        plt.tight_layout()
        p2 = os.path.join(out_dir, 'drl_episode_avg_rate.png')
        plt.savefig(p2, dpi=150)
        plt.close()
        print(f"Wrote {p1} and {p2}")
    else:
        print(f"Wrote {p1}")

def main():
    # default input path
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('reports', 'drl_episode_rewards.txt')
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(path) or '.'
    rows = parse_file(path)
    plot(rows, out_dir)

if __name__ == '__main__':
    main()

