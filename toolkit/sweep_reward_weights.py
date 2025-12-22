#!/usr/bin/env python3
"""
Lightweight reward-weight sweep orchestrator.

Runs ONE for a given scenario multiple times while adjusting DRL reward weights
via the rMAPPO server runtime control endpoint.

Usage examples:
  # Random 50 samples from the defined ranges
  python3 toolkit/sweep_reward_weights.py \
      --scenario scenarios/dynamic/drl_eval/drl_eval_buf20_100k.txt \
      --samples 50

  # Small grid (warning: full grid is huge ~9801 combos)
  python3 toolkit/sweep_reward_weights.py \
      --scenario scenarios/dynamic/drl_eval/drl_eval_buf20_100k.txt \
      --grid --max-combos 60

Notes:
  - Assumes DRL server is already running, e.g.:
      export DRL_PORT=5010; python3 toolkit/drl_server_rmappo.py
  - Uses /set_reward_weights (apply=now) BEFORE each run.
  - Collects episode metrics from latest reports/*_EpisodeReward.txt.
  - Results are appended to reports/reward_logs/reward_weight_sweep_results.csv
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json
import urllib.request


def post_json(url, payload, timeout=5):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def frange(start, stop, step):
    vals = []
    x = start
    # inclusive stop
    while x <= stop + 1e-9:
        # avoid floating dust like 0.6000000000001
        vals.append(round(x, 3))
        x += step
    return vals


def latest_episode_reward_file(report_dirs: list[Path], min_mtime: float = 0.0) -> Path | None:
    """Find the newest *_EpisodeReward.txt under any of the given directories (recursive).

    Args:
        report_dirs: list of base directories to search
        min_mtime: only consider files newer than this mtime
    """
    best = None
    best_mtime = min_mtime
    for base in report_dirs:
        if not base:
            continue
        # search both top-level and recursively to accommodate custom reportDir
        patterns = ["*_EpisodeReward.txt", "**/*_EpisodeReward.txt"]
        for pat in patterns:
            for p in base.glob(pat):
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    continue
                if mt >= best_mtime:
                    best = p
                    best_mtime = mt
    return best


def parse_episode_reward(path: Path):
    """Parse 'average_delivery_rate <val> delivered <X> created <Y>'"""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        # Expect a single line
        line = "".join([ln for ln in txt.splitlines() if ln and not ln.startswith("#")])
        tokens = line.split()
        # very defensive parsing
        rate = float(tokens[tokens.index("average_delivery_rate") + 1]) if "average_delivery_rate" in tokens else float("nan")
        delivered = int(tokens[tokens.index("delivered") + 1]) if "delivered" in tokens else -1
        created = int(tokens[tokens.index("created") + 1]) if "created" in tokens else -1
        return rate, delivered, created
    except Exception:
        return float("nan"), -1, -1


def run_one(scenario_path: str, one_cmd: str = "./one.sh", batch_runs: int = 1) -> int:
    cmd = [one_cmd]
    if batch_runs and batch_runs > 0:
        cmd += ["-b", str(batch_runs)]
    cmd += [scenario_path]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        # try fallback on Windows powershell WSL call
        return subprocess.call(["bash", "-lc", " ".join(cmd)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="Path to ONE settings file")
    ap.add_argument("--server", default="http://127.0.0.1:5010", help="DRL server base URL")
    ap.add_argument("--samples", type=int, default=0, help="Random sample count from grid (0=disabled)")
    ap.add_argument("--grid", action="store_true", help="Use grid order instead of random samples")
    ap.add_argument("--max-combos", type=int, default=0, help="Cap number of grid combos (0=no cap)")
    ap.add_argument("--one", default="./one.sh", help="ONE launcher (default ./one.sh)")
    ap.add_argument("--batch", type=int, default=1, help="ONE -b runs (default 1)")
    ap.add_argument("--no-reset-per-combo", action="store_true", help="Do not restore baseline before each combo (default: reset)")
    args = ap.parse_args()

    # Define search ranges
    pbase_vals = frange(0.0, 2.0, 0.2)  # 11
    buf_vals = frange(0.0, 2.0, 0.2)    # 11
    sr_vals = list(range(0, 401, 50))   # 0,50,...,400 (9)

    combos = []
    for p in pbase_vals:
        for b in buf_vals:
            for srw in sr_vals:
                for srb in sr_vals:
                    combos.append((p, b, float(srw), float(srb)))

    total = len(combos)
    if not args.grid:
        k = args.samples if args.samples > 0 else min(100, total)  # sensible default
        combos = random.sample(combos, k)
        print(f"[SWEEP] Random sampling {len(combos)} / {total} combos")
    else:
        if args.max_combos and args.max_combos > 0 and args.max_combos < total:
            combos = combos[: args.max_combos]
            print(f"[SWEEP] Grid with cap {len(combos)} / {total} combos")
        else:
            print(f"[SWEEP] Full grid {total} combos (this can take very long!)")

    results_dir = Path("reports") / "reward_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "reward_weight_sweep_results.csv"
    write_header = not out_csv.exists()

    # Track baseline mtime to detect latest EpisodeReward file after each run
    # Determine candidate report directories: default 'reports' + scenario-defined Report.reportDir
    rep_dir_default = Path("reports")
    rep_dirs = [rep_dir_default]
    try:
        scen_txt = Path(args.scenario).read_text(encoding="utf-8", errors="ignore")
        for line in scen_txt.splitlines():
            line = line.strip()
            if line.startswith("Report.reportDir") and "=" in line:
                # e.g., Report.reportDir = reports_drl_train/buf20/
                val = line.split("=", 1)[1].strip()
                # strip quotes/spaces
                val = val.strip().strip('"').strip("'")
                if val:
                    rep_dirs.append(Path(val))
    except Exception:
        pass
    # Also add a top-level training reports dir if present
    rep_dirs.append(Path("reports_drl_train"))
    # Deduplicate while preserving order
    seen = set()
    rep_dirs = [p for p in rep_dirs if not (str(p) in seen or seen.add(str(p)))]
    # baseline mtime
    rep_benchmark_mtime = time.time()
    last_seen = time.time()

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp","scenario",
                "PBASE_REWARD_WEIGHT","BUFFER_DIFF_REWARD_WEIGHT",
                "SUCCESS_RATE_WEIGHT","SUCCESS_RATE_BONUS_WEIGHT",
                "avg_delivery_rate","delivered","created","rc"
            ])

        for i, (p, b, srw, srb) in enumerate(combos, 1):
            # Optional: restore baseline to avoid cross-combo interference
            if not args.no_reset_per_combo:
                try:
                    post_json(args.server.rstrip('/') + "/restore_baseline", {}, timeout=5)
                    print(f"[{i}/{len(combos)}] Restored baseline")
                except Exception as e:
                    print(f"[WARN] Failed to restore baseline: {e}")
            # Apply weights (NOW is safe since we apply before starting this episode/run)
            payload = {
                "apply": "now",
                "weights": {
                    "PBASE_REWARD_WEIGHT": p,
                    "BUFFER_DIFF_REWARD_WEIGHT": b,
                    "SUCCESS_RATE_WEIGHT": srw,
                    "SUCCESS_RATE_BONUS_WEIGHT": srb,
                }
            }
            try:
                post_json(args.server.rstrip('/') + "/set_reward_weights", payload, timeout=3)
                print(f"[{i}/{len(combos)}] Applied weights p={p} b={b} sr={srw} srb={srb}")
            except Exception as e:
                print(f"[WARN] Failed to set weights: {e}")

            # Run ONE
            # If batch == 1: single episode; else run N single-episode runs and keep the MAX rate
            if args.batch <= 1:
                rc = run_one(args.scenario, one_cmd=args.one, batch_runs=1)
                time.sleep(0.5)
                reward_file = latest_episode_reward_file(rep_dirs, min_mtime=last_seen)
                if reward_file is None:
                    print(f"[WARN] No EpisodeReward file found after run rc={rc}")
                    rate, delivered, created = float("nan"), -1, -1
                else:
                    last_seen = reward_file.stat().st_mtime
                    rate, delivered, created = parse_episode_reward(reward_file)
                    print(f"  -> Result rate={rate:.4f} delivered={delivered} created={created} file={reward_file.name}")

                w.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    os.path.basename(args.scenario),
                    p, b, srw, srb,
                    rate, delivered, created, rc
                ])
            else:
                best_rate = float('-inf')
                best_del, best_cre, last_rc = -1, -1, 0
                valid_runs = 0
                for j in range(int(args.batch)):
                    rc = run_one(args.scenario, one_cmd=args.one, batch_runs=1)
                    last_rc = rc
                    time.sleep(0.5)
                    reward_file = latest_episode_reward_file(rep_dirs, min_mtime=last_seen)
                    if reward_file is None:
                        print(f"  [WARN] batch {j+1}/{args.batch}: EpisodeReward not found (rc={rc})")
                        continue
                    last_seen = reward_file.stat().st_mtime
                    rate, delivered, created = parse_episode_reward(reward_file)
                    if rate == rate:  # not NaN
                        valid_runs += 1
                        if rate > best_rate:
                            best_rate, best_del, best_cre = rate, delivered, created
                    print(f"    - run {j+1}/{args.batch}: rate={rate:.4f} delivered={delivered} created={created}")

                final_rate = float('nan') if valid_runs == 0 else best_rate
                print(f"  -> MAX over {valid_runs}/{int(args.batch)}: {final_rate if final_rate==final_rate else float('nan'):.4f}")
                w.writerow([
                    datetime.now().isoformat(timespec='seconds'),
                    os.path.basename(args.scenario),
                    p, b, srw, srb,
                    final_rate, best_del, best_cre, last_rc
                ])
            f.flush()

    print(f"[DONE] Results saved to {out_csv}")


if __name__ == "__main__":
    main()
