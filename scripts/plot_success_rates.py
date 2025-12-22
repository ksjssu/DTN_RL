import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

BASELINE_CONFIG: Dict[str, Dict[str, str]] = {
    "heuristic_d010": {
        "base": "reports_summary",
        "pattern": "heuristic_d010_{tag}_summary.csv",
        "mode": "summary",
    },
    "heuristic_d020": {
        "base": "reports_summary",
        "pattern": "heuristic_d020_{tag}_summary.csv",
        "mode": "summary",
    },
    "heuristic_d030": {
        "base": "reports_summary",
        "pattern": "heuristic_d030_{tag}_summary.csv",
        "mode": "summary",
    },
    "prophet": {
        "base": "reports_summary",
        "pattern": "prophet_{tag}_summary.csv",
        "mode": "summary",
    },
    "drl_eval": {
        "base": "reports_drl_eval_summary",
        "pattern": "{tag}/delivery_success_stats.csv",
        "mode": "aggregated",
    },
}


def load_summary(csv_path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    success: List[float] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                end = float(row.get("end", row.get("time", "nan")))
                rate = float(row.get("success_rate", row.get("mean_success", "nan")))
            except (ValueError, TypeError):
                continue
            times.append(end)
            success.append(rate)
    return times, success


def load_aggregated(csv_path: Path) -> Tuple[List[float], List[float]]:
    return load_summary(csv_path)


MODE_LOADERS = {
    "summary": load_summary,
    "aggregated": load_aggregated,
}


def plot_buffer(buffer: int, baselines: List[str], out_dir: Path, time_divisor: float) -> None:
    buf_tag = f"buf{buffer:02d}"
    plt.figure(figsize=(10, 6))
    plotted = False

    for baseline in baselines:
        config = BASELINE_CONFIG.get(baseline)
        if not config:
            print(f"[WARN] Unknown baseline '{baseline}'")
            continue

        base_dir = Path(config["base"]).resolve()
        pattern = config["pattern"].format(tag=buf_tag)
        csv_path = base_dir / pattern
        if not csv_path.exists():
            print(f"[WARN] Missing {csv_path}")
            continue

        loader = MODE_LOADERS.get(config["mode"], load_summary)
        times, success = loader(csv_path)
        if not times:
            print(f"[WARN] No data in {csv_path}")
            continue

        if time_divisor and time_divisor != 0:
            times = [t / time_divisor for t in times]

        plt.plot(times, success, marker="o", linewidth=2, label=baseline)
        plotted = True

    if not plotted:
        plt.close()
        print(f"[INFO] Nothing plotted for {buf_tag}")
        return

    plt.title(f"Buffer {buffer}M - Delivery Success Rate")
    xlabel = "Time (seconds)" if time_divisor == 1 else "Time (hours)" if time_divisor == 3600 else "Time"
    plt.xlabel(xlabel)
    plt.ylabel("Delivery Success Rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"success_rate_buf{buffer:02d}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot for {buf_tag} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot delivery success rates per buffer size")
    parser.add_argument("out_dir", type=Path, help="Output directory for plots")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=list(BASELINE_CONFIG.keys()),
        help="Baselines to include",
    )
    parser.add_argument(
        "--buffers",
        nargs="*",
        type=int,
        default=[10, 20, 30, 40, 50],
        help="Buffer sizes to plot",
    )
    parser.add_argument(
        "--time-divisor",
        type=float,
        default=3600.0,
        help="Divide time axis by this value (default 3600 for hours)",
    )
    args = parser.parse_args()

    baselines = args.baselines or list(BASELINE_CONFIG.keys())
    for buffer in args.buffers:
        plot_buffer(buffer, baselines, args.out_dir, args.time_divisor)


if __name__ == "__main__":
    main()
