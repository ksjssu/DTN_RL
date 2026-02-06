#!/usr/bin/env python3
"""Plot hourly average network buffer utilization for buf20 rng1 runs 1-12."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_hourly_means(path: Path, window: int = 3600) -> Dict[int, float]:
    sums: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = int(float(parts[0]))
                bufocc_mean = float(parts[7])
            except ValueError:
                continue
            hour = t // window
            sums[hour] += bufocc_mean
            counts[hour] += 1

    hourly_means = {}
    for hour, total in sums.items():
        count = counts.get(hour, 0)
        if count > 0:
            hourly_means[hour] = total / count
    return dict(sorted(hourly_means.items()))


def series_from_means(means: Dict[int, float]) -> Tuple[List[int], List[float]]:
    hours = list(means.keys())
    values = [means[h] for h in hours]
    return hours, values


def lerp_color(start: Tuple[int, int, int], end: Tuple[int, int, int], t: float) -> str:
    r = int(round(start[0] + (end[0] - start[0]) * t))
    g = int(round(start[1] + (end[1] - start[1]) * t))
    b = int(round(start[2] + (end[2] - start[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def rainbow_red_to_purple(count: int) -> List[str]:
    start = (255, 0, 0)
    end = (128, 0, 128)
    if count <= 1:
        return [lerp_color(start, end, 0.0)]
    return [lerp_color(start, end, i / (count - 1)) for i in range(count)]


def write_svg_plot(
    series: List[Tuple[int, List[int], List[float]]],
    colors: List[str],
    out_path: Path,
) -> None:
    width, height = 1000, 600
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 60, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_hours = [h for _, hours, _ in series for h in hours]
    all_values = [v for _, _, values in series for v in values]
    x_min = min(all_hours)
    x_max = max(all_hours)
    y_min = min(all_values)
    y_max = max(all_values)

    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min = max(0.0, y_min - 0.1)
        y_max = min(1.0, y_max + 0.1)

    y_pad = (y_max - y_min) * 0.05
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def x_to_svg(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_svg(y: float) -> float:
        return margin_top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>'
    )

    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        y_svg = y_to_svg(y_val)
        lines.append(
            f'<line x1="{margin_left}" y1="{y_svg:.2f}" x2="{margin_left + plot_w}" y2="{y_svg:.2f}" stroke="#e6e6e6" />'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y_svg + 4:.2f}" font-size="12" text-anchor="end" fill="#444">{y_val:.2f}</text>'
        )

    x_ticks = sorted(set(all_hours))
    if len(x_ticks) > 12:
        step = max(1, len(x_ticks) // 10)
        x_ticks = x_ticks[::step]
    for x_val in x_ticks:
        x_svg = x_to_svg(x_val)
        lines.append(
            f'<line x1="{x_svg:.2f}" y1="{margin_top}" x2="{x_svg:.2f}" y2="{margin_top + plot_h}" stroke="#efefef" />'
        )
        lines.append(
            f'<text x="{x_svg:.2f}" y="{margin_top + plot_h + 20}" font-size="12" text-anchor="middle" fill="#444">{x_val}</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333" />'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333" />'
    )

    for idx, (run, hours, values) in enumerate(series):
        color = colors[idx]
        points = " ".join(f"{x_to_svg(h):.2f},{y_to_svg(v):.2f}" for h, v in zip(hours, values))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')

    lines.append(
        f'<text x="{width / 2}" y="30" font-size="18" text-anchor="middle" fill="#222">Hourly Avg Network Buffer Utilization (bufocc_mean)</text>'
    )
    lines.append(
        f'<text x="{width / 2}" y="{height - 20}" font-size="14" text-anchor="middle" fill="#222">Time window (hours)</text>'
    )
    lines.append(
        f'<text x="20" y="{height / 2}" font-size="14" text-anchor="middle" fill="#222" transform="rotate(-90 20 {height / 2})">Average buffer utilization</text>'
    )

    legend_x = margin_left + plot_w - 120
    legend_y = margin_top + 10
    for idx, (run, _, _) in enumerate(series):
        color = colors[idx]
        y = legend_y + idx * 18
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<text x="{legend_x + 24}" y="{y + 4}" font-size="12" fill="#222">run{run}</text>')

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    runs = list(range(1, 13))
    base_dir = Path("reports_drl_train/buf20")
    out_dir = Path("reports_plots/buf20")
    out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    missing = []
    for run in runs:
        path = base_dir / f"drl_train_buf20_100k_rng1_run{run}_RLStateReport.txt"
        if not path.exists():
            missing.append(path)
            continue
        hourly_means = load_hourly_means(path)
        if not hourly_means:
            print(f"[WARN] No data rows parsed: {path}")
            continue
        hours, values = series_from_means(hourly_means)
        series.append((run, hours, values))

    if missing:
        for path in missing:
            print(f"[WARN] Missing RLStateReport: {path}")

    if len(series) < len(runs):
        raise SystemExit("Missing required RLStateReport data for runs 1-12.")

    colors = rainbow_red_to_purple(len(series))
    out_path = out_dir / "buf20_rng1_runs1_12_hourly_bufutil.svg"
    write_svg_plot(series, colors, out_path)
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
