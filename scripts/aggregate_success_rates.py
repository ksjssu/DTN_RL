import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Number = float
REPORT_PATTERN = "DeliveryRatePerIntervalReport.txt"


def collect_files(base_dir: Path, buf_tag: str) -> List[Path]:
    files: List[Path] = []
    for path in base_dir.rglob(f"*{buf_tag}*{REPORT_PATTERN}"):
        if path.is_file():
            files.append(path)
    return files


def parse_success_rates(file_path: Path) -> Dict[Tuple[int, int], Number]:
    intervals: Dict[Tuple[int, int], Number] = {}
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                start = int(float(parts[0]))
                end = int(float(parts[1]))
                success = float(parts[-1])
            except ValueError:
                continue
            intervals[(start, end)] = success
    return intervals


def aggregate(base_dir: Path, out_root: Path, buffers: Sequence[int]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for buf in buffers:
        buf_tag = f"buf{buf:02d}"
        files = collect_files(base_dir, buf_tag)
        if not files:
            print(f"No reports found for {buf_tag} under {base_dir}")
            continue

        bucket: Dict[Tuple[int, int], List[Number]] = {}
        for file_path in files:
            intervals = parse_success_rates(file_path)
            for window, success in intervals.items():
                bucket.setdefault(window, []).append(success)

        if not bucket:
            print(f"No interval data parsed for {buf_tag}")
            continue

        rows = []
        for window, values in bucket.items():
            start, end = window
            count = len(values)
            mean_val = statistics.fmean(values)
            min_val = min(values)
            max_val = max(values)
            stdev = statistics.pstdev(values) if count > 1 else 0.0
            rows.append({
                "start": start,
                "end": end,
                "count": count,
                "mean_success": f"{mean_val:.6f}",
                "min_success": f"{min_val:.6f}",
                "max_success": f"{max_val:.6f}",
                "stdev_success": f"{stdev:.6f}",
            })

        rows.sort(key=lambda item: item["start"])

        out_dir = out_root / buf_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "delivery_success_stats.csv"
        with out_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "start",
                    "end",
                    "count",
                    "mean_success",
                    "min_success",
                    "max_success",
                    "stdev_success",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Aggregated {len(files)} files for {buf_tag} -> {out_file}")


def parse_buffers(values: Iterable[str]) -> List[int]:
    buffers: List[int] = []
    for item in values:
        try:
            buffers.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid buffer size: {item}") from exc
    return buffers


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate success rates by buffer size")
    parser.add_argument("base_dir", type=Path, help="Root directory containing reports")
    parser.add_argument("out_root", type=Path, help="Output directory for aggregated stats")
    parser.add_argument(
        "--buffers",
        nargs="*",
        type=int,
        default=[30, 40, 50],
        help="Buffer sizes to include (default: 30 40 50)",
    )
    args = parser.parse_args()

    aggregate(args.base_dir, args.out_root, args.buffers)


if __name__ == "__main__":
    main()
