import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    name: str
    delivery_prob: float
    overhead_ratio: float | None
    latency_avg: float | None
    created: int | None
    delivered: int | None
    dropped: int | None
    report_dir: str
    scenario_path: str


def _replace_setting(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{value}", text)
    if not text.endswith("\n"):
        text += "\n"
    return text + f"{key} = {value}\n"


def _parse_msg_stats(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def run_one(
    *,
    java_exe: str,
    classpath: str,
    template_path: str,
    out_scenario_dir: str,
    exp_name: str,
    settings: dict[str, str],
    batch_runs: int,
) -> Result:
    with open(template_path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    text = _replace_setting(
        text,
        "Scenario.name",
        f"{exp_name}_rng%%MovementModel.rngSeed%%",
    )
    text = _replace_setting(text, "Report.reportDir", f"reports/{exp_name}/")
    for k, v in settings.items():
        text = _replace_setting(text, k, v)

    out_dir = Path(out_scenario_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = out_dir / f"{exp_name}.txt"
    scenario_path.write_text(text, encoding="utf-8")

    cmd = [
        java_exe,
        "-Xmx512M",
        "-cp",
        classpath,
        "core.DTNSim",
        "-b",
        str(batch_runs),
        str(scenario_path),
    ]
    subprocess.run(cmd, check=True)

    report_dir = Path("reports") / exp_name
    msg_files = sorted(glob.glob(str(report_dir / "*MessageStatsReport.txt")))
    if not msg_files:
        raise RuntimeError(f"No MessageStatsReport found under {report_dir}")
    stats = _parse_msg_stats(msg_files[0])

    delivery_prob = _safe_float(stats.get("delivery_prob"))
    if delivery_prob is None:
        raise RuntimeError(f"Failed to parse delivery_prob from {msg_files[0]}")

    return Result(
        name=exp_name,
        delivery_prob=delivery_prob,
        overhead_ratio=_safe_float(stats.get("overhead_ratio")),
        latency_avg=_safe_float(stats.get("latency_avg")),
        created=_safe_int(stats.get("created")),
        delivered=_safe_int(stats.get("delivered")),
        dropped=_safe_int(stats.get("dropped")),
        report_dir=str(report_dir),
        scenario_path=str(scenario_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        default=(
            "scenarios/dynamic/complex_12h_126/"
            "heuristic_category_d010_complex_12h_126_mixTTL_mixBufVar50M_mixIF_100k_contactLite.txt"
        ),
        help="Base scenario template to mutate",
    )
    parser.add_argument(
        "--java",
        default=r"C:\Program Files\Java\jdk-11\bin\java.exe",
        help="Path to java.exe (JDK/JRE 8+ recommended)",
    )
    parser.add_argument(
        "--cp",
        default="target;lib/ECLA.jar;lib/DTNConsoleConnection.jar",
        help="Classpath for core.DTNSim (Windows separator ';')",
    )
    parser.add_argument(
        "--out-scenarios",
        default=".tmp/heur_sweep",
        help="Directory for generated scenarios",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch runs (-b). Use 1 for rngSeed list first element (usually 1).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Regex to select a subset of experiments by name",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of experiments to run (0 = no limit)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have a MessageStatsReport",
    )
    args = parser.parse_args()

    template_path = args.template
    if not os.path.exists(template_path):
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 2

    experiments: list[tuple[str, dict[str, str]]] = []

    # Delta-only sweeps (category mode)
    for d in (0.08, 0.085, 0.09, 0.095, 0.10, 0.11, 0.12, 0.15):
        name = f"heur2_category_d{int(round(d * 1000)):03d}_complex_12h_126_contactLite"
        experiments.append(
            (
                name,
                {
                    "ProphetRouter.heuristicDelta": f"{d}",
                    "BufferLoadHeuristicReport.mode": "category",
                    "BufferLoadHeuristicReport.paramsEnable": "false",
                },
            )
        )

    # Role split: hubs carry, others offload
    for d in (0.08, 0.10, 0.12):
        name = f"heur2_role_split_d{int(round(d * 1000)):03d}_complex_12h_126_contactLite"
        experiments.append(
            (
                name,
                {
                    "ProphetRouter.heuristicDelta": f"{d}",
                    "BufferLoadHeuristicReport.mode": "role_split",
                    "BufferLoadHeuristicReport.roleThreshold": "0.60",
                    "BufferLoadHeuristicReport.carrierOffloadThreshold": "0.85",
                    "BufferLoadHeuristicReport.carrierOffloadScale": "1.0",
                    "BufferLoadHeuristicReport.paramsEnable": "false",
                },
            )
        )

    # Params enabled variants (category + role_split at d=0.10)
    experiments.append(
        (
            "heur2_category_d100_params_complex_12h_126_contactLite",
            {
                "ProphetRouter.heuristicDelta": "0.10",
                "BufferLoadHeuristicReport.mode": "category",
                "BufferLoadHeuristicReport.paramsEnable": "true",
            },
        )
    )
    experiments.append(
        (
            "heur2_role_split_d100_params_complex_12h_126_contactLite",
            {
                "ProphetRouter.heuristicDelta": "0.10",
                "BufferLoadHeuristicReport.mode": "role_split",
                "BufferLoadHeuristicReport.roleThreshold": "0.60",
                "BufferLoadHeuristicReport.carrierOffloadThreshold": "0.85",
                "BufferLoadHeuristicReport.carrierOffloadScale": "1.0",
                "BufferLoadHeuristicReport.paramsEnable": "true",
            },
        )
    )

    # Combined conditional delta (per-dest) + params: pbase_hybrid mode
    hybrid_weight_sets = [
        ("h0", 0.8, 0.6, 0.2, 0.8),
        ("h1", 1.0, 0.4, 0.2, 1.0),
        ("h2", 0.6, 0.8, 0.2, 0.6),
        ("h3", 1.0, 0.0, 0.2, 1.0),  # pressure+load only (no p_base term)
    ]
    for d in (0.10, 0.15):
        for tag, wp, wb, wr, wl in hybrid_weight_sets:
            name = (
                f"heur7_pbase_hybrid_d{int(round(d * 1000)):03d}_{tag}_"
                "pb10_of90_uf105_go100_gu090_"
                "complex_12h_126_contactLite"
            )
            experiments.append(
                (
                    name,
                    {
                        "ProphetRouter.heuristicDelta": f"{d}",
                        "BufferLoadHeuristicReport.mode": "pbase_hybrid",
                        "BufferLoadHeuristicReport.deltaValue": f"{d}",
                        "BufferLoadHeuristicReport.pBaseThreshold": "0.50",
                        "BufferLoadHeuristicReport.pressureScale": "0.15",
                        "BufferLoadHeuristicReport.hybridWPressure": f"{wp}",
                        "BufferLoadHeuristicReport.hybridWPBase": f"{wb}",
                        "BufferLoadHeuristicReport.hybridWRole": f"{wr}",
                        "BufferLoadHeuristicReport.hybridWLoad": f"{wl}",
                        "BufferLoadHeuristicReport.paramsEnable": "true",
                        "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                        "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
                        "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
                        "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
                        "BufferLoadHeuristicReport.paramsGammaExpOverload": "1.0",
                        "BufferLoadHeuristicReport.paramsGammaExpUnderload": "0.9",
                    },
                )
            )

    # Combined conditional delta (pressure-scaled) + params: pressure mode
    for d in (0.10, 0.15):
        for ps in (0.10, 0.15, 0.20):
            name = (
                f"heur8_pressure_params_d{int(round(d * 1000)):03d}_ps{int(round(ps*100)):02d}_"
                "pb10_of90_uf105_go100_gu090_"
                "complex_12h_126_contactLite"
            )
            experiments.append(
                (
                    name,
                    {
                        "ProphetRouter.heuristicDelta": f"{d}",
                        "BufferLoadHeuristicReport.mode": "pressure",
                        "BufferLoadHeuristicReport.deltaValue": f"{d}",
                        "BufferLoadHeuristicReport.pressureScale": f"{ps}",
                        "BufferLoadHeuristicReport.paramsEnable": "true",
                        "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                        "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
                        "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
                        "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
                        "BufferLoadHeuristicReport.paramsGammaExpOverload": "1.0",
                        "BufferLoadHeuristicReport.paramsGammaExpUnderload": "0.9",
                    },
                )
            )

    # Combined conditional delta (network congestion + self pressure) + optional params: net_pressure mode
    for d in (0.10, 0.15, 0.20):
        for w_mean, w_press in ((0.7, 0.3), (0.5, 0.5)):
            for params_on in (False, True):
                tag = f"nm{int(round(w_mean*100)):02d}_np{int(round(w_press*100)):02d}"
                ptag = "p1" if params_on else "p0"
                name = (
                    f"heur9_net_pressure_{ptag}_d{int(round(d * 1000)):03d}_{tag}_"
                    "complex_12h_126_contactLite"
                )
                settings = {
                    "ProphetRouter.heuristicDelta": f"{d}",
                    "BufferLoadHeuristicReport.mode": "net_pressure",
                    "BufferLoadHeuristicReport.deltaValue": f"{d}",
                    "BufferLoadHeuristicReport.pressureScale": "0.15",
                    "BufferLoadHeuristicReport.netWMean": f"{w_mean}",
                    "BufferLoadHeuristicReport.netWPressure": f"{w_press}",
                    "BufferLoadHeuristicReport.paramsEnable": "true" if params_on else "false",
                }
                if params_on:
                    settings.update(
                        {
                            "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                            "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
                            "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
                            "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
                            "BufferLoadHeuristicReport.paramsGammaExpOverload": "1.0",
                            "BufferLoadHeuristicReport.paramsGammaExpUnderload": "0.9",
                        }
                    )
                experiments.append((name, settings))

    # Param grid sweeps (conditional pInit/beta/gamma updates) around the best delta-only setting.
    base_delta = 0.09
    for pressure_band in (0.10, 0.15):
        pb_tag = int(round(pressure_band * 100))
        for overload_factor in (0.90, 0.95):
            of_tag = int(round(overload_factor * 100))
            for underload_factor in (1.00, 1.05):
                uf_tag = int(round(underload_factor * 100))
                for gamma_over_exp in (1.0, 1.2):
                    go_tag = int(round(gamma_over_exp * 100))
                    for gamma_under_exp in (1.0, 0.9):
                        gu_tag = int(round(gamma_under_exp * 100))
                        name = (
                            f"heur3_params_d{int(round(base_delta * 1000)):03d}_"
                            f"pb{pb_tag:02d}_of{of_tag:02d}_uf{uf_tag:03d}_"
                            f"go{go_tag:03d}_gu{gu_tag:03d}_"
                            "complex_12h_126_contactLite"
                        )
                        experiments.append(
                            (
                                name,
                                {
                                    "ProphetRouter.heuristicDelta": f"{base_delta}",
                                    "BufferLoadHeuristicReport.mode": "category",
                                    "BufferLoadHeuristicReport.paramsEnable": "true",
                                    "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                                    "BufferLoadHeuristicReport.paramsPressureBand": f"{pressure_band}",
                                    "BufferLoadHeuristicReport.paramsOverloadFactor": f"{overload_factor}",
                                    "BufferLoadHeuristicReport.paramsUnderloadFactor": f"{underload_factor}",
                                    "BufferLoadHeuristicReport.paramsGammaExpOverload": f"{gamma_over_exp}",
                                    "BufferLoadHeuristicReport.paramsGammaExpUnderload": f"{gamma_under_exp}",
                                        },
                                    )
                                )

    # Delta sweep for the best param combo found so far (kept small on purpose).
    best_combo = {
        "BufferLoadHeuristicReport.mode": "category",
        "BufferLoadHeuristicReport.paramsEnable": "true",
        "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
        "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
        "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
        "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
        "BufferLoadHeuristicReport.paramsGammaExpOverload": "1.0",
        "BufferLoadHeuristicReport.paramsGammaExpUnderload": "0.9",
    }
    for d in (0.08, 0.085, 0.09, 0.095, 0.10):
        name = (
            f"heur4_paramsbest_d{int(round(d * 1000)):03d}_"
            "pb10_of90_uf105_go100_gu090_"
            "complex_12h_126_contactLite"
        )
        settings = {"ProphetRouter.heuristicDelta": f"{d}"}
        settings.update(best_combo)
        experiments.append((name, settings))

    # Small refinement around the best combo (sweep gamma exponents).
    for gamma_over_exp in (1.0, 1.05, 1.10):
        go_tag = int(round(gamma_over_exp * 100))
        for gamma_under_exp in (0.85, 0.90, 0.95):
            gu_tag = int(round(gamma_under_exp * 100))
            name = (
                "heur5_paramsref_d090_pb10_of90_uf105_"
                f"go{go_tag:03d}_gu{gu_tag:03d}_"
                "complex_12h_126_contactLite"
            )
            experiments.append(
                (
                    name,
                    {
                        "ProphetRouter.heuristicDelta": "0.09",
                        "BufferLoadHeuristicReport.mode": "category",
                        "BufferLoadHeuristicReport.paramsEnable": "true",
                        "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                        "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
                        "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
                        "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
                        "BufferLoadHeuristicReport.paramsGammaExpOverload": f"{gamma_over_exp}",
                        "BufferLoadHeuristicReport.paramsGammaExpUnderload": f"{gamma_under_exp}",
                    },
                )
            )

    # Sweep base wifi/backhaul param profiles with the best factor combo.
    wifi_profiles = [
        ("w0", 0.80, 0.30, 0.985),  # baseline-ish
        ("w1", 0.90, 0.40, 0.990),  # aggressive
        ("w2", 0.70, 0.25, 0.980),  # conservative
    ]
    backhaul_profiles = [
        ("b0", 0.90, 0.40, 0.992),  # baseline-ish
        ("b1", 0.95, 0.50, 0.995),  # aggressive
        ("b2", 0.85, 0.35, 0.990),  # conservative
    ]
    for w_tag, w_pi, w_be, w_ga in wifi_profiles:
        for b_tag, b_pi, b_be, b_ga in backhaul_profiles:
            name = (
                f"heur6_paramsbase_d090_{w_tag}_{b_tag}_"
                "pb10_of90_uf105_go100_gu090_"
                "complex_12h_126_contactLite"
            )
            experiments.append(
                (
                    name,
                    {
                        "ProphetRouter.heuristicDelta": "0.09",
                        "BufferLoadHeuristicReport.mode": "category",
                        "BufferLoadHeuristicReport.paramsEnable": "true",
                        "BufferLoadHeuristicReport.paramsEmaBeta": "0.85",
                        "BufferLoadHeuristicReport.paramsPressureBand": "0.10",
                        "BufferLoadHeuristicReport.paramsOverloadFactor": "0.90",
                        "BufferLoadHeuristicReport.paramsUnderloadFactor": "1.05",
                        "BufferLoadHeuristicReport.paramsGammaExpOverload": "1.0",
                        "BufferLoadHeuristicReport.paramsGammaExpUnderload": "0.9",
                        "BufferLoadHeuristicReport.paramsPInitWifi": f"{w_pi}",
                        "BufferLoadHeuristicReport.paramsBetaWifi": f"{w_be}",
                        "BufferLoadHeuristicReport.paramsGammaWifi": f"{w_ga}",
                        "BufferLoadHeuristicReport.paramsPInitBackhaul": f"{b_pi}",
                        "BufferLoadHeuristicReport.paramsBetaBackhaul": f"{b_be}",
                        "BufferLoadHeuristicReport.paramsGammaBackhaul": f"{b_ga}",
                    },
                )
            )

    results: list[Result] = []
    selected = experiments
    if args.filter:
        rx = re.compile(args.filter)
        selected = [(n, s) for (n, s) in experiments if rx.search(n)]
    if args.max and args.max > 0:
        selected = selected[: args.max]

    for name, settings in selected:
        print(f"[RUN] {name}")
        try:
            report_dir = Path("reports") / name
            if args.skip_existing and report_dir.exists():
                existing = sorted(glob.glob(str(report_dir / "*MessageStatsReport.txt")))
                if existing:
                    stats = _parse_msg_stats(existing[0])
                    delivery_prob = _safe_float(stats.get("delivery_prob"))
                    if delivery_prob is not None:
                        results.append(
                            Result(
                                name=name,
                                delivery_prob=delivery_prob,
                                overhead_ratio=_safe_float(stats.get("overhead_ratio")),
                                latency_avg=_safe_float(stats.get("latency_avg")),
                                created=_safe_int(stats.get("created")),
                                delivered=_safe_int(stats.get("delivered")),
                                dropped=_safe_int(stats.get("dropped")),
                                report_dir=str(report_dir),
                                scenario_path="(skipped)",
                            )
                        )
                        print(f"  SKIP (existing) delivery_prob={delivery_prob:.4f}")
                        continue
            res = run_one(
                java_exe=args.java,
                classpath=args.cp,
                template_path=template_path,
                out_scenario_dir=args.out_scenarios,
                exp_name=name,
                settings=settings,
                batch_runs=args.batch,
            )
            results.append(res)
            print(
                f"  delivery_prob={res.delivery_prob:.4f}"
                + (
                    f" overhead={res.overhead_ratio:.2f}"
                    if res.overhead_ratio is not None
                    else ""
                )
            )
        except subprocess.CalledProcessError as e:
            print(f"  FAILED (exit {e.returncode})", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED ({e})", file=sys.stderr)

    if not results:
        print("No successful runs.", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r.delivery_prob, reverse=True)
    best = results[0]

    print("\n== Summary (top 10) ==")
    for r in results[:10]:
        overhead = f"{r.overhead_ratio:.2f}" if r.overhead_ratio is not None else "-"
        lat = f"{r.latency_avg:.0f}" if r.latency_avg is not None else "-"
        print(f"{r.delivery_prob:.4f}\tovh={overhead}\tlat={lat}\t{r.name}")
    print(f"\nBEST: {best.delivery_prob:.4f} :: {best.name}")
    print(f"Report: {best.report_dir}")
    print(f"Scenario: {best.scenario_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
