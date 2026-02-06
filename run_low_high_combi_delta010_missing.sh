#!/usr/bin/env bash
set -euo pipefail

scen_dir="/mnt/c/dtn/DTN_RL/scenarios/dynamic/low_high_combi_delta010"
report_root="/mnt/c/dtn/DTN_RL/reports/heuristic_test/low_high_combi"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
one_sh="${root_dir}/one.sh"

if [[ ! -d "${scen_dir}" ]]; then
  echo "Scenario directory not found: ${scen_dir}"
  exit 1
fi

if [[ ! -d "${report_root}" ]]; then
  echo "Report directory not found: ${report_root}"
  exit 1
fi

if [[ ! -x "${one_sh}" ]]; then
  echo "one.sh not found or not executable: ${one_sh}"
  exit 1
fi

shopt -s nullglob globstar
for scen in "${scen_dir}"/*.txt; do
  scen_base="$(basename "${scen}" .txt)"
  pattern="${report_root}"/**/"${scen_base}"_rng1_*
  matches=( ${pattern} )
  if (( ${#matches[@]} )); then
    echo "Skipping ${scen_base} (reports exist)"
    continue
  fi
  echo "Running ${scen_base}..."
  "${one_sh}" -b 1 "${scen}"
done

echo "Missing scenarios completed."
