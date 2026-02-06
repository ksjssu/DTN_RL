#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
one_sh="${root_dir}/one.sh"
reports_root="${root_dir}/reports/heuristic_test/low_high_combi"

scen_dirs=(
  "${root_dir}/scenarios/dynamic/low_high_combi_delta010_5to50"
  "${root_dir}/scenarios/dynamic/low_high_combi_delta020_5to50"
  "${root_dir}/scenarios/dynamic/low_high_combi_delta030_5to50"
)

if [[ ! -x "${one_sh}" ]]; then
  echo "one.sh not found or not executable: ${one_sh}"
  exit 1
fi

if [[ ! -d "${reports_root}" ]]; then
  echo "Report root not found (will run all): ${reports_root}"
fi

shopt -s nullglob globstar
for dir in "${scen_dirs[@]}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "Scenario directory not found: ${dir}"
    exit 1
  fi

  base_dir="$(basename "${dir}")"
  delta_tag="${base_dir#low_high_combi_}"
  delta_tag="${delta_tag%_5to50}"

  report_dir="${reports_root}/${delta_tag}"
  if [[ ! -d "${report_dir}" ]]; then
    report_dir="${reports_root}"
  fi

  for scen in "${dir}"/*.txt; do
    scen_base="$(basename "${scen}" .txt)"
    pattern="${report_dir}"/**/"${scen_base}"_rng1_*
    matches=( ${pattern} )
    if (( ${#matches[@]} )); then
      echo "Skipping ${scen_base} (reports exist)"
      continue
    fi
    echo "Running ${scen_base}..."
    "${one_sh}" -b 1 "${scen}"
  done

  echo "Completed: ${dir}"
  echo ""
done

echo "Missing scenarios completed."
