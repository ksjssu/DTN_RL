#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
one_sh="${root_dir}/one.sh"

scen_dirs=(
  "${root_dir}/scenarios/dynamic/low_high_combi_delta010_5to50"
  "${root_dir}/scenarios/dynamic/low_high_combi_delta020_5to50"
  "${root_dir}/scenarios/dynamic/low_high_combi_delta030_5to50"
)

if [[ ! -x "${one_sh}" ]]; then
  echo "one.sh not found or not executable: ${one_sh}"
  exit 1
fi

for dir in "${scen_dirs[@]}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "Scenario directory not found: ${dir}"
    exit 1
  fi
  shopt -s nullglob
  for scen in "${dir}"/*.txt; do
    echo "Running $(basename "${scen}")..."
    "${one_sh}" -b 1 "${scen}"
  done
  shopt -u nullglob
  echo "Completed: ${dir}"
  echo ""
done

echo "All scenarios completed."
