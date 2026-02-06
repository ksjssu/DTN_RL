#!/usr/bin/env bash
set -euo pipefail

scen_dir="/mnt/c/dtn/DTN_RL/scenarios/dynamic/low_high_combi_delta010"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
one_sh="${root_dir}/one.sh"

if [[ ! -d "${scen_dir}" ]]; then
  echo "Scenario directory not found: ${scen_dir}"
  exit 1
fi

if [[ ! -x "${one_sh}" ]]; then
  echo "one.sh not found or not executable: ${one_sh}"
  exit 1
fi

shopt -s nullglob
for scen in "${scen_dir}"/*.txt; do
  echo "Running $(basename "${scen}")..."
  "${one_sh}" -b 1 "${scen}"
done

echo "All scenarios completed."
