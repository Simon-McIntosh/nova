#!/usr/bin/env bash
# Run one imas-efit closed-shell arm on an IMAS pulse.
#
#   run_case.sh <efit-binary> <pulse-dir> <output-dir> <options-dir> <log>
#
# imas-efit is invoked in unified mode: --pulse carries every IDS (pf_active,
# iron_core, wall, magnetics, tf) and --options-dir the solver XML only.
set -uo pipefail

efit=$1
pulse=$2
out=$3
opts=$4
log=$5

export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$(dirname "${log}")"
{
  echo "# $(date -u +%FT%TZ) ${efit} --pulse imas:hdf5?path=${pulse}"
  "${efit}" --pulse "imas:hdf5?path=${pulse}" \
            --output "imas:hdf5?path=${out}" \
            --options-dir "${opts}" \
            --tvals 0 --log-dir "$(dirname "${log}")/solver-logs" \
            "${opts}/efitOptions.xml" -v
  echo "EXIT=$?"
} 2>&1 | tee "${log}"