#!/usr/bin/env bash
# Run one imas-efit closed-shell arm on an IMAS pulse.
#
#   run_case.sh <srcdir> <efit-binary> <pulse-dir> <options-dir> <output-dir> <log>
#
# imas-efit is invoked in unified mode: --pulse carries every IDS (pf_active,
# iron_core, wall, magnetics, tf) and --options-dir the solver XML only.  The
# log always ends with EXIT=<n>, and the script exits with that status, so a
# failed slice is readable from the log without a scheduler record.
set -uo pipefail

src=$1
efit=$2
pulse=$3
opts=$4
out=$5
log=$6

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh" "${src}"
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$(dirname "${log}")"
status_file=$(mktemp)
{
  echo "# $(date -u +%FT%TZ) ${efit} --pulse imas:hdf5?path=${pulse}"
  "${efit}" --pulse "imas:hdf5?path=${pulse}" \
            --output "imas:hdf5?path=${out}" \
            --options-dir "${opts}" \
            --tvals 0 --log-dir "${log%.log}-solver" \
            "${opts}/efitOptions.xml" -v
  status=$?
  printf '%s' "${status}" > "${status_file}"
  echo "EXIT=${status}"
} 2>&1 | tee "${log}"
status=$(cat "${status_file}")
rm -f "${status_file}"
exit "${status}"