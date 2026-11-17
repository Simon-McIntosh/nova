#!/usr/bin/env bash
# Collect the iron and convergence records of every arm from its solver log.
#
#   collect-convergence.sh <run-dir> > convergence-lines.txt
#
# Each arm's efit writes jsonl under <run-dir>/logs/<arm>-solver/efit.jsonl.
# The records needed to read a number are the iron source, the build summary,
# IRON_FORM/IRON_CONVERGENCE/IRON_BMOD and the CONVERGENCE_SUMMARY verdict.
set -uo pipefail

run=$1
for dir in "${run}"/logs/*-solver; do
  jsonl="${dir}/efit.jsonl"
  [ -f "${jsonl}" ] || continue
  arm=$(basename "${dir}" -solver)
  grep -q 'IRON_FORM\|CONVERGENCE_SUMMARY' "${jsonl}" || continue
  echo "### ${arm}"
  grep -o '"msg":"[^"]*"' "${jsonl}" \
    | grep -E 'Iron source|IronCoreBuilder: built|IRON_FORM|IRON_CONVERGENCE|IRON_MAXITER|IRON_BMOD|PLATEAU|CONVERGENCE_SUMMARY' \
    | sed 's/^"msg":"//; s/"$//'
  echo
done