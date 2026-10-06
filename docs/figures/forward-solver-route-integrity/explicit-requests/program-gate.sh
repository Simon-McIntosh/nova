#!/bin/bash
script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=${repository_root}
cd "$PYTHONPATH" || exit 1
OUT="$PYTHONPATH/docs/figures/forward-solver-route-integrity/explicit-requests"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$PY" "$OUT/program-gate.py" 300 "$OUT/candidate-300.json" candidate > "$OUT/candidate-300.log" 2>&1
printf 'CANDIDATE_300_EXIT=%s\n' "$?"
"$PY" "$OUT/program-gate.py" 1000 "$OUT/candidate-1000.json" candidate > "$OUT/candidate-1000.log" 2>&1
printf 'CANDIDATE_1000_EXIT=%s\n' "$?"
"$PY" "$OUT/program-gate.py" 300 "$OUT/per-request-300.json" per-request > "$OUT/negative-control.log" 2>&1
printf 'NEGATIVE_CONTROL_EXIT=%s\n' "$?"
