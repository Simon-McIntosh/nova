#!/usr/bin/env bash
set -uo pipefail
export TMPDIR=/tmp
D=$(mktemp -d "$TMPDIR/fidelity-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$D"' EXIT
export TMPDIR="$D" PYTHONPATH="$FIDELITY_ROOT" MPLBACKEND=Agg
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REV=$(git -C "$FIDELITY_ROOT" rev-parse HEAD)
printf '# revision: %s tree: %s command: %s scripts/diiid_map_fidelity/measure.py --output %s/measurement\n' "$REV" "$FIDELITY_ROOT" "$PY" "$FIDELITY_RUN"
"$PY" "$FIDELITY_ROOT/scripts/diiid_map_fidelity/measure.py" --output "$FIDELITY_RUN/measurement"
RESULT=$?
rm -rvf "$D"
trap - EXIT
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
