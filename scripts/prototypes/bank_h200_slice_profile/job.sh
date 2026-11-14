#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
REVISION=$3
shift 3
printf '# revision: %s tree: %s command: /home/ITER/mcintos/Code/nova/.venv/bin/python %s/scripts/prototypes/bank_h200_slice_profile/run.py %s %s %s\n' "$REVISION" "$ROOT" "$ROOT" "$ROOT" "$RUN" "$REVISION"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$ROOT/scripts/prototypes/bank_h200_slice_profile/run.py" "$ROOT" "$RUN" "$REVISION" "$@"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
