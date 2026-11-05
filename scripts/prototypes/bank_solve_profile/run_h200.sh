#!/usr/bin/env bash
# Run the bank solve stage profile on one reserved H200.
set -u
export TMPDIR=${TMPDIR:-/tmp}
# SLURM copies the batch script into /var/spool and runs it from there, so the
# tree is named by the submitter rather than derived from this file's own path.
ROOT=${NOVA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
if [ ! -f "$ROOT/scripts/prototypes/bank_solve_profile/run_h200.sh" ]; then
  printf 'NO_TREE ROOT=%s\nEXIT=3\n' "$ROOT"
  exit 3
fi
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TARGET=$ROOT/scripts/prototypes/bank_solve_profile/${SCRIPT:-measure}.py
printf 'WRAPPER=%s JOB=%s\n' "$(sha256sum "$0" | cut -d' ' -f1)" "${SLURM_JOB_ID:-none}"
printf 'TREE=%s MODULE=%s REVISION=%s CWD=%s COMMAND=%s\n' \
  "$ROOT" "$ROOT/nova/equilibrium/support_read.py" "$(git -C "$ROOT" rev-parse HEAD)" \
  "$ROOT" "PYTHONPATH=$ROOT $PY $TARGET --repeats ${REPEATS:-12}"
cd "$ROOT" || exit 1
PYTHONPATH="$ROOT" "$PY" "$TARGET" --repeats "${REPEATS:-12}"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"