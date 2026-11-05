#!/usr/bin/env bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$ROOT"
COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_solve_driver/suite_arms.py --base $1 --logs $2"
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$COMMAND"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_solve_driver/suite_arms.py --base "$1" --logs "$2"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
