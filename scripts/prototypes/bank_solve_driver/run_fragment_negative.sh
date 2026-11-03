#!/usr/bin/env bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$ROOT"
SCRATCH=$(mktemp -d "$TMPDIR"/bank-fragment-negative-XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR="$SCRATCH"
COMMAND='/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_solve_driver/fragment_negative_control.py'
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$COMMAND"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_solve_driver/fragment_negative_control.py
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
