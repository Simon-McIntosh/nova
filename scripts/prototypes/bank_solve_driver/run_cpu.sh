#!/usr/bin/env bash
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="$ROOT"
SCRATCH=$(mktemp -d "$TMPDIR"/bank-solve-driver-XXXXXX)
trap 'rm -rf "$SCRATCH"' EXIT
COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider tests/test_bank_solve_driver.py -q --basetemp=$SCRATCH/pytest"
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$COMMAND"
/home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova; from nova.equilibrium import support_read; print("NOVA=" + nova.__file__ + " MODULE=" + support_read.__file__)'
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider tests/test_bank_solve_driver.py -q --basetemp="$SCRATCH/pytest"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
