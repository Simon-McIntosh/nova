#!/bin/bash
set -uo pipefail

BASE=${1:?base revision required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
unset JAX_PLATFORMS
SOURCE=$(git rev-parse --show-toplevel)
SCRATCH=$(mktemp -d "$TMPDIR/bank-step-base-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH" || exit $?
mkdir -p "$SCRATCH/scripts/prototypes/bank_step_launch_fusion"
cp "$SOURCE/scripts/prototypes/bank_step_launch_fusion/base_solve.py" \
  "$SCRATCH/scripts/prototypes/bank_step_launch_fusion/base_solve.py" || exit $?
cd "$SCRATCH" || exit 2
export PYTHONPATH="$SCRATCH" BASE_REVISION="$BASE"
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$BASE" "$SCRATCH" "$PWD" \
  "$PY -m scripts.prototypes.bank_step_launch_fusion.base_solve"
"$PY" -m scripts.prototypes.bank_step_launch_fusion.base_solve
