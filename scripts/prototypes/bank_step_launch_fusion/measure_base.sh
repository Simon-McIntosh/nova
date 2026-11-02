#!/bin/bash
set -u

BASE=${1:?base revision required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=gpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-step-base-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH"
cd "$SCRATCH" || exit 2
export PYTHONPATH="$SCRATCH"
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$BASE" "$SCRATCH" "$PWD" \
  "$PY -m scripts.prototypes.bank_step_booking.measure --arm fresh"
"$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
"$PY" -m scripts.prototypes.bank_step_booking.measure --arm fresh
