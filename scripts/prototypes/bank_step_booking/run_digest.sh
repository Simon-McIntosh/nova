#!/bin/bash
set -u

ROOT=$(git rev-parse --show-toplevel)
BASE=${1:?base revision required}
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-booking-digest-XXXXXX") || exit 2
trap '"$PY" -c '\''import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])'\'' "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/base"
git archive "$BASE" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH/base"

for arm in base head; do
  if [ "$arm" = base ]; then
    tree="$SCRATCH/base"
    revision="$BASE"
  else
    tree="$ROOT"
    revision=$(git rev-parse HEAD)
  fi
  (
    cd "$tree" || exit 2
    export PYTHONPATH="$tree"
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=digest read_and_book\n' "$revision" "$tree" "$PWD"
    "$PY" - <<'PY'
from nova.jax.config import configure_dtypes
configure_dtypes()
import hashlib
import numpy as np
import nova
from nova.equilibrium import current
from tests.test_support_read import _case
print('NOVA='+nova.__file__+' MODULE='+current.__file__, flush=True)
operands, cache, roots, pitch, step, _, _ = _case('diverted')
_, booked = step(operands, cache, roots + 0.03 * pitch, roots)
for name in booked._fields:
    value = np.asarray(getattr(booked, name))
    print('DIGEST', name, hashlib.sha256(value.tobytes()).hexdigest(), flush=True)
PY
  ) || exit $?
done
printf 'EXIT=0\n'
