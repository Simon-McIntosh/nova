#!/bin/bash
set -uo pipefail

REVISION=$(git rev-parse HEAD)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/bank-step-control-XXXXXX") || exit 2
finish() {
  code=$?
  "$PY" -c 'import shutil, sys; print("removing", sys.argv[1]); shutil.rmtree(sys.argv[1])' "$SCRATCH"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
git archive "$REVISION" nova tests scripts benchmarks pyproject.toml | tar -xf - -C "$SCRATCH" || exit 2
"$PY" - "$SCRATCH/nova/equilibrium/support_read.py" <<'PY' || exit $?
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
needle = "    residual = step_residual(operands, moments, moments, support, profile, quadrature)\n"
assert source.count(needle) == 1
path.write_text(source.replace(needle, needle + "    float(residual.booked.raw_current)\n"))
PY
cd "$SCRATCH" || exit 2
export PYTHONPATH="$SCRATCH"
TARGET="$SCRATCH/tests/test_bank_step_launch_fusion.py::test_newton_step_traces_booking_through_assembly"
printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$REVISION" "$SCRATCH" "$PWD" \
  "$PY -m pytest -p no:cacheprovider $TARGET --basetemp=$SCRATCH/basetemp -vv"
"$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
printf 'MUTATION=materialize booked raw current on host before dense assembly\n'
"$PY" -m pytest -p no:cacheprovider "$TARGET" --basetemp="$SCRATCH/basetemp" -vv
