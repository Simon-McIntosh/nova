#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONSAFEPATH=1
ROOT=$1
RUN=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REVISION=$(git -C "$ROOT" rev-parse HEAD)
SCRATCH=$(mktemp -d "$TMPDIR/booking-controls-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH" > "$RUN/cleanup.log"' EXIT
git -C "$ROOT" archive "$REVISION" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$SCRATCH"
export PYTHONPATH=$SCRATCH
capture() {
  LOG=$1
  shift
  printf '# revision: %s tree: %s command:' "$REVISION" "$SCRATCH" > "$LOG"
  printf ' %q' "$@" >> "$LOG"
  printf '\n' >> "$LOG"
  "$PY" -P -c 'from nova.equilibrium import support_read; from scripts.prototypes.bank_booking_cost import measure; from pathlib import Path; print("MODULE",support_read.__file__,"COMPARISON_MODULE",measure.__file__,"CWD",Path.cwd().resolve())' >> "$LOG" 2>&1
  "$@" >> "$LOG" 2>&1
  RESULT=$?
  echo EXIT=$RESULT >> "$LOG"
}
TILE=$SCRATCH/tests/test_bank_newton_step.py::test_local_booking_uses_wide_default_cell_tile
PARITY=$SCRATCH/tests/test_bank_newton_step.py::test_booking_matrix_parity_guard_refuses_perturbed_matrix
capture "$RUN/head-controls.log" "$PY" -m pytest -p no:cacheprovider -q -s "$TILE" "$PARITY"
if [ "$RESULT" != 0 ]; then exit "$RESULT"; fi
"$PY" - "$SCRATCH/nova/equilibrium/support_read.py" <<'PYCODE'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
assert s.count('_LOCAL_BOOKING_CUT_BATCH_WIDTH = 128')==1
p.write_text(s.replace('_LOCAL_BOOKING_CUT_BATCH_WIDTH = 128','_LOCAL_BOOKING_CUT_BATCH_WIDTH = 32'))
PYCODE
diff -u "$ROOT/nova/equilibrium/support_read.py" "$SCRATCH/nova/equilibrium/support_read.py" > "$RUN/tile-reversion.diff"
capture "$RUN/tile-reverted.log" "$PY" -m pytest -p no:cacheprovider -q -s "$TILE"
if [ "$RESULT" != 1 ]; then exit 17; fi
"$PY" - "$SCRATCH/nova/equilibrium/support_read.py" <<'PYCODE'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
p.write_text(s.replace('_LOCAL_BOOKING_CUT_BATCH_WIDTH = 32','_LOCAL_BOOKING_CUT_BATCH_WIDTH = 128'))
PYCODE
cat > "$SCRATCH/perturbed_matrix.py" <<'PYCODE'
from nova.jax.config import configure_dtypes
configure_dtypes()
import jax
import numpy as np
from scripts.prototypes.bank_booking_cost.measure import compare_matrices
assert jax.config.jax_enable_x64
reference=np.eye(3,dtype=np.float64)
compare_matrices(1.001*reference,reference,1)
PYCODE
capture "$RUN/parity-perturbed.log" "$PY" "$SCRATCH/perturbed_matrix.py"
if [ "$RESULT" != 1 ]; then exit 18; fi
"$PY" - "$SCRATCH/scripts/prototypes/bank_booking_cost/measure.py" <<'PYCODE'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
line='    assert relative <= 1e-12, (lanes, relative)'
assert s.count(line)==1
p.write_text(s.replace(line,'    pass'))
PYCODE
diff -u "$ROOT/scripts/prototypes/bank_booking_cost/measure.py" "$SCRATCH/scripts/prototypes/bank_booking_cost/measure.py" > "$RUN/parity-guard-removal.diff"
capture "$RUN/parity-guard-removed.log" "$PY" -m pytest -p no:cacheprovider -q -s "$PARITY"
if [ "$RESULT" != 1 ]; then exit 19; fi
printf 'CONTROLS_CONFIRMED: tile reversion and parity guard removal both fail the head tests\n'
