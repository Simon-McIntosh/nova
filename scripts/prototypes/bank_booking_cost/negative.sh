#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
ROOT=$1
RUN=$2
REVISION=$(git -C "$ROOT" rev-parse HEAD)
SCRATCH=$(mktemp -d "$TMPDIR/booking-negative-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH" > "$RUN/negative-cleanup.log"' EXIT
git -C "$ROOT" archive "$REVISION" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$SCRATCH"
cat > "$SCRATCH/drop_booking.py" <<'PY'
"""Drop the booking derivative only after the converged fixture is constructed."""
import jax.numpy as jnp
from nova.equilibrium import support_read


def pytest_runtest_call(item):
    original = support_read.step_jacobian

    def dropped(*args, **kwargs):
        matrix = original(*args, **kwargs)
        return jnp.eye(matrix.shape[0], dtype=matrix.dtype)

    support_read.step_jacobian = dropped
PY
export PYTHONPATH=$SCRATCH
LOG=$RUN/negative-control.log
COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -p drop_booking -q -s $SCRATCH/tests/test_bank_newton_step.py::test_booking_jacobian_matches_fresh_residual_difference"
printf '# revision: %s tree: %s command: %s\n' "$REVISION" "$SCRATCH" "$COMMAND" > "$LOG"
/home/ITER/mcintos/Code/nova/.venv/bin/python -P -c 'from nova.equilibrium import support_read; from pathlib import Path; print("MODULE", support_read.__file__, "CWD", Path.cwd().resolve())' >> "$LOG" 2>&1
cat "$SCRATCH/drop_booking.py" > "$RUN/negative-mutation.py"
$COMMAND >> "$LOG" 2>&1
RESULT=$?
echo EXIT=$RESULT >> "$LOG"
exit "$RESULT"
