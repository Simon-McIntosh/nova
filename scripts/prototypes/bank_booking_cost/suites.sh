#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONSAFEPATH=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
ROOT=$1
RUN=$2
BASE=$3
SCRATCH=$(mktemp -d "$TMPDIR/booking-suite-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH" > "$RUN/suite-cleanup.log"' EXIT
mkdir -p "$SCRATCH/base"
git -C "$ROOT" archive "$BASE" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$SCRATCH/base"
cat > "$SCRATCH/booking_provenance.py" <<'PYCODE'
from pathlib import Path
import os

from nova.jax.config import configure_dtypes
configure_dtypes()
from nova.equilibrium import support_read


def pytest_sessionstart(session):
    module = Path(support_read.__file__).resolve()
    expected = Path(os.environ['BOOKING_EXPECTED_TREE']).resolve()
    assert module.is_relative_to(expected), (module, expected)
    print('MODULE_UNDER_TEST', module, 'CWD', Path.cwd().resolve(), flush=True)
PYCODE
arm() {
  ARM=$1
  TREE=$2
  REVISION=$3
  export PYTHONPATH=$TREE:$SCRATCH BOOKING_EXPECTED_TREE=$TREE PYTEST_PLUGINS=booking_provenance
  for NAME in test_bank_newton_step test_bank_step_booking_batch test_bank_step_launch_fusion test_bank_solve_driver; do
    LOG=$RUN/$ARM-$NAME.log
    COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q $TREE/tests/$NAME.py"
    printf '# revision: %s tree: %s command: %s\n' "$REVISION" "$TREE" "$COMMAND" > "$LOG"
    /home/ITER/mcintos/Code/nova/.venv/bin/python -P -c 'from nova.equilibrium import support_read; from pathlib import Path; print("MODULE", support_read.__file__, "CWD", Path.cwd().resolve())' >> "$LOG" 2>&1
    $COMMAND >> "$LOG" 2>&1
    echo EXIT=$? >> "$LOG"
  done
}
arm base "$SCRATCH/base" "$BASE" &
BASE_PID=$!
arm head "$ROOT" "$(git -C "$ROOT" rev-parse HEAD)"
wait "$BASE_PID"
