#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
ROOT=$1
RUN=$2
BASE=$3
SCRATCH=$(mktemp -d "$TMPDIR/booking-difference-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$SCRATCH" > "$RUN/difference-cleanup.log"' EXIT
mkdir -p "$SCRATCH/base"
git -C "$ROOT" archive HEAD nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$SCRATCH/base"
cp "$ROOT/tests/test_bank_newton_step.py" "$SCRATCH/base/tests/test_bank_newton_step.py"
git -C "$ROOT" show "$BASE:nova/equilibrium/support_read.py" > "$SCRATCH/base/nova/equilibrium/support_read.py"
for ARM in base head; do
  TREE=$ROOT
  if [ "$ARM" = base ]; then TREE=$SCRATCH/base; fi
  LOG=$RUN/$ARM-difference.log
  COMMAND="/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s $TREE/tests/test_bank_newton_step.py::test_booking_jacobian_matches_fresh_residual_difference"
  printf '# revision: %s tree: %s command: %s\n' "$BASE/$ARM" "$TREE" "$COMMAND" > "$LOG"
  export PYTHONPATH=$TREE
  /home/ITER/mcintos/Code/nova/.venv/bin/python -P -c 'from nova.equilibrium import support_read; from pathlib import Path; print("MODULE", support_read.__file__, "CWD", Path.cwd().resolve())' >> "$LOG" 2>&1
  $COMMAND >> "$LOG" 2>&1
  echo EXIT=$? >> "$LOG"
done
