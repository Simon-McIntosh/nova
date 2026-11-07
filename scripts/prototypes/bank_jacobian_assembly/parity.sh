#!/bin/bash
#SBATCH --job-name=jac-assembly-parity
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:35:00
set -u
ROOT=$1
LOG=$2
cd "$ROOT" || exit 1
printf '# revision: %s tree: %s command: %s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" \
  "$PY -m pytest -p no:cacheprovider -q tests/test_bank_step_booking_batch.py::test_shared_ring_geometry_matches_per_direction"
export TMPDIR=/tmp PYTHONPATH=""
export PYTHONPATH="$ROOT"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$PY" -c "import nova; print('NOVA', nova.__file__)"
"$PY" -m pytest -p no:cacheprovider -p timeout -q \
  --basetemp=/tmp/jacparity-$SLURM_JOB_ID \
  tests/test_bank_step_booking_batch.py::test_shared_ring_geometry_matches_per_direction
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
