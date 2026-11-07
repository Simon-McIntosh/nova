#!/bin/bash
#SBATCH --job-name=jac-head-suites
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=00:40:00
set -u
ROOT=$1
OUT=$2
export TMPDIR=/tmp
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
FILES="tests/test_bank_step_booking_batch.py tests/test_bank_step_booking.py tests/test_book_primal_buckets.py tests/test_bank_newton_step.py tests/test_bank_solve_driver.py"
rc=0
for f in $FILES; do
  name=$(basename "$f" .py)
  log="$OUT/head2-$name.log"
  printf '# revision: %s tree: %s command: %s -m pytest -p no:cacheprovider -q %s/%s\n' \
    "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$PY" "$ROOT" "$f" > "$log"
  ( cd "$ROOT" && env TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH="$ROOT" \
    "$PY" -m pytest -p no:cacheprovider -q --basetemp="/tmp/bt2-$SLURM_JOB_ID-$name" "$ROOT/$f" ) >> "$log" 2>&1
  code=$?
  printf 'EXIT=%s\n' "$code" >> "$log"
  [ "$code" -ne 0 ] && rc=1
done
printf 'EXIT=%s\n' "$rc"
exit "$rc"
