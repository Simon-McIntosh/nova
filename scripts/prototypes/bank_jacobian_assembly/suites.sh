#!/bin/bash
#SBATCH --job-name=jac-assembly-suites
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=00:40:00
set -u
HEAD=$1
BASE_SHA=$2
OUT=$3
export TMPDIR=/tmp
SCRATCH=$(mktemp -d "$TMPDIR/jacsuites-$SLURM_JOB_ID-XXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export PYTHONPATH=""
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
FILES="tests/test_bank_step_booking_batch.py tests/test_bank_step_booking.py tests/test_book_primal_buckets.py tests/test_bank_newton_step.py tests/test_bank_solve_driver.py"
mkdir -p "$SCRATCH/base"
git -C "$HEAD" archive "$BASE_SHA" | tar -x -C "$SCRATCH/base"
rc=0
for arm in base head; do
  if [ "$arm" = base ]; then ROOT="$SCRATCH/base"; REV="$BASE_SHA"; else ROOT="$HEAD"; REV=$(git -C "$HEAD" rev-parse HEAD); fi
  for f in $FILES; do
    name=$(basename "$f" .py)
    log="$OUT/$arm-$name.log"
    printf '# revision: %s tree: %s command: %s -m pytest -p no:cacheprovider -q %s/%s\n' \
      "$REV" "$ROOT" "$PY" "$ROOT" "$f" > "$log"
    ( cd "$ROOT" && env TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH="$ROOT" \
      "$PY" -m pytest -p no:cacheprovider -q --basetemp="$SCRATCH/bt-$arm-$name" "$ROOT/$f" ) >> "$log" 2>&1
    code=$?
    printf 'EXIT=%s\n' "$code" >> "$log"
    [ "$code" -ne 0 ] && rc=1
  done
done
printf 'EXIT=%s\n' "$rc"
exit "$rc"
