#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/booking-verify-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
revision=$(git -C "$repo" rev-parse HEAD)
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$scratch/head" "$scratch/receipts"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/head"
suite() {
  for name in test_bank_solve_cycle test_bank_solve_driver test_support_read; do
    log="$run/rings-head-$name.log"
    command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/$name" "$scratch/head/tests/$name.py" -m 'slow or not slow' -vv -s)
    printf '# revision: %s tree: %s command:' "$revision" "$scratch/head" > "$log"
    printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
    PYTHONPATH="$scratch/head" "$python" -P -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
    PYTHONPATH="$scratch/head" "${command[@]}" >> "$log" 2>&1
    echo "EXIT=$?" >> "$log"
  done
}
suite &
suite_pid=$!
log="$run/rings-base-matrices.log"
command=("$python" "$repo/scripts/prototypes/bank_step_booking_batch/measure.py" --run "$scratch/receipts" --arm base --revision "$base" --matrices)
printf '# revision: %s tree: %s command:' "$base" "$scratch/base" > "$log"
printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
PYTHONPATH="$scratch/base" "${command[@]}" >> "$log" 2>&1
result=$?; echo "EXIT=$result" >> "$log"
if [ "$result" -eq 0 ]; then
  log="$run/rings-head-booking.log"
  command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/booking" "$scratch/head/tests/test_bank_step_booking_batch.py" -m 'slow or not slow' -vv -s)
  printf '# revision: %s tree: %s command:' "$revision" "$scratch/head" > "$log"
  printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
  PYTHONPATH="$scratch/head" BOOKING_BASE_MATRICES="$scratch/receipts/base-matrices.npz" "${command[@]}" >> "$log" 2>&1
  result=$?; echo "EXIT=$result" >> "$log"
fi
wait "$suite_pid"
exit "$result"
