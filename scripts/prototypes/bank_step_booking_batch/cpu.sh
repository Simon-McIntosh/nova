#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:25:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/booking-cpu-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$scratch/receipts"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
arm() {
  label=$1; tree=$2; revision=$3
  for name in test_bank_solve_cycle test_bank_solve_driver test_support_read; do
    log="$run/cpu-$label-$name.log"
    command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/$label-$name" "$tree/tests/$name.py" -m 'slow or not slow' -vv -s)
    printf '# revision: %s tree: %s command:' "$revision" "$tree" > "$log"
    printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
    PYTHONPATH="$tree" "$python" -P -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
    PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
    echo "EXIT=$?" >> "$log"
  done
}
arm base "$scratch/base" "$base" &
base_pid=$!
log="$run/cpu-base-matrices.log"
command=("$python" "$repo/scripts/prototypes/bank_step_booking_batch/measure.py" --run "$scratch/receipts" --arm base --revision "$base" --matrices)
printf '# revision: %s tree: %s command:' "$base" "$scratch/base" > "$log"
printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
PYTHONPATH="$scratch/base" "${command[@]}" >> "$log" 2>&1
result=$?; echo "EXIT=$result" >> "$log"
if [ "$result" -ne 0 ]; then wait "$base_pid"; exit "$result"; fi
for ((attempt=0;attempt<240;attempt++)); do
  if [ -f "$run/head-ready" ]; then break; fi
  sleep 10
done
test -f "$run/head-ready" || exit 3
revision=$(git -C "$repo" rev-parse HEAD)
log="$run/cpu-head-booking.log"
command=("$python" -m pytest -p no:cacheprovider "--basetemp=$scratch/head-booking" "$repo/tests/test_bank_step_booking_batch.py" -m 'slow or not slow' -vv -s)
printf '# revision: %s tree: %s command:' "$revision" "$repo" > "$log"
printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
PYTHONPATH="$repo" BOOKING_BASE_MATRICES="$scratch/receipts/base-matrices.npz" "${command[@]}" >> "$log" 2>&1
result=$?; echo "EXIT=$result" >> "$log"
arm head "$repo" "$revision"
wait "$base_pid"
exit "$result"
