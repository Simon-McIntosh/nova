#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/booking-paired-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 BOOKING_SKIP_CONTROL=1
repo=$1
run=$2
base=$3
head=$(git -C "$repo" rev-parse HEAD)
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$scratch/head" "$run"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/head"
for arm in base head; do
  revision="$base"; if [ "$arm" = head ]; then revision="$head"; fi
  for stage in measure step_profile; do
    command=("$python" "$repo/scripts/prototypes/bank_step_booking_batch/$stage.py" --run "$run" --arm "$arm" --revision "$revision")
    if [ "$stage" = measure ]; then command+=(--solve); fi
    log="$run/$stage-$arm.log"
    printf '# revision: %s tree: %s command:' "$revision" "$scratch/$arm" > "$log"
    printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
    PYTHONPATH="$scratch/$arm" "${command[@]}" >> "$log" 2>&1
    result=$?; echo "EXIT=$result" >> "$log"
    if [ "$result" -ne 0 ]; then exit "$result"; fi
  done
done
PYTHONPATH="$repo" "$python" -c 'from pathlib import Path; import sys; from scripts.prototypes.bank_step_booking_batch.control import main; main(Path(sys.argv[1]),sys.argv[2])' "$run" "$head"
exit "$?"
