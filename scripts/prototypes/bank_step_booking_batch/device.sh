#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:20:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/booking-device-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
for arm in base head; do
  tree="$scratch/base"; revision="$base"; extra=--profile
  if [ "$arm" = head ]; then
    for ((attempt=0;attempt<240;attempt++)); do
      if [ -f "$run/head-ready" ]; then break; fi
      sleep 10
    done
    test -f "$run/head-ready" || exit 3
    tree="$repo"; revision=$(git -C "$repo" rev-parse HEAD); extra=
  fi
  command=("$python" "$repo/scripts/prototypes/bank_step_booking_batch/measure.py" --run "$run" --arm "$arm" --revision "$revision" --solve)
  if [ -n "$extra" ]; then command+=("$extra"); fi
  log="$run/device-$arm.log"
  printf '# revision: %s tree: %s command:' "$revision" "$tree" > "$log"
  printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
  PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
  result=$?; echo "EXIT=$result" >> "$log"
  if [ "$result" -ne 0 ]; then exit "$result"; fi
done
