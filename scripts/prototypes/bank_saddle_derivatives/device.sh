#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:20:00
set -euo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/saddle-device-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" PYTHONDONTWRITEBYTECODE=1
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
for arm in base head; do
  revision=$base
  if [ "$arm" = head ]; then revision=$head; fi
  mkdir -p "$scratch/$arm" "$run/floor-$arm"
  git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml conftest.py | tar -x -C "$scratch/$arm"
  for process in 0 1 2; do
    log="$run/$arm-$process.log"
    command=(bash "$repo/scripts/prototypes/bank_booking_determinism/measurement.sh" "$run/cache-$arm" "$python" -P "$repo/scripts/prototypes/bank_near_root_polish/measure.py" --run "$run" --arm "$arm" --revision "$revision" --process "$process")
    printf '# revision: %s tree: %s command: PYTHONPATH=%q' "$revision" "$scratch/$arm" "$scratch/$arm" > "$log"
    printf ' %q' "${command[@]}" >> "$log"
    printf '\n' >> "$log"
    set +e
    (cd "$scratch/$arm"; PYTHONPATH="$scratch/$arm" "${command[@]}") >> "$log" 2>&1
    result=$?
    set -e
    echo "EXIT=$result" >> "$log"
    test "$result" -eq 0
  done
  log="$run/floor-$arm.log"
  command=("$python" -P "$repo/scripts/prototypes/bank_saddle_derivatives/floor.py" --run "$run/floor-$arm" --revision "$revision")
  printf '# revision: %s tree: %s command: PYTHONPATH=%q NOVA_COMMON_SEEDS=%q' "$revision" "$scratch/$arm" "$scratch/$arm" "$run/common-seeds.npz" > "$log"
  printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
  set +e
  (cd "$scratch/$arm"; unset XLA_FLAGS; export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=false XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; NOVA_COMMON_SEEDS="$run/common-seeds.npz" PYTHONPATH="$scratch/$arm" "${command[@]}") >> "$log" 2>&1
  result=$?
  set -e
  echo "EXIT=$result" >> "$log"
  test "$result" -eq 0
done
