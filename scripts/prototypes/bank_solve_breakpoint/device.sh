#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:15:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/breakpoint-device-XXXXXX")
trap 'result=$?; rm -rfv "$scratch"; echo "EXIT=$result"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cuda,cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
repo=$1
run=$2
base=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/base"
log="$run/device-diagnose.log"
printf '# revision: %s tree: %s command: PYTHONPATH=%s %s %s\n' "$base" "$scratch/base" "$scratch/base" "$python" "$repo/scripts/prototypes/bank_solve_breakpoint/diagnose.py" > "$log"
PYTHONPATH="$scratch/base" "$python" "$repo/scripts/prototypes/bank_solve_breakpoint/diagnose.py" >> "$log" 2>&1
echo "EXIT=$?" >> "$log"
for ((attempt=0;attempt<240;attempt++)); do
  if [ -f "$run/head-ready" ]; then break; fi
  sleep 10
done
test -f "$run/head-ready" || exit 3
head=$(cat "$run/head-ready")
mkdir -p "$scratch/head"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml uv.lock conftest.py | tar -x -C "$scratch/head"
for arm in base head; do
  revision=$base; if [ "$arm" = head ]; then revision=$head; fi
  log="$run/device-$arm.log"
  command=("$python" "$repo/scripts/prototypes/bank_solve_breakpoint/measure.py" --run "$run" --arm "$arm" --revision "$revision")
  printf '# revision: %s tree: %s command: PYTHONPATH=%q' "$revision" "$scratch/$arm" "$scratch/$arm" > "$log"
  printf ' %q' "${command[@]}" >> "$log"; printf '\n' >> "$log"
  PYTHONPATH="$scratch/$arm" "${command[@]}" >> "$log" 2>&1
  result=$?; echo "EXIT=$result" >> "$log"
  if [ "$result" -ne 0 ]; then exit "$result"; fi
done
