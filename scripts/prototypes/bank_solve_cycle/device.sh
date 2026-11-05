#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-cycle-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 JAX_PLATFORMS=cuda,cpu
repo=$1
run=$2
label=$3
base=$4
shift 4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$scratch/base"
for arm in "$@"; do
  tree="$repo"
  revision=$(git -C "$repo" rev-parse HEAD)
  if [ "$arm" = base ]; then tree="$scratch/base"; revision=$base; fi
  log="$run/$label-$arm.log"
  command="$python $repo/scripts/prototypes/bank_solve_cycle/measure.py --revision $revision --output $run/$label-$arm.json"
  if [ "${CYCLE_TRACE:-0}" = 1 ]; then command="$python $repo/scripts/prototypes/bank_solve_cycle/trace.py --output $run/$label-$arm.json"; fi
  printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command" > "$log"
  PYTHONPATH="$tree" $command >> "$log" 2>&1
  code=$?
  echo "EXIT=$code" >> "$log"
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
