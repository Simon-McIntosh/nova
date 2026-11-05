#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-search-device-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
repo=$1
run=$2
base=$3
head=$4
label=${5:-device}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
for arm in base head; do
  revision=$base
  if [ "$arm" = head ]; then revision=$head; fi
  tree="$scratch/$arm"
  mkdir -p "$tree"
  git -C "$repo" archive "$revision" nova scripts tests benchmarks pyproject.toml uv.lock | tar -x -C "$tree"
  log="$run/$label-$arm.log"
  command=("$python" -P "$repo/scripts/prototypes/bank_solve_linesearch/measure.py" --revision "$revision" --output "$run/$label-$arm.json" --lanes 16)
  printf -v command_text '%q ' "${command[@]}"
  printf '# revision: %s tree: %s command: PYTHONPATH=%s %s\n' "$revision" "$tree" "$tree" "$command_text" > "$log"
  PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
  code=$?
  echo "EXIT=$code" >> "$log"
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
