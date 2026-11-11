#!/usr/bin/env bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=west-iron-solve
set -euo pipefail
repo="$1"
out="$2"
source="$3"
shift 3
export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu
export PYTHONPATH="$repo" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
scratch=$(mktemp -d "$TMPDIR/west-iron-${SLURM_JOB_ID}-XXXXXX")
finish() {
  code=$?
  rm -rvf "$scratch"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$scratch"
export NOVA_COMPILATION_CACHE_ROOT="$out/compilation-cache"
mkdir -p "$out"
printf '# revision: %s tree: %s command: bash scripts/west_iron_solve/run.sbatch %q %q %q' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$repo" "$out" "$source"
printf ' %q' "$@"
printf '\n'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python -u "$repo/scripts/west_iron_solve/measure.py" --source "$source" --output "$out" "$@"
