#!/usr/bin/env bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=west-newton-diagnosis
set -euo pipefail
repo="$1"
saved="$2"
out="$3"
export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu PYTHONPATH="$repo"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
scratch=$(mktemp -d "$TMPDIR/west-diagnosis-${SLURM_JOB_ID}-XXXXXX")
finish() {
  code=$?
  rm -rvf "$scratch"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$scratch" NOVA_COMPILATION_CACHE_ROOT="$out/compilation-cache"
printf '# revision: %s tree: %s command: bash scripts/west_iron_solve/diagnose.sbatch %q %q %q\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$repo" "$saved" "$out"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python -u "$repo/scripts/west_iron_solve/diagnose.py" --saved "$saved" --output "$out"
