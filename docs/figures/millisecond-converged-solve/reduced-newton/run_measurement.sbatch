#!/usr/bin/env bash
#SBATCH --job-name=reduced-newton-rows
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# The receipt resolves the checkout root from its own committed location so it
# runs unchanged from any clone or worktree this revision is checked into.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout_root="${NOVA_CHECKOUT_ROOT:-$(cd "$here/../../../.." && pwd)}"

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${NOVA_PYTHONPATH:-$checkout_root}"

# The shared project environment's interpreter is called directly: the compute
# node's uv wrapper injects flags that fail before Python starts.
python_bin="${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile}"

cd "$checkout_root"

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "${SLURM_JOB_ID:-none}" "$(hostname)" "$(git -C "$checkout_root" rev-parse HEAD)"

"$python_bin" -m benchmarks.reduced_newton_rows \
  --output "$here/four-rows.json" \
  --cache-root "$cache_root" \
  --no-production

printf 'MEASUREMENT_COMPLETE job=%s\n' "${SLURM_JOB_ID:-none}"
