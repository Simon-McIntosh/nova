#!/usr/bin/env bash
#SBATCH --job-name=trip-quantum-compiled
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# The receipt resolves the checkout root from its own committed location so it
# runs unchanged from any clone or worktree this revision is checked into.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checkout_root="${NOVA_CHECKOUT_ROOT:-$(cd "$here/../../../../.." && pwd)}"

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${NOVA_PYTHONPATH:-$checkout_root}"

python_bin="${NOVA_PYTHON:-$checkout_root/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-compiled}"

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git -C "$checkout_root" rev-parse HEAD)"

"$python_bin" "$checkout_root/benchmarks/trip_quantum_width_one.py" \
  --cache-root "$cache_root" \
  --member-repeats 2 \
  --repeats 8

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
