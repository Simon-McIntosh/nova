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
#SBATCH --output=docs/figures/millisecond-converged-solve/trip-quantum/trip-quantum-compiled/slurm-%j.out
#SBATCH --error=docs/figures/millisecond-converged-solve/trip-quantum/trip-quantum-compiled/slurm-%j.err

set -euo pipefail

# SLURM spools the batch script, so ``BASH_SOURCE`` does not name the
# committed file; the job instead starts its working directory at the
# submission directory.  The checkout root is therefore the submission
# directory (or an explicit override), which keeps the runner bound to the
# checkout it was launched from rather than to a fixed path.
checkout_root="${NOVA_CHECKOUT_ROOT:-$(pwd)}"
# A worktree's ``.git`` is a file pointing at the repository, not a directory.
if [ ! -e "$checkout_root/.git" ]; then
  echo "checkout root is not a repository (submit from the checkout root): " \
    "$checkout_root" >&2
  exit 2
fi

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
