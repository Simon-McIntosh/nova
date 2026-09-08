#!/usr/bin/env bash
#SBATCH --job-name=adaptive-refresh
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# The receipt resolves the checkout root from its own committed location so it
# runs unchanged from any clone or worktree this revision is checked into.  A
# batch launcher copies the script to a node-local spool directory, so the
# script's own location identifies the receipt directory only when it is run
# directly; the submission directory carries it in a batch job.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ ! -f "$here/reduced_newton_rows.py" ]; then
  here="$SLURM_SUBMIT_DIR"
fi
checkout_root="${NOVA_CHECKOUT_ROOT:-$(cd "$here/../../../.." && pwd)}"
if [ ! -f "$checkout_root/benchmarks/reduced_newton_rows.py" ]; then
  printf 'checkout root %s holds no measurement driver; ' "$checkout_root" >&2
  printf 'submit from the receipt directory or set NOVA_CHECKOUT_ROOT\n' >&2
  exit 2
fi

export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${NOVA_PYTHONPATH:-$checkout_root}"

# The shared project environment's interpreter is called directly: the compute
# node's uv wrapper injects flags that fail before Python starts.
python_bin="${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/adaptive-refresh}"

cd "$checkout_root"

printf 'MEASUREMENT_START job=%s host=%s root=%s revision=%s\n' \
  "${SLURM_JOB_ID:-none}" "$(hostname)" "$checkout_root" \
  "$(git -C "$checkout_root" rev-parse HEAD)"

receipt="${NOVA_RECEIPT_OUTPUT:-$here/four-rows.json}"

"$python_bin" -m benchmarks.reduced_newton_rows \
  --output "$receipt" \
  --cache-root "$cache_root" \
  --no-production

printf 'MEASUREMENT_COMPLETE job=%s\n' "${SLURM_JOB_ID:-none}"
