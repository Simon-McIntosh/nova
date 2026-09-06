#!/usr/bin/env bash
#SBATCH --job-name=frozen-partition
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-playable/msc-frozen-partition-hooks
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260906T155154456153-msc-frozen-partition-hooks/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/.config/reckon/crew/runs/r-20260906T155154456153-msc-frozen-partition-hooks/slurm-%j.err

set -euo pipefail

checkout_root=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-playable/msc-frozen-partition-hooks
output_root="$checkout_root/docs/figures/millisecond-converged-solve/frozen-partition"
python_bin="${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}"
cache_root="${NOVA_JAX_CACHE_ROOT:-/work/projects/imas_gpu/sophelio/jax-cache/frozen-partition}"

export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$checkout_root"

printf 'MEASUREMENT_START job=%s host=%s revision=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$(git -C "$checkout_root" rev-parse HEAD)"

"$python_bin" "$checkout_root/benchmarks/efit_forward_parity_slice.py" \
  --checkout-root "$checkout_root" \
  --cache-root "$cache_root" \
  --output "$output_root" \
  --frozen-partition-comparison

printf 'MEASUREMENT_COMPLETE job=%s\n' "$SLURM_JOB_ID"
