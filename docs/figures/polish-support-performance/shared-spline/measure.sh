#!/usr/bin/env bash
#SBATCH --job-name=polish-shared
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/nia-polish-support-shared-spline/docs/figures/polish-support-performance/shared-spline/slurm-%j.out
#SBATCH --error=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/nia-polish-support-shared-spline/docs/figures/polish-support-performance/shared-spline/slurm-%j.err

set -euo pipefail

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv

worktree=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-relaunch/nia-polish-support-shared-spline
baseline_source="$worktree/docs/figures/polish-support-performance/shared-spline/.baseline-source"
artifact_dir="$worktree/docs/figures/polish-support-performance/shared-spline"
driver="$worktree/benchmarks/polish_support_performance.py"
operands=/home/ITER/mcintos/.config/reckon/crew/reports/nova/bank-regeneration-raw-20260902/current-operands.npz
cache_root="/work/projects/imas_gpu/sophelio/jax-cache/polish-support-performance/shared-spline-${SLURM_JOB_ID}"

mkdir -p "$cache_root/main" "$cache_root/shared"

PYTHONPATH="$baseline_source" uv run --no-sync python "$driver" capture \
  --source-root "$baseline_source" \
  --revision e2aaf78853a70b51b339981e6b6f65c97fa84614 \
  --output "$artifact_dir/raw-main.json" \
  --operands "$operands" \
  --cache "$cache_root/main"

PYTHONPATH="$worktree" uv run --no-sync python "$driver" capture \
  --source-root "$worktree" \
  --revision aecbcebac4a95166c24f1a2df9257f0fd2bf1879 \
  --output "$artifact_dir/raw-shared.json" \
  --operands "$operands" \
  --cache "$cache_root/shared"
