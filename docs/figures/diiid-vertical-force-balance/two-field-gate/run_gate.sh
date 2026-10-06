#!/usr/bin/env bash
#SBATCH --job-name=single-shot-gate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=.
#SBATCH --output=docs/figures/diiid-vertical-force-balance/two-field-gate/slurm-%j.out
#SBATCH --error=docs/figures/diiid-vertical-force-balance/two-field-gate/slurm-%j.err

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -euo pipefail

readonly worktree=${repository_root}
readonly output="$worktree/docs/figures/diiid-vertical-force-balance/two-field-gate"

export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv
export PYTHONPATH="$worktree"

printf 'GATE_START job=%s revision=%s cache=%s\n' \
  "$SLURM_JOB_ID" "$(git -C "$worktree" rev-parse HEAD)" "$HOME/.cache"
uv run --no-sync --directory "$worktree" python \
  "$worktree/benchmarks/diiid_forward_gs_match.py" \
  --single-shot-gate \
  --single-shot-output "$output"
printf 'GATE_COMPLETE job=%s\n' "$SLURM_JOB_ID"
