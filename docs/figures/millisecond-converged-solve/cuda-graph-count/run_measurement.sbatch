#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS

worktree=${NOVA_MEASUREMENT_WORKTREE:?set NOVA_MEASUREMENT_WORKTREE}
output_dir=${NOVA_MEASUREMENT_OUTPUT:?set NOVA_MEASUREMENT_OUTPUT}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
instrument="$worktree/docs/figures/millisecond-converged-solve/cuda-graph-count/measure_cuda_graph_count.py"

printf 'REVISION=%s\n' "$(git -C "$worktree" rev-parse HEAD)"
printf 'TREE=%s\n' "$worktree"
printf 'COMMAND=%q %q --output-dir %q --programs 20\n' "$python" "$instrument" "$output_dir"
printf 'MEASUREMENT_MODULE=%s\n' "$instrument"
printf 'MEASUREMENT_CWD=%s\n' "$(pwd -P)"
printf 'GPU=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
PYTHONPATH="$worktree" "$python" "$instrument" --output-dir "$output_dir" --programs 20
