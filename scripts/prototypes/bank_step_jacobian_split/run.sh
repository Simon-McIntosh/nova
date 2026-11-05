#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:50:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/jacobian-split-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
run_dir="$1"
revision=$(git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD)
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
command=("$python" "$SLURM_SUBMIT_DIR/scripts/prototypes/bank_step_jacobian_split/measure.py" --output "$run_dir/timings.json")
printf '# revision: %s tree: %s command:' "$revision" "$SLURM_SUBMIT_DIR"
printf ' %q' "${command[@]}"
printf '\n'
nvidia-smi --query-gpu=name,uuid --format=csv
"${command[@]}"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
