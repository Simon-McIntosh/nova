#!/usr/bin/env bash
#SBATCH --job-name=read-count-sensitivity
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -u
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH=$repo PYTHONDONTWRITEBYTECODE=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
printf '# revision: %s tree: %s command: %s -P -m scripts.prototypes.bank_read_count_sensitivity.device %s %s %s %s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$python" "$repo" "$run" "$base" "$head"
"$python" -P -m scripts.prototypes.bank_read_count_sensitivity.device "$repo" "$run" "$base" "$head"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
