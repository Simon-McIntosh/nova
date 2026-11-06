#!/bin/bash
#SBATCH --job-name=candidate-device
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
ROOT=$1
RUN=$2
BASE=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=%s -P -m scripts.prototypes.bank_candidate_loop.device %s %s %s\n' "$BASE" "$ROOT" "$PY" "$ROOT" "$RUN" "$BASE"
PYTHONPATH="$ROOT" "$PY" -P -m scripts.prototypes.bank_candidate_loop.device "$ROOT" "$RUN" "$BASE"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
