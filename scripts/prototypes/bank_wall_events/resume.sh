#!/bin/bash
#SBATCH --job-name=wall-precision-path
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:10:00
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
ROOT=$1
RUN=$2
BASE=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=%s -P %s/scripts/prototypes/bank_wall_events/resume.py %s %s %s\n' "$BASE" "$ROOT" "$PY" "$ROOT" "$ROOT" "$RUN" "$BASE"
PYTHONPATH="$ROOT" "$PY" -P "$ROOT/scripts/prototypes/bank_wall_events/resume.py" "$ROOT" "$RUN" "$BASE"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
