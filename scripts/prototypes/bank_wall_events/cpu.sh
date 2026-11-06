#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:20:00
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu
ROOT=$1
RUN=$2
BASE=$3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=%s -P %s/scripts/prototypes/bank_wall_events/suite.py %s %s %s\n' "$BASE" "$ROOT" "$PY" "$ROOT" "$ROOT" "$RUN" "$BASE"
PYTHONPATH="$ROOT" "$PY" -P "$ROOT/scripts/prototypes/bank_wall_events/suite.py" "$ROOT" "$RUN" "$BASE"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
