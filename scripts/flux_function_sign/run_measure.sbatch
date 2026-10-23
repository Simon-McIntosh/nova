#!/bin/bash
# One measurement, one job: the DD4-native regression and the five sources'
# digits are read once, inside this allocation.
#SBATCH --partition=all_debug
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=flux-function-sign
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261023T195812953681-flux-function-sign/slurm-%j.out

set -u
WORKTREE=.
LOG=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261023T195812953681-flux-function-sign/measure.log
OUT=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261023T195812953681-flux-function-sign/measure.json

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE"

{
  echo "# command: sbatch scripts/flux_function_sign/run_measure.sbatch"
  echo "# job: ${SLURM_JOB_ID:-local}"
  echo "# worktree: $WORKTREE"
  echo "# python: /home/ITER/mcintos/Code/nova/.venv/bin/python"
  echo "# nova module: $(/home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import nova; print(nova.__file__)')"
  echo "# revision: $(git -C "$WORKTREE" rev-parse HEAD)"
  cd "$WORKTREE" || exit 1
  /home/ITER/mcintos/Code/nova/.venv/bin/python scripts/flux_function_sign/measure.py --out "$OUT" --slices 3
  status=$?
  echo "EXIT=$status"
  exit $status
} > "$LOG" 2>&1