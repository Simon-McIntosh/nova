#!/usr/bin/env bash
#SBATCH --job-name=sheet-accuracy
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --time=01:50:00
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
SCRATCH=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
cd "$WORKTREE" || exit 2
COMMAND='/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/sheet_tile_cpu_gate/accuracy_census.py --workers 32 --output '"$RUN"
{
 echo "# revision: $(git rev-parse HEAD) tree: $(pwd -P) command: PYTHONPATH=. JAX_PLATFORMS=cpu $COMMAND"
 PYTHONPATH=. eval "$COMMAND"
 EXIT=$?
 echo "EXIT=$EXIT"
} > "$RUN/accuracy-census.log" 2>&1
exit "$EXIT"
