#!/usr/bin/env bash
#SBATCH --job-name=sheet-head-suite
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -u
WORKTREE=${WORKTREE:?set WORKTREE}
RUN=${RUN:?set RUN}
SPY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NOVA_SHEET_TILE_WORKERS=8
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
SCRATCH=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-$$}-XXXX")
exec 3>&1
trap 'rm -rvf "$SCRATCH" >&3' EXIT
export TMPDIR=$SCRATCH
cd "$WORKTREE" || exit 2
COMMAND='PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_edge_sheet.py tests/test_sheet_tile.py'
{
    echo "# revision: $(git rev-parse HEAD) tree: $(pwd -P) command: $COMMAND"
    echo "# cwd: $(pwd -P)"
    PYTHONPATH=. "$SPY" -c 'import nova.biot.sheet as m; print("# module:", m.__file__)'
    eval "$COMMAND"
    SUITE_EXIT=$?
    echo "EXIT=$SUITE_EXIT"
} > "$RUN/after_suite.log" 2>&1
exit "$SUITE_EXIT"
