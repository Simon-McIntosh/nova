#!/usr/bin/env bash
#SBATCH --job-name=sheet-axis-gate
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=400G
#SBATCH --time=02:30:00

# The whole-mesh measurement and both suite arms share one allocation.
set -u
WORKTREE=${WORKTREE:?set WORKTREE}
RUN=${RUN:?set RUN}
BASE=${BASE:-4221309d2e66fd30eadafa7eacc3d967dd233481}
SPY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
export NOVA_SHEET_TILE_WORKERS=16
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
SCRATCH=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
cd "$WORKTREE" || exit 2
REV=$(git rev-parse HEAD)
COMMAND='PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_edge_sheet.py tests/test_sheet_tile.py'
GATE='PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s tests/test_sheet_tile.py::test_tile_agrees_with_the_unbatched_closed_form'

header() {
    echo "# revision: $1 tree: $(pwd -P) command: $2"
    echo "# cwd: $(pwd -P)"
    PYTHONPATH=. "$SPY" -c 'import nova.biot.sheet as m; print("# module:", m.__file__)'
}

{
    header "$REV" "$GATE"
    eval "$GATE"
    GATE_EXIT=$?
    echo "EXIT=$GATE_EXIT"
} > "$RUN/full-mesh-gate.log" 2>&1

if [ "${GATE_ONLY:-0}" -eq 1 ]; then
    exit "$GATE_EXIT"
fi

mkdir "$SCRATCH/base"
git archive "$BASE" nova tests scripts conftest.py pyproject.toml | tar -x -C "$SCRATCH/base"
cd "$SCRATCH/base" || exit 2
{
    header "$BASE" "$COMMAND"
    eval "$COMMAND"
    BASE_EXIT=$?
    echo "EXIT=$BASE_EXIT"
} > "$RUN/baseline_suite.log" 2>&1

cd "$WORKTREE" || exit 2
REV=$(git rev-parse HEAD)
{
    header "$REV" "$COMMAND"
    eval "$COMMAND"
    SUITE_EXIT=$?
    echo "EXIT=$SUITE_EXIT"
} > "$RUN/after_suite.log" 2>&1

printf 'whole-mesh=%s baseline=%s after=%s\n' "$GATE_EXIT" "$BASE_EXIT" "$SUITE_EXIT"
if [ "$GATE_EXIT" -ne 0 ] || [ "$BASE_EXIT" -ne 0 ]; then
    exit 1
fi
exit "$SUITE_EXIT"
