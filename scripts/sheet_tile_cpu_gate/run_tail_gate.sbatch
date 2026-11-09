#!/usr/bin/env bash
#SBATCH --job-name=sheet-tail-gate
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=400G
#SBATCH --time=01:30:00

# Submit with --output pointing into RUN; the suite and mutation share one job.
set -u
WORKTREE=${WORKTREE:?set WORKTREE}
RUN=${RUN:?set RUN}
SPY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=2
export NOVA_SHEET_TILE_WORKERS=16
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
SCRATCH=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$SCRATCH"' EXIT
cd "$WORKTREE" || exit 2
REV=$(git rev-parse HEAD)
COMMAND='PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_edge_sheet.py tests/test_sheet_tile.py'
{
    echo "# revision: $REV tree: $WORKTREE command: $COMMAND"
    echo "# cwd: $(pwd -P)"
    PYTHONPATH=. "$SPY" -c 'import nova.biot.sheet as m; print("# module:", m.__file__)'
    PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q tests/test_edge_sheet.py tests/test_sheet_tile.py
    SUITE_EXIT=$?
    echo "EXIT=$SUITE_EXIT"
} > "$RUN/after_suite.log" 2>&1

git archive HEAD nova tests scripts conftest.py pyproject.toml | tar -x -C "$SCRATCH"
"$SPY" scripts/sheet_tile_cpu_gate/restore_cylinder_projection.py \
    "$SCRATCH/nova/biot/sheet.py" > "$RUN/mutation.log" 2>&1
MUTATION_EXIT=$?
cd "$SCRATCH" || exit 2
CONTROL='PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s tests/test_sheet_tile.py::test_exterior_cylinder_cpu_pair_is_independent_of_batch_size'
{
    echo "# revision: $REV tree: $SCRATCH command: $CONTROL"
    echo "# cwd: $(pwd -P)"
    MODULE=$(PYTHONPATH=. "$SPY" -c 'import nova.biot.sheet as m; print(m.__file__)')
    echo "# module: $MODULE"
    if [ "$MODULE" != "$SCRATCH/nova/biot/sheet.py" ] || [ "$MUTATION_EXIT" -ne 0 ]; then
        echo 'Control refused: mutant source was not imported or mutation failed.'
        CONTROL_EXIT=3
    else
        PYTHONPATH=. JAX_PLATFORMS=cpu /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s tests/test_sheet_tile.py::test_exterior_cylinder_cpu_pair_is_independent_of_batch_size
        CONTROL_EXIT=$?
    fi
    echo "EXIT=$CONTROL_EXIT"
} > "$RUN/negative-control.log" 2>&1
cd "$WORKTREE" || exit 2
{
    ACTUAL=$(git hash-object nova/biot/sheet.py)
    EXPECTED=$(git rev-parse HEAD:nova/biot/sheet.py)
    echo "# worktree blob: $ACTUAL"
    echo "# head blob: $EXPECTED"
    echo "# introduced marker: $(git show HEAD:nova/biot/sheet.py | grep -c cylinder_height_series)"
    if [ "$ACTUAL" = "$EXPECTED" ]; then
        echo 'Source matches HEAD.'
        SOURCE_EXIT=0
    else
        echo 'Source differs from HEAD.'
        SOURCE_EXIT=3
    fi
    echo "EXIT=$SOURCE_EXIT"
} > "$RUN/source-blob-check.log"
if [ "$CONTROL_EXIT" -ne 1 ] || [ "$SOURCE_EXIT" -ne 0 ]; then
    exit 3
fi
grep -q 'FAILED tests/test_sheet_tile.py::test_exterior_cylinder_cpu_pair_is_independent_of_batch_size' "$RUN/negative-control.log" || exit 3
exit "$SUITE_EXIT"
