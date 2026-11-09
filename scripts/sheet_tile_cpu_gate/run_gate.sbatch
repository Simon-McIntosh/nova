#!/usr/bin/env bash
#SBATCH --job-name=sheet-tile-full-mesh
#SBATCH --partition=sirius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261109T071913971640-icm-sheet-full-mesh-cpu/slurm-%j.out

# Acceptance measurement for the edge-sheet tile over the whole prototype mesh.
#
# Step 1 runs the tiled build against the unbatched host evaluation over all
# 1696 WEST targets and every one of the mesh's edges, at the test's own 3600 s
# bound. Step 2 restores the subtractive flat-sheet radial bracket in a scratch
# copy of the package and shows the eight-target agreement fails and that the
# far flat pair moves with batch size, then confirms the tracked source is
# untouched.

set -u

WORKTREE=${WORKTREE:?set WORKTREE to the checkout under test}
RUN=${RUN:?set RUN to the directory that receives the logs}
SPY=${SPY:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
WORKERS=${WORKERS:-16}

export TMPDIR=/tmp
export PYTHONPATH="$WORKTREE"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

cd "$WORKTREE" || exit 2
REV=$(git rev-parse HEAD)
STEP1="$RUN/full-mesh-gate.log"
STEP2="$RUN/subtractive-bracket-control.log"
CHECK="$RUN/source-blob-check.log"
MUTATION="$RUN/mutation.log"

header() {
    local log=$1
    local command=$2
    {
        echo "# revision: $REV"
        echo "# tree: $WORKTREE"
        echo "# host: $(hostname) job ${SLURM_JOB_ID:-none}"
        echo "# cwd: $(pwd -P)"
        echo "# module: $($SPY -c 'import nova.biot.sheet as m; print(m.__file__)')"
        echo "# command: $command"
    } > "$log"
}

echo "job ${SLURM_JOB_ID:-none} on $(hostname), $WORKERS workers"
echo "revision $REV"

# Step 1: the whole mesh, every target, every edge, at the unchanged bound.
STEP1_CMD="NOVA_SHEET_TILE_WORKERS=$WORKERS $SPY -m pytest -p no:cacheprovider -q -s tests/test_sheet_tile.py::test_tile_agrees_with_the_unbatched_closed_form"
header "$STEP1" "$STEP1_CMD"
START=$(date +%s)
NOVA_SHEET_TILE_WORKERS=$WORKERS $SPY -m pytest -p no:cacheprovider -q -s \
    tests/test_sheet_tile.py::test_tile_agrees_with_the_unbatched_closed_form \
    >> "$STEP1" 2>&1
STEP1_EXIT=$?
STEP1_WALL=$(( $(date +%s) - START ))
echo "EXIT=$STEP1_EXIT" >> "$STEP1"
echo "# wall_seconds: $STEP1_WALL" >> "$STEP1"

# Step 2: restore the subtractive radial bracket in a scratch copy and re-run
# the eight-target agreement and the batch-size identity against it.
SCRATCH=$(mktemp -d "$TMPDIR/${USER}-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rf "$SCRATCH"' EXIT
cp -a "$WORKTREE/nova" "$SCRATCH/nova"
find "$SCRATCH/nova" -name __pycache__ -type d -prune -exec rm -rf {} +

$SPY "$WORKTREE/scripts/sheet_tile_cpu_gate/restore_subtractive_bracket.py" \
    "$SCRATCH/nova/biot/sheet.py" > "$MUTATION" 2>&1
MUTATION_EXIT=$?
{
    echo "# worktree blob: $(git -C "$WORKTREE" hash-object nova/biot/sheet.py)"
    echo "# head blob: $(git -C "$WORKTREE" rev-parse HEAD:nova/biot/sheet.py)"
    echo "# mutant blob: $(git -C "$WORKTREE" hash-object "$SCRATCH/nova/biot/sheet.py")"
    echo "# mutation exit: $MUTATION_EXIT"
} >> "$MUTATION"

STEP2_CMD="PYTHONPATH=$SCRATCH:$WORKTREE NOVA_SHEET_TILE_TARGETS=8 NOVA_SHEET_TILE_WORKERS=1 $SPY -m pytest -p no:cacheprovider -q -s tests/test_sheet_tile.py::test_tile_agrees_with_the_unbatched_closed_form tests/test_sheet_tile.py::test_flat_cpu_pair_is_independent_of_batch_size"
header "$STEP2" "$STEP2_CMD"
START=$(date +%s)
PYTHONPATH="$SCRATCH:$WORKTREE" NOVA_SHEET_TILE_TARGETS=8 NOVA_SHEET_TILE_WORKERS=1 \
    $SPY -m pytest -p no:cacheprovider -q -s \
    tests/test_sheet_tile.py::test_tile_agrees_with_the_unbatched_closed_form \
    tests/test_sheet_tile.py::test_flat_cpu_pair_is_independent_of_batch_size \
    >> "$STEP2" 2>&1
STEP2_EXIT=$?
STEP2_WALL=$(( $(date +%s) - START ))
echo "EXIT=$STEP2_EXIT" >> "$STEP2"
echo "# wall_seconds: $STEP2_WALL" >> "$STEP2"

# The tracked source is never edited: confirm its blob is still the head blob.
WORKTREE_BLOB=$(git -C "$WORKTREE" hash-object nova/biot/sheet.py)
HEAD_BLOB=$(git -C "$WORKTREE" rev-parse HEAD:nova/biot/sheet.py)
{
    echo "# worktree blob: $WORKTREE_BLOB"
    echo "# head blob: $HEAD_BLOB"
    if [ "$WORKTREE_BLOB" = "$HEAD_BLOB" ]; then
        echo "# source result: tracked source matches HEAD"
    else
        echo "# source result: tracked source DIFFERS from HEAD"
    fi
    for name in test_tile_agrees_with_the_unbatched_closed_form \
                test_flat_cpu_pair_is_independent_of_batch_size; do
        if grep -q "FAILED tests/test_sheet_tile.py::$name" "$STEP2"; then
            echo "# control result: $name failed as required"
        else
            echo "# control result: $name DID NOT FAIL"
        fi
    done
} > "$CHECK"

echo "step 1 exit $STEP1_EXIT in ${STEP1_WALL}s; step 2 exit $STEP2_EXIT in ${STEP2_WALL}s"
# Step 1 must pass; step 2 is the mutation and must fail as a test run (exit 1),
# with both witnesses listed as failures.
if [ "$STEP1_EXIT" -ne 0 ] || [ "$STEP2_EXIT" -ne 1 ]; then
    exit 1
fi
grep -q 'control result: .* DID NOT FAIL' "$CHECK" && exit 1
exit 0