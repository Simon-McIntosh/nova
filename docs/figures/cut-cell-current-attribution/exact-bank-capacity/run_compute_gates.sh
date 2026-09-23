#!/usr/bin/env bash
set -u

WORKTREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-exact-bank-callback-capacity-from-the-mesh
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
OUTPUT_ROOT="$WORKTREE/docs/figures/cut-cell-current-attribution/exact-bank-capacity"
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE:$WORKTREE/scripts"

status=0
run_test() {
    name=$1
    target=$2
    {
        printf 'revision=%s tree=%s command=%s\n' "$(git -C "$WORKTREE" rev-parse HEAD)" "$WORKTREE" "$target"
        "$PYTHON" -m pytest -p no:cacheprovider "$target"
    } > "$OUTPUT_ROOT/$name.log" 2>&1
    exit_code=$?
    printf 'exit=%s\n' "$exit_code" > "$OUTPUT_ROOT/$name.exit"
    if [ "$exit_code" -ne 0 ]; then
        status=1
    fi
}

run_test exact-clip-moments tests/test_exact_clip_moments.py
run_test exact-clip-seed tests/test_exact_clip_seed.py
{
    printf 'revision=%s tree=%s command=unit amplitude census\n' "$(git -C "$WORKTREE" rev-parse HEAD)" "$WORKTREE"
    CENSUS_OUTPUT_ROOT="$OUTPUT_ROOT" "$PYTHON" "$OUTPUT_ROOT/run_census.py"
} > "$OUTPUT_ROOT/unit-amplitude-census.log" 2>&1
exit_code=$?
printf 'exit=%s\n' "$exit_code" > "$OUTPUT_ROOT/unit-amplitude-census.exit"
if [ "$exit_code" -ne 0 ]; then
    status=1
fi

printf 'COMPUTE_GATES_EXIT=%s\n' "$status"
exit "$status"
