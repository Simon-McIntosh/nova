#!/usr/bin/env bash
set -u

WORKTREE=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-census-partition-probe-reads-the-full-state
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
OUT="$WORKTREE/docs/figures/cut-cell-current-attribution/census-after-bank-repair"
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$WORKTREE:$WORKTREE/scripts"

stamp() {
    printf 'revision=%s tree=%s command=%s\n' \
        "$(git -C "$WORKTREE" rev-parse HEAD)" "$WORKTREE" "$1"
}

status=0

{
    stamp "negative control: pass the sliced state to _fixed_design_read again"
    "$PYTHON" "$OUT/negative_control.py"
} > "$OUT/negative-control.log" 2>&1
printf 'exit=%s\n' "$?" > "$OUT/negative-control.exit"

{
    stamp "unit amplitude census"
    CENSUS_OUTPUT_ROOT="$OUT" "$PYTHON" "$OUT/run_census.py"
} > "$OUT/unit-amplitude-census.log" 2>&1
code=$?
printf 'exit=%s\n' "$code" > "$OUT/unit-amplitude-census.exit"
if [ "$code" -ne 0 ]; then status=1; fi

run_test() {
    name=$1
    target=$2
    {
        stamp "$target"
        "$PYTHON" -m pytest -p no:cacheprovider "$target"
    } > "$OUT/$name.log" 2>&1
    code=$?
    printf 'exit=%s\n' "$code" > "$OUT/$name.exit"
    if [ "$code" -ne 0 ]; then status=1; fi
}

run_test moment-path-separatrix tests/test_moment_path_separatrix_test.py
run_test continuous-confined-moments tests/test_continuous_confined_moments.py
run_test exact-bank-callback-capacity tests/test_exact_bank_callback_capacity.py

printf 'GATES_EXIT=%s\n' "$status"
exit "$status"