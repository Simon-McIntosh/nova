#!/usr/bin/env bash
# Run the focused equilibrium gate at the base revision and at the two heads.
set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
TREE=${1:?worktree}
BASE_SHA=${2:?base revision}
OUT=${3:?output directory}
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT"
WORK=$(mktemp -d /tmp/base-XXXXXX)
BASE="$WORK/base"
mkdir -p "$BASE"
git -C "$TREE" archive "$BASE_SHA" | tar -x -C "$BASE"
printf '# revision: head %s; base %s; tree %s; command: gate.sh\n' \
    "$(git -C "$TREE" rev-parse HEAD)" "$BASE_SHA" "$TREE"
TARGETS=(
    tests/test_current_book_fragments.py
    tests/test_current_book_vectorised.py
    tests/test_booking_scalar_tile.py
    tests/test_equilibrium_forward_solve.py
    tests/test_equilibrium_forward.py
)
printf '== baseline arm at %s\n' "$BASE_SHA"
cd "$BASE" || exit 2
PYTHONPATH="$BASE" "$PYTHON" -m pytest -p no:cacheprovider -q \
    "${TARGETS[@]}" > "$OUT/baseline.log" 2>&1
BASELINE_STATUS=$?
printf 'baseline_exit=%s\n' "$BASELINE_STATUS"
printf '== after arm at %s\n' "$TREE"
cd "$TREE" || exit 2
PYTHONPATH="$TREE" "$PYTHON" -m pytest -p no:cacheprovider -q -s \
    tests/equilibrium/test_finite_edge_booking.py "${TARGETS[@]}" \
    > "$OUT/after.log" 2>&1
AFTER_STATUS=$?
printf 'after_exit=%s\n' "$AFTER_STATUS"
grep -E '^(FAILED|ERROR)' "$OUT/baseline.log" > "$OUT/baseline-failures.txt"
grep -E '^(FAILED|ERROR)' "$OUT/after.log" > "$OUT/after-failures.txt"
tail -2 "$OUT/baseline.log"
tail -2 "$OUT/after.log"
rm -rf "$WORK"
if [ "$AFTER_STATUS" -ne 0 ]; then
    printf 'EXIT=%s\n' "$AFTER_STATUS"
else
    printf 'EXIT=0\n'
fi