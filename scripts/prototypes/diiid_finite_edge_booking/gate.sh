#!/usr/bin/env bash
# Run one arm of the focused equilibrium gate.
#
# Usage: gate.sh <tree> <base-sha> <output-dir> <base|head>
#
# The arms run as separate jobs because together they outlast a one-hour debug
# partition. The base arm resolves the target list over the base tree, so a
# test file that exists only at the head is not named there.
set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
# The compilation cache is off: two arms sharing one cache directory write the
# same entries, and a cancelled arm leaves a half-written entry behind.
export JAX_ENABLE_COMPILATION_CACHE=0
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
TREE=${1:?worktree}
BASE_SHA=${2:?base revision}
OUT=${3:?output directory}
ARM=${4:?arm}
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT"
WORK=$(mktemp -d /tmp/base-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
if [ "$ARM" = base ]; then
    ROOT="$WORK/base"
    mkdir -p "$ROOT"
    git -C "$TREE" archive "$BASE_SHA" | tar -x -C "$ROOT"
else
    ROOT="$WORK/head"
    mkdir -p "$ROOT"
    git -C "$TREE" archive "$(git -C "$TREE" rev-parse HEAD)" | tar -x -C "$ROOT"
fi
TARGETS=(
    tests/test_current_book_fragments.py
    tests/test_current_book_vectorised.py
    tests/test_booking_scalar_tile.py
    tests/test_equilibrium_forward_solve.py
    tests/test_equilibrium_forward.py
)
if [ "$ARM" = head ]; then
    TARGETS+=(tests/equilibrium/test_finite_edge_booking.py)
fi
printf '# arm: %s; revision: %s; tree: %s; command: gate.sh %s %s %s %s\n' \
    "$ARM" "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "$BASE_SHA")" \
    "$TREE" "$TREE" "$BASE_SHA" "$OUT" "$ARM"
printf 'TARGETS %s\n' "${TARGETS[*]}"
cd "$ROOT" || exit 2
PYTHONPATH="$ROOT" "$PYTHON" -m pytest -p no:cacheprovider -v -rA -s \
    "${TARGETS[@]}" > "$OUT/$ARM.log" 2>&1
STATUS=$?
grep -E '^(FAILED|ERROR)' "$OUT/$ARM.log" > "$OUT/$ARM-failures.txt"
grep -E '^(ROUTE|CONTROL|TANGENT)' "$OUT/$ARM.log" > "$OUT/$ARM-printed.txt"
printf 'arm_exit=%s failures=%s printed=%s\n' "$STATUS" \
    "$(wc -l < "$OUT/$ARM-failures.txt")" "$(wc -l < "$OUT/$ARM-printed.txt")"
tail -3 "$OUT/$ARM.log"
printf 'EXIT=%s\n' "$STATUS"