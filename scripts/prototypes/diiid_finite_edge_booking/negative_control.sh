#!/usr/bin/env bash
# Run the continuity and route tests with the consumer back on the point route.
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
TREE=${1:?worktree}
OUT=${2:?output directory}
mkdir -p "$OUT"
WORK=$(mktemp -d /tmp/neg-XXXXXX)
COPY="$WORK/tree"
mkdir -p "$COPY"
git -C "$TREE" archive HEAD | tar -x -C "$COPY"
sed -i "s/^FINITE_EDGE_MOMENT_ROUTE = .*/FINITE_EDGE_MOMENT_ROUTE = \"point\"/" "$COPY/benchmarks/diiid_forward_gs_match.py"
grep -n "^FINITE_EDGE_MOMENT_ROUTE" "$COPY/benchmarks/diiid_forward_gs_match.py"
printf "# revision: %s; copy: %s; mutation: consumer restored to the value-only point-current route\n" "$(git -C "$TREE" rev-parse HEAD)" "$COPY"
cd "$COPY" || exit 2
PYTHONPATH="$COPY" /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q -s tests/equilibrium/test_finite_edge_booking.py -k "takes_no_step or books_a_finite_edge" > "$OUT/negative-control.log" 2>&1
RESULT=$?
grep -E "^(ROUTE|CONTROL|FAILED|PASSED)" "$OUT/negative-control.log" 2>/dev/null || true
rm -rf "$WORK"
# The log carries its own exit status, so a reader of the file alone can tell
# whether the control ran and failed as it should, without the job log.
printf "EXIT=%s\n" "$RESULT" >> "$OUT/negative-control.log"
printf "EXIT=%s\n" "$RESULT"
exit "$RESULT"
