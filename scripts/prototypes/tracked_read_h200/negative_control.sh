#!/usr/bin/env bash
# Apply the declared mutation in a scratch copy and run the accuracy gate there.
#
# The mutation averages the Bell fit across the three nodes of the triangle a
# point occupies instead of contracting each node's own jets, which is what the
# prototype did before the cell-local fit. It is applied to an exported copy of
# this revision, never to the worktree, so the log records a run of one tree and
# the worktree stays clean.
set -eu
REV=${1:?REV must name the revision to mutate}
LOG=${2:?LOG must name the log to write}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
export TMPDIR=${TMPDIR:-/tmp}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
D=$(mktemp -d "$TMPDIR/nova-mutant-XXXX")
trap 'rm -rf "$D"' EXIT
mkdir -p "$D/tree"
git -C "$WT" archive "$REV" | tar -x -C "$D/tree"
python3 - "$D/tree/nova/equilibrium/fixed_points.py" <<'PATCH'
import sys

path = sys.argv[1]
text = open(path).read()
needle = """        jets = self.exterior[slots] + jnp.einsum(
            "ncmq,cm->nq", self.rows[slots], self.moments
        )
"""
replacement = needle + """        jets = jnp.broadcast_to(
            jnp.mean(jets, axis=0, keepdims=True), jets.shape
        )
"""
assert text.count(needle) == 1, "the contraction site moved"
open(path, "w").write(text.replace(needle, replacement))
PATCH
{
  echo "# mutant tree $D/tree revision $(git -C "$WT" rev-parse "$REV")"
  echo "# mutation: average the Bell fit across the triangle's nodes"
  echo "# command: python -m pytest -p no:cacheprovider -q -m slow ${SELECT:-} tests/test_tracked_read_accuracy.py"
  cd "$D/tree"
  PYTHONPATH="$D/tree" /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
    -p no:cacheprovider -q -m slow ${SELECT:-} tests/test_tracked_read_accuracy.py \
    --basetemp="$D/bt-mutant"
  echo "EXIT=$?"
} > "$LOG" 2>&1
tail -20 "$LOG"