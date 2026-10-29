#!/usr/bin/env bash
# The declared mutation: drop the tracking state from the batched call in a
# scratch copy, so every lane rebuilds and the steady clause must refuse it.
set -euo pipefail
W=${NOVA_TREE:?set NOVA_TREE to the tree under test}
L=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T210641793337-tracked-read-steady-batch/logs
mkdir -p "$L"
export TMPDIR=/tmp
export PYTHONPATH="$W"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$W"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
D=$(mktemp -d /tmp/nova-nc-XXXXXX)
sed 's/tracking=tracking,/tracking=None,/' tests/test_tracked_read_steady_batch.py \
  > "$D/test_tracked_read_steady_batch.py"
grep -n "tracking=None," "$D/test_tracked_read_steady_batch.py" > "$L/negative-control.mutation"
B=$(mktemp -d /tmp/nova-nc-base-XXXXXX)
echo "# mutation: drop the tracking state from the batched call in a scratch copy" \
  > "$L/negative-control.log"
set +e
"$PY" -m pytest -p no:cacheprovider --basetemp "$B" -q \
  "$D/test_tracked_read_steady_batch.py" >> "$L/negative-control.log" 2>&1
code=$?
echo "EXIT=$code" >> "$L/negative-control.log"
rm -rf "$D" "$B"
exit 0