#!/usr/bin/env bash
# The steady-batch gate: one fresh pytest process, its own basetemp, no timeout wrapper.
set -euo pipefail
W=$(git -C "$(dirname "$0")/../../.." rev-parse --show-toplevel)
L=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T210641793337-tracked-read-steady-batch/logs
mkdir -p "$L"
export TMPDIR=/tmp
export PYTHONPATH="$W"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$W"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
B=$(mktemp -d /tmp/nova-gate-XXXXXX)
UP=$(cd "$W" && git rev-parse HEAD)
echo "# revision=$UP tree=$W command=$PY -m pytest -q tests/test_tracked_read_steady_batch.py" \
  > "$L/gate.log"
echo "# module=$("$PY" -c 'import nova.equilibrium.support_read as m; print(m.__file__)' 2>/dev/null) cwd=$W" \
  >> "$L/gate.log"
set +e
"$PY" -m pytest -p no:cacheprovider --basetemp "$B" \
  -q tests/test_tracked_read_steady_batch.py >> "$L/gate.log" 2>&1
code=$?
echo "EXIT=$code" >> "$L/gate.log"
exit $code