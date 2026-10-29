#!/usr/bin/env bash
# One H200, one bank: the batched steady read's wall, decision flags and kernels.
set -euo pipefail
W=${NOVA_TREE:?set NOVA_TREE to the tree under test}
L=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T210641793337-tracked-read-steady-batch/logs
mkdir -p "$L"
export TMPDIR=/tmp
export PYTHONPATH="$W"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$W"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
set +e
"$PY" -m scripts.prototypes.tracked_read_steady_batch.measure \
  --cells 550 --repeats 20 \
  --store /work/projects/imas_gpu/sophelio/nova-assembled-step-bank \
  --output "$L/batched-steady.json" > "$L/measure.log" 2>&1
code=$?
echo "EXIT=$code" >> "$L/measure.log"
exit $code