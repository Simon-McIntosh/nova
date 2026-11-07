#!/bin/bash
# Run the merged-residual test files at the current head revision, in place on
# the allocation this script is launched from.
LOG="$1"
echo "# command: PYTHONPATH=\$PWD JAX_PLATFORMS=cpu .venv/bin/python -m pytest -p no:cacheprovider -q --timeout=600 tests/biot/test_vertex_edge_tangent_recurrences.py tests/biot/test_graded_layer_near_corner.py" > "$LOG"
echo "# revision: $(git -C "$(dirname "$0")/../../.." rev-parse HEAD)" >> "$LOG"
cd "$(dirname "$0")/../../.." || exit 1
export JAX_PLATFORMS=cpu PYTHONPATH="$PWD" TMPDIR=/dev/shm
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv \
  /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -q \
  --timeout=600 --basetemp /dev/shm/pytest-head-weighted \
  tests/biot/test_vertex_edge_tangent_recurrences.py \
  tests/biot/test_graded_layer_near_corner.py >> "$LOG" 2>&1
echo EXIT=$? >> "$LOG"
tail -1 "$LOG"