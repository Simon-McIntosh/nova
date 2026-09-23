#!/bin/bash
# One compute job for the exact-clip stage probe.
#
# The shared interpreter is invoked directly: `uv` on a compute node injects
# its own --no-sync and the payload dies before Python starts. PYTHONPATH makes
# the worktree's own nova shadow the editable install in the shared
# environment. The queue markers bracket the payload so a follower can tell
# "not started" from "left the queue" by reading the log alone.
set -u
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$1"
echo IN_QUEUE
/home/ITER/mcintos/Code/nova/.venv/bin/python3 "$1/benchmarks/empty_exact_polygon_probe.py"
echo "EXIT=$?"
echo LEFT_QUEUE