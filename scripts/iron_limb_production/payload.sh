#!/bin/bash
# One scheduler allocation contains every numerical arm; Python owns the logs.
set -u
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export PYTHONPATH=.
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.iron_limb_production.gate "$@"
code=$?
echo "EXIT=$code"
exit "$code"
