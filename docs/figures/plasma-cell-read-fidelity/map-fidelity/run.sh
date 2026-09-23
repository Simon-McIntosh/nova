#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s18-fable-20260922/pcrf-map-fidelity-at-the-analytic-state
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
BENCH="$PYTHONPATH/benchmarks/plasma_cell_map_fidelity.py"
echo "revision=$(git -C "$PYTHONPATH" rev-parse HEAD) tree=$PYTHONPATH command=/home/ITER/mcintos/Code/nova/.venv/bin/python $BENCH"
nvidia-smi --query-gpu=name,uuid --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python "$BENCH"
