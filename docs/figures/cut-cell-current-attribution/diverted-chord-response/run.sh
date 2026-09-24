#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-diverted-chord-response-mismatch-attribution
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
BENCH="$PYTHONPATH/benchmarks/diverted_chord_response_attribution.py"
echo "revision=$(git -C "$PYTHONPATH" rev-parse HEAD) tree=$PYTHONPATH command=/home/ITER/mcintos/Code/nova/.venv/bin/python $BENCH"
nvidia-smi --query-gpu=name,uuid --format=csv
/home/ITER/mcintos/Code/nova/.venv/bin/python "$BENCH"
