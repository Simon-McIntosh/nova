#!/usr/bin/env bash
# Cold stage split of the diverted certificate carrier's forward-solve programs.
set -u
repo="$1"; out="$2"; cells="${3:-550}"
export TMPDIR=/tmp
export PYTHONPATH="$repo"
export JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_LOG_COMPILES=1
export JAX_PLATFORMS=cuda
interpreter=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$out"
printf 'repo=%s cells=%s out=%s\n' "$repo" "$cells" "$out"
"$interpreter" -u "$repo/scripts/prototypes/solve_wall_breakdown/measure.py" \
  --cells "$cells" --batch 16 --output "$out/stages.json"
code=$?
printf 'measure_exit=%s\n' "$code"
printf 'EXIT=%s\n' "$code"
exit "$code"
