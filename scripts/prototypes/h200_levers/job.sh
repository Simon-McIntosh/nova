#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
/home/ITER/mcintos/Code/nova/.venv/bin/python "$1/scripts/prototypes/h200_levers/run.py" "$1" "$2"
