#!/usr/bin/env bash
set -u
ROOT=$1
RUN=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=bash %s/scripts/prototypes/bank_solve_trace/report.sh %s %s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$ROOT" "$RUN"
export PYTHONPATH=$ROOT JAX_PLATFORMS=cpu MPLBACKEND=Agg
trap 'printf "EXIT=%s\n" "$?"' EXIT
"$PY" -P "$ROOT/scripts/prototypes/bank_solve_trace/analyse.py" "$RUN" || exit "$?"
"$PY" -P "$ROOT/scripts/prototypes/bank_solve_trace/plot.py" "$RUN" || exit "$?"
