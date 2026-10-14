#!/usr/bin/env bash
set -uo pipefail
TREE="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261014T013636329481-carrier-agreement
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export PYTHONPATH="$TREE"
export NOVA_ANALYTIC_ORACLE_CACHE="$RUN/cache"
printf 'REVISION=%s TREE=%s COMMAND=%s\n' "$(git -C "$TREE" rev-parse HEAD)" "$TREE" "$PY -u $TREE/scripts/analytic_oracle_fixtures/requalify_carriers.py $RUN/diverted-carriers.json diverted:5000"
cd "$TREE"
"$PY" -u -c 'import nova; print("NOVA_FILE=" + nova.__file__)'
"$PY" -u scripts/analytic_oracle_fixtures/requalify_carriers.py "$RUN/diverted-carriers.json" diverted:5000
printf 'EXIT=%s\n' "$?"
