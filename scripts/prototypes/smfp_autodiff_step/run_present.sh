#!/bin/bash
# One H200 job: the present route's step count and terminal residual from the
# certificate row at the diverted 550-cell seed, tabled beside the fast step.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T174204179431-smfp-autodiff-free-newton
OUT=${RUN}/logs
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT"

export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
export SMFP_REVISION=$(git -C "$ROOT" rev-parse HEAD)

LOG=$OUT/step-present-route.log
: > "$LOG"
{
  echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} DATE=$(date -u +%FT%TZ)"
  echo "COMMAND=$PY -c present_route(diverted, 550)"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" - <<'PY'
import json
from pathlib import Path
from scripts.prototypes.bank_route_rows.measure import PRESENT_CELLS, present_route

row = present_route("diverted", 550, PRESENT_CELLS[550], root=Path("/tmp/smfp-present-route"))
print("PRESENT_ROUTE " + json.dumps({key: value for key, value in row.items() if key not in ("coordinates", "flux", "wall_units")}, sort_keys=True, default=str), flush=True)
PY
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -5 "$LOG"