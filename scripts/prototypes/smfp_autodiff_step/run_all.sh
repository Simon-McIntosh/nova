#!/bin/bash
# One H200 job: every remaining arm of the section gate on one bank identity.
#
# The bank is already published on the store, so this job only loads it.  Each
# arm runs in its own process: the correctness arms share one compilation
# cache, the timing arm runs cache-disabled in a fresh process because its
# warm wall is the measurement, and the present route does not touch the bank.
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T174204179431-smfp-autodiff-free-newton
STORE=${SMFP_STORE:-${RUN}/bank-store}
OUT=${RUN}/logs
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$OUT" "$STORE"

export TMPDIR=/tmp
export PYTHONPATH="$ROOT"
export JAX_PLATFORMS=cuda
export SMFP_REVISION=$(git -C "$ROOT" rev-parse HEAD)
export NOVA_OPERAND_CACHE=${NOVA_OPERAND_CACHE:-/work/projects/imas_gpu/sophelio/evidence/operands}

{
  echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} DATE=$(date -u +%FT%TZ)"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "$OUT/step-environment.log" 2>&1

LOG=$OUT/step-gate.log
: > "$LOG"
{
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.gate --store $STORE"
  "$PY" -m scripts.prototypes.smfp_autodiff_step.gate \
    --store "$STORE" --state-out "$OUT/converged-state.npz" --skip-present
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -3 "$LOG"

LOG=$OUT/step-stages.log
: > "$LOG"
{
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.stages --store $STORE"
  "$PY" -m scripts.prototypes.smfp_autodiff_step.stages --store "$STORE"
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -3 "$LOG"

LOG=$OUT/step-timing-v2.log
: > "$LOG"
export JAX_COMPILATION_CACHE_DIR="$OUT/no-cache-v2-$$"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1000000000
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
{
  echo "COMMAND=$PY -m scripts.prototypes.smfp_autodiff_step.timing --store $STORE"
  "$PY" -m scripts.prototypes.smfp_autodiff_step.timing \
    --store "$STORE" --trace "$OUT/step-trace-v2" --out "$OUT/step-timing-v2.json"
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -3 "$LOG"

LOG=$OUT/step-present-route.log
: > "$LOG"
{
  echo "COMMAND=present_route(diverted, 550)"
  "$PY" - <<'PY'
import json
from pathlib import Path
from scripts.prototypes.bank_route_rows.measure import PRESENT_CELLS, present_route

row = present_route("diverted", 550, PRESENT_CELLS[550], root=Path("/tmp/smfp-present-route"))
print("PRESENT_ROUTE " + json.dumps({key: value for key, value in row.items() if key not in ("coordinates", "flux", "wall_units")}, sort_keys=True, default=str), flush=True)
PY
  echo "EXIT=$?"
} >> "$LOG" 2>&1
tail -3 "$LOG"