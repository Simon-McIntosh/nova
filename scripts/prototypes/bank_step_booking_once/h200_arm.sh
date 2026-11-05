#!/bin/bash
# Stage-wall, launch, and HLO-copy census of one booking-tangent source arm.
set -u
ARM=${1:?arm name required}
LOG=${2:?log path required}
TRACE=${3:?trace directory required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp PYTHONPATH="$ROOT" JAX_PLATFORMS=cuda
export BANK_STEP_TRACE_DIR="$TRACE"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
mkdir -p "$TRACE"
{
  printf 'ARM=%s REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$ARM" \
    "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m scripts/prototypes.bank_step_booking_once.h200_arm $ARM"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m, nova.equilibrium.current as c; print("NOVA="+nova.__file__+" MODULE="+m.__file__+" CURRENT="+c.__file__)'
  "$PY" -m scripts.prototypes.bank_step_speed.profile || { printf 'PROFILE_FAILED\nEXIT=1\n'; exit 1; }
  TRACE_FILE=$(find "$TRACE" -name '*.trace.json.gz' | head -1)
  "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$TRACE_FILE"
  "$PY" -m scripts.prototypes.bank_step_booking_once.hlo_tags "$TRACE/hlo-$ARM.txt"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1