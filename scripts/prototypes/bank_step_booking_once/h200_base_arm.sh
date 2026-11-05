#!/bin/bash
# Stage-wall, launch, and HLO-copy census of the archived base tangent source,
# run from a mirror tree so the working tree keeps the head source.
set -u
ARM=${1:?arm name required}
LOG=${2:?log path required}
TRACE=${3:?trace directory required}
BASE_SOURCE=${4:?base current.py required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
MIRROR=${TMPDIR:?TMPDIR must name the run scratch}/base-tree
export TMPDIR=/tmp PYTHONPATH="$MIRROR:$ROOT" JAX_PLATFORMS=cuda
export BANK_STEP_TRACE_DIR="$TRACE"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
mkdir -p "$TRACE"
rm -rf "$MIRROR"
mkdir -p "$MIRROR/nova"
ln -s "$ROOT"/nova/* "$MIRROR/nova/"
rm "$MIRROR/nova/equilibrium"
mkdir -p "$MIRROR/nova/equilibrium"
ln -s "$ROOT"/nova/equilibrium/* "$MIRROR/nova/equilibrium/"
rm "$MIRROR/nova/equilibrium/current.py"
cp "$BASE_SOURCE" "$MIRROR/nova/equilibrium/current.py"
for name in scripts tests benchmarks data input; do
  [ -e "$ROOT/$name" ] && ln -s "$ROOT/$name" "$MIRROR/$name"
done
cd "$ROOT" || exit 1
{
  printf 'ARM=%s REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$ARM" \
    "$(git -C "$ROOT" rev-parse HEAD)" "$MIRROR" "$ROOT" \
    "$PY -m scripts.prototypes.bank_step_speed.profile (base source: $BASE_SOURCE)"
  "$PY" -P -c 'import nova, nova.equilibrium.support_read as m, nova.equilibrium.current as c; print("NOVA="+nova.__file__+" MODULE="+m.__file__+" CURRENT="+c.__file__)'
  "$PY" -m scripts.prototypes.bank_step_speed.profile || { printf 'PROFILE_FAILED\nEXIT=1\n'; exit 1; }
  TRACE_FILE=$(find "$TRACE" -name '*.trace.json.gz' | head -1)
  "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$TRACE_FILE"
  "$PY" -m scripts.prototypes.bank_step_booking_once.hlo_tags "$TRACE/hlo-$ARM.txt"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1