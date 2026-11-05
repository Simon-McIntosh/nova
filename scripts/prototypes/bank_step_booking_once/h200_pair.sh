#!/bin/bash
# Both booking-tangent arms in one allocation, each from its own mirror tree.
set -u
LOG=${1:?log path required}
TRACE_ROOT=${2:?trace root required}
BASE_SOURCE=${3:?base current.py required}
COLLAPSED_SOURCE=${4:?collapsed current.py required}
ROOT=$(git rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm

mirror() {
  local name=$1 source=$2 tree="${TMPDIR}/pair-$name"
  rm -rf "$tree"
  mkdir -p "$tree/nova/equilibrium"
  for entry in "$ROOT"/nova/*; do
    [ "$(basename "$entry")" = equilibrium ] || ln -s "$entry" "$tree/nova/"
  done
  for entry in "$ROOT"/nova/equilibrium/*; do
    [ "$(basename "$entry")" = current.py ] || ln -s "$entry" "$tree/nova/equilibrium/"
  done
  cp "$source" "$tree/nova/equilibrium/current.py"
  for name2 in scripts tests data input benchmarks; do
    [ -e "$ROOT/$name2" ] && ln -s "$ROOT/$name2" "$tree/$name2"
  done
  printf '%s\n' "$tree"
}

{
  for arm in base collapsed; do
    source=$BASE_SOURCE
    [ "$arm" = collapsed ] && source=$COLLAPSED_SOURCE
    tree=$(mirror "$arm" "$source")
    trace="$TRACE_ROOT/$arm"
    mkdir -p "$trace"
    printf 'ARM=%s REVISION=%s SLURM_JOB_ID=%s TREE=%s SOURCE=%s\n' "$arm" \
      "$(git -C "$ROOT" rev-parse HEAD)" "${SLURM_JOB_ID:-none}" "$tree" "$source"
    (
      cd "$tree" || exit 1
      export PYTHONPATH="$tree:$ROOT" JAX_PLATFORMS=cuda BANK_STEP_TRACE_DIR="$trace"
      "$PY" -P -c 'import nova, nova.equilibrium.current as c; print("NOVA="+nova.__file__+" CURRENT="+c.__file__)'
      "$PY" -m scripts.prototypes.bank_step_speed.profile || { printf 'PROFILE_FAILED arm=%s\n' "$arm"; exit 1; }
      trace_file=$(find "$trace" -name '*.trace.json.gz' | head -1)
      "$PY" -m scripts.prototypes.bank_step_speed.parse_trace "$trace_file"
    ) 2>&1
    printf 'EXIT_ARM=%s arm=%s\n' "$?" "$arm"
  done
  printf 'EXIT=0\n'
} > "$LOG" 2>&1