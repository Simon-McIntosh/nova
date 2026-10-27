#!/usr/bin/env bash
# One fresh pytest process for one test file, on the CPU lane.
#
# usage: cpu_arm.sh <log> <target> [pytest args...]
set -euo pipefail

LOG=$1
TARGET=$2
shift 2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
TREE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export TMPDIR="${TMPDIR:-/tmp}"
export JAX_PLATFORMS=cpu
export PYTHONPATH="$TREE"
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true
BASE=$(mktemp -d "$TMPDIR/etb-basetemp-XXXXXX")
trap 'rm -rf "$BASE"' EXIT

{
  printf '# revision: %s\n' "$(git -C "$TREE" rev-parse HEAD)"
  printf '# tree: %s\n' "$TREE"
  printf '# command: %s\n' "$PY -m pytest -p no:cacheprovider -m 'slow or not slow' -q --tb=short --basetemp=$BASE $TARGET $*"
  "$PY" -c 'import nova, nova.biot.polygonanalytic as p; print("nova=" + nova.__file__); print("measurement_module=" + p.__file__)'
  cd "$TREE"
  set +e
  "$PY" -m pytest -p no:cacheprovider -m 'slow or not slow' -q --tb=short \
    --basetemp="$BASE" "$TARGET" "$@" 2>&1
  code=$?
  set -e
  printf 'EXIT=%s\n' "$code"
} >"$LOG" 2>&1