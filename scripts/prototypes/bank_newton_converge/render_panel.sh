#!/bin/bash
set -u
export TMPDIR=/tmp
ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT
export JAX_PLATFORMS=cpu
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG=${1:?log path required}
STATE=${2:?state path required}
SVG=${3:?figure path required}
PREVIEW=${4:?preview path required}
{
  printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY -m scripts.prototypes.bank_newton_converge.render_panel $STATE $SVG $PREVIEW"
  "$PY" -c 'import nova, nova.equilibrium.support_read as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_newton_converge.render_panel "$STATE" "$SVG" "$PREVIEW"
  code=$?
  printf 'EXIT=%s\n' "$code"
  exit "$code"
} > "$LOG" 2>&1
