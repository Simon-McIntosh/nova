#!/usr/bin/env bash
set -u
export TMPDIR=/tmp NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T095720496569-bank-tangent-contract
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=$(mktemp -d "$TMPDIR/tangent-suite-XXXX")
trap 'rm -rv "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/base/nova/equilibrium"
for path in "$ROOT"/nova/*; do
  [ "$(basename "$path")" = equilibrium ] || ln -s "$path" "$SCRATCH/base/nova/"
done
for path in "$ROOT"/nova/equilibrium/*; do
  [ "$(basename "$path")" = current.py ] || ln -s "$path" "$SCRATCH/base/nova/equilibrium/"
done
git -C "$ROOT" show 8ff5dae2a49a63fcfac9f3dd9dea53a9c8744626:nova/equilibrium/current.py > "$SCRATCH/base/nova/equilibrium/current.py"
ln -s "$ROOT/scripts" "$SCRATCH/base/scripts"
ln -s "$ROOT/tests" "$SCRATCH/base/tests"
for arm in base head; do
  if [ "$arm" = base ]; then TREE="$SCRATCH/base"; REV=8ff5dae2a49a63fcfac9f3dd9dea53a9c8744626; else TREE="$ROOT"; REV=$(git -C "$ROOT" rev-parse HEAD); fi
  LOG="$RUN/suite-$arm.log"
  (
    cd "$TREE" || exit 1
    export PYTHONPATH="$TREE:$ROOT" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
    printf '# revision: %s tree: %s cwd: %s command: %s\n' "$REV" "$TREE" "$PWD" "$PY -m pytest -q -p no:cacheprovider tests/test_bank_tangent_contract.py --basetemp=$SCRATCH/basetemp-$arm"
    "$PY" -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
    "$PY" -m pytest -q -p no:cacheprovider tests/test_bank_tangent_contract.py --basetemp="$SCRATCH/basetemp-$arm"
  ) > "$LOG" 2>&1
  code=$?
  printf 'EXIT=%s\n' "$code" >> "$LOG"
done
