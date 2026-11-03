#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T063317710937-bank-step-cut-buckets
BASE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261102T215004202991-bank-step-booking-once/current-cond-preserved.py
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
SCRATCH=$(mktemp -d "$TMPDIR/cut-control-XXXX")
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/nova/equilibrium"
for path in "$ROOT"/nova/*; do
    [ "$(basename "$path")" = equilibrium ] || ln -s "$path" "$SCRATCH/nova/"
done
for path in "$ROOT"/nova/equilibrium/*; do
    [ "$(basename "$path")" = current.py ] || ln -s "$path" "$SCRATCH/nova/equilibrium/"
done
cp "$BASE" "$SCRATCH/nova/equilibrium/current.py"
ln -s "$ROOT/tests" "$SCRATCH/tests"
cd "$SCRATCH" || exit 1
export PYTHONPATH="$SCRATCH:$ROOT" JAX_PLATFORMS=cpu
LOG="$RUN/negative-control-sirius.log"
{
    printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
        "2d2172dd7b06548d06c634fe8b3def7acd88f9b4" "$SCRATCH" "$PWD" \
        "$PY -m pytest -p no:cacheprovider tests/test_bank_step_cut_buckets.py::test_cell_vmap_has_no_per_cell_branch -q"
    printf 'SUBMITTING_REVISION=%s SOURCE_SHA256=%s\n' \
        "$(git -C "$ROOT" rev-parse HEAD)" "$(sha256sum "$BASE" | cut -d' ' -f1)"
    "$PY" -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
    set +e
    "$PY" -m pytest -p no:cacheprovider \
        tests/test_bank_step_cut_buckets.py::test_cell_vmap_has_no_per_cell_branch \
        -q --basetemp="$SCRATCH/basetemp"
    code=$?
    set -e
    printf 'EXIT=%s\n' "$code"
    exit "$code"
} > "$LOG" 2>&1
