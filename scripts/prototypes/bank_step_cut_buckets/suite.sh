#!/usr/bin/env bash
set -u

ROOT=$(git rev-parse --show-toplevel)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T063317710937-bank-step-cut-buckets
BASE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261102T215004202991-bank-step-booking-once/current-cond-preserved.py
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp JAX_PLATFORMS=cpu
SCRATCH=$(mktemp -d "$TMPDIR/cut-suite-XXXX")
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/nova/equilibrium"
for path in "$ROOT"/nova/*; do
    [ "$(basename "$path")" = equilibrium ] || ln -s "$path" "$SCRATCH/nova/"
done
for path in "$ROOT"/nova/equilibrium/*; do
    [ "$(basename "$path")" = current.py ] || ln -s "$path" "$SCRATCH/nova/equilibrium/"
done
cp "$BASE" "$SCRATCH/nova/equilibrium/current.py"
for name in scripts tests data input benchmarks; do
    [ -e "$ROOT/$name" ] && ln -s "$ROOT/$name" "$SCRATCH/$name"
done

run_file() {
    local arm=$1 tree=$2 file=$3 log="$RUN/suite-$1-$4.log" temp="$SCRATCH/temp-$1-$4"
    (
        cd "$tree" || exit 1
        export PYTHONPATH="$tree:$ROOT"
        printf 'REVISION=%s TREE=%s CWD=%s COMMAND=%s\n' \
            "$(git -C "$ROOT" rev-parse HEAD)" "$tree" "$PWD" \
            "$PY -m pytest -p no:cacheprovider $file -q --basetemp=$temp"
        "$PY" -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
        "$PY" -m pytest -p no:cacheprovider "$file" -q --basetemp="$temp"
        code=$?
        printf 'EXIT=%s\n' "$code"
        exit "$code"
    ) > "$log" 2>&1
}

run_file base "$SCRATCH" tests/test_bank_step_booking.py booking
base_code=$?
run_file head "$ROOT" tests/test_bank_step_booking.py booking
head_code=$?
run_file head "$ROOT" tests/test_bank_step_cut_buckets.py buckets
bucket_code=$?
if [ "$base_code" -ne 0 ] || [ "$head_code" -ne 0 ] || [ "$bucket_code" -ne 0 ]; then
    exit 1
fi
