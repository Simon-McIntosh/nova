#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
repo=$PWD
run=${BANK_BOOKING_RUN_DIR:?}
scratch=$(mktemp -d "$TMPDIR/bank-booking-diagnostic-XXXXXX")
finish() {
    result=$?
    rm -rf "$scratch"
    echo "EXIT=$result"
}
trap finish EXIT
printf 'REVISION=%s TREE=%s CWD=%s\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$PWD"
printf 'COMMAND=%s\n' '/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/bank_step_booking/diagnose.py'
git -C "$repo" archive HEAD nova tests | tar -x -C "$scratch"
if ! rg -q '^def book_tangent_channels\(' "$scratch/nova/equilibrium/current.py"; then
    patch --directory "$scratch" --strip 1 --input "$run/failed-vectorized-tangent.patch"
fi
export PYTHONPATH="$scratch"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$repo/scripts/prototypes/bank_step_booking/diagnose.py"
