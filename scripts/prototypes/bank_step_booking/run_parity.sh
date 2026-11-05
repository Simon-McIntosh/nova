#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH="$PWD"
scratch=$(mktemp -d "$TMPDIR/bank-booking-parity-XXXXXX")
printf 'REVISION=%s TREE=%s CWD=%s\n' "$(git rev-parse HEAD)" "$PWD" "$PWD"
/home/ITER/mcintos/Code/nova/.venv/bin/python -P -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import nova; print("NOVA_FILE", nova.__file__)'
command=(/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider tests/test_bank_step_booking.py::test_batched_booking_tangent_matches_single_channel_reference --basetemp="$scratch" -vv)
printf 'COMMAND=%s\n' "${command[*]}"
"${command[@]}"
result=$?
rm -rf "$scratch"
echo "EXIT=$result"
exit "$result"
