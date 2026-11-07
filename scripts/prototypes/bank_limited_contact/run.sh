#!/usr/bin/env bash
set -uo pipefail
log=$1
shift
root=$(git rev-parse --show-toplevel)
revision=$(git rev-parse HEAD)
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$root" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
command=("$python" -m scripts.prototypes.bank_limited_contact.measure "$@")
started=$SECONDS
{
    printf 'REVISION=%s TREE=%s COMMAND=' "$revision" "$root"
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
    result=$?
    printf 'PROCESS_WALL_SECONDS=%s HOST=%s\n' "$((SECONDS-started))" "$(hostname)"
    printf 'EXIT=%s\n' "$result"
} > "$log" 2>&1
exit "$result"
