#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
root=$1
output=$2
revision=$3
shift 3
scratch=$(mktemp -d "$TMPDIR/h200-driver-XXXXXX")
cleanup() {
    code=$?
    printf 'Removing job-owned driver scratch: %s\n' "$scratch"
    rm -rvf "$scratch"
    printf 'EXIT=%s\n' "$code"
}
trap cleanup EXIT
printf '# revision: %s tree: %s command: python run.py %s %s --revision %s %s\n' "$revision" "$root" "$root" "$output" "$revision" "$*"
git -C "$root" show "$revision:scripts/prototypes/h200_levers/run.py" > "$scratch/run.py"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$scratch/run.py" "$root" "$output" --revision "$revision" "$@"
