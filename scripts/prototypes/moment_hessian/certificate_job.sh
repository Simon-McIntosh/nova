#!/bin/bash
set -uo pipefail

export TMPDIR=/tmp JAX_PLATFORMS=cpu
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
tree=$1
revision=$2
run=$3
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export PYTHONPATH="$tree"
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_BOOK_FIXTURE_ROOT="$NOVA_OPERAND_CACHE/certificate-fixtures"
export NOVA_COMPILATION_CACHE_ROOT="$NOVA_OPERAND_CACHE/compile-cache"
mkdir -p "$NOVA_OPERAND_CACHE" "$run/certificate-row"
command=("$python" -u "$tree/scripts/prototypes/moment_hessian/certificate_row.py" --output "$run/certificate-row/row.json")
(
    cd "$tree" || exit 2
    printf 'revision=%s tree=%s command=' "$revision" "$tree"
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$run/certificate-row/row.log" 2>&1
exit $?
