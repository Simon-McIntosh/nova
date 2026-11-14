#!/bin/bash
# Validate retained measurements and render the one evidence figure.
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/hessian-verify-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
run=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T170651657652-bank-hessian-diagnosis
execute() {
    local log=$1
    shift
    printf '# revision: %s tree: %s command:' "$(git rev-parse HEAD)" "$SLURM_SUBMIT_DIR" > "$log"
    printf ' %q' "$@" >> "$log"
    printf '\n' >> "$log"
    "$@" >> "$log" 2>&1
    local result=$?
    printf 'EXIT=%s\n' "$result" >> "$log"
    return "$result"
}
execute "$run/report-positive.log" "$python" -m scripts.prototypes.bank_hessian_diagnosis.report || exit $?
execute "$run/report-negative.log" "$python" -m scripts.prototypes.bank_hessian_diagnosis.report --mutate
negative=$?
if [ "$negative" -ne 1 ]; then
    printf 'Unexpected mutation exit %s\n' "$negative"
    exit 1
fi
execute "$run/figure.log" "$python" -m scripts.prototypes.bank_hessian_diagnosis.plot || exit $?
rm -rvf "$scratch"
trap - EXIT
printf 'VERIFY_COMPLETE\nEXIT=0\n'
