#!/bin/bash
# Submit this script as a separate CPU allocation.
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/hessian-diagnosis-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf '# revision: %s tree: %s command: %s -m scripts.prototypes.bank_hessian_diagnosis.measure' "$(git rev-parse HEAD)" "$SLURM_SUBMIT_DIR" "$python"
printf ' %q' "$@"
printf '\n'
"$python" -m scripts.prototypes.bank_hessian_diagnosis.measure "$@"
result=$?
rm -rvf "$scratch"
trap - EXIT
printf 'EXIT=%s\n' "$result"
exit "$result"
