#!/bin/bash
set -eu
repo=$1
output=$2
revision=$3
extra=("${@:4}")
test "$(git -C "$repo" rev-parse HEAD)" = "$revision"
export TMPDIR=/tmp JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/nova/jax-cache/nova-prewarm
export PYTHONPATH="$repo"
export NOVA_HESSIAN_MEASURE_REVISION="$revision"
scratch=$(mktemp -d "$TMPDIR/hessian-${SLURM_JOB_ID:-$$}-XXXX")
trap 'rm -rvf "$scratch"' EXIT
export TMPDIR="$scratch"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
run_one() {
    name=$1
    module=$2
    shift 2
    if [ "$module" != pytest ]; then module="scripts.prototypes.bank_hessian_boundary.$module"; fi
    command=(env "PYTHONPATH=$repo" JAX_PLATFORMS=cpu "$python" -m "$module" "$@")
    {
        printf '# revision: %s; tree: %s; command: ' "$revision" "$repo"
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '# job: %s; host: %s\n' "${SLURM_JOB_ID:-none}" "$(hostname)"
        if "${command[@]}"; then result=0; else result=$?; fi
        echo "EXIT=$result"
    } > "$output/$name.log" 2>&1
    return "$result"
}
run_one regression pytest -p no:cacheprovider -q "$repo/scripts/prototypes/bank_hessian_boundary/test_measurement.py"
run_one split inspect
if [ "${extra[0]:-}" != --reuse-limits ]; then
    run_one limits limits --workers 16 "${extra[@]}"
fi
run_one positive report
set +e
run_one negative report --mutate
negative_status=$?
set -e
test "$negative_status" = 1
fragment="$repo/docs/evidence/fragments/sub-millisecond-forward-path/bank-hessian-boundary.html"
run_one document document "$fragment"
run_one document-check document "$fragment" --check
echo 'MEASURE_COMPLETE; positive accepted; deliberate interior perturbation refused'
