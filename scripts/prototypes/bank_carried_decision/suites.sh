#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
root=$1
run=$2
base=$3
scratch=$(mktemp -d "$TMPDIR/carried-suite-XXXXXX")
trap 'rm -rvf "$scratch"' EXIT
mkdir -p "$scratch/base"
git -C "$root" archive "$base" | tar -x -C "$scratch/base"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
run_file() {
    arm=$1
    tree=$2
    file=$3
    name=${file##*/}
    log="$run/$arm-${name%.py}.log"
    extra=()
    if [ "$name" = test_candidate_inspection_batched.py ]; then extra=(-m "not slow"); fi
    command="PYTHONPATH=. JAX_PLATFORMS=cpu $python -m pytest -p no:cacheprovider -q $file ${extra[*]}"
    (
        cd "$tree" || exit 2
        if [ "$arm" = base ]; then revision=$base; else revision=$(git -C "$root" rev-parse HEAD); fi
        printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command"
        export PYTHONPATH="$tree" PYTEST_ADDOPTS="--basetemp=$scratch/$arm-${name%.py}"
        "$python" -c 'import os,nova; from nova.equilibrium import support_read; print("nova",nova.__file__,"module",support_read.__file__,"cwd",os.getcwd(),flush=True)'
        "$python" -m pytest -p no:cacheprovider -q "$file" "${extra[@]}"
        code=$?
        printf 'EXIT=%s\n' "$code"
        exit "$code"
    ) > "$log" 2>&1
}
# One fresh process per file; arms share this one scheduler job.
files=${4:-"tests/test_tracked_read.py tests/test_support_read.py tests/test_bank_solve_driver.py tests/test_bank_solve_cycle.py tests/test_candidate_inspection_batched.py"}
result=0
for file in $files; do
    run_file base "$scratch/base" "$file" &
    base_pid=$!
    run_file HEAD "$root" "$file" &
    head_pid=$!
    wait "$base_pid" || result=1
    wait "$head_pid" || result=1
done
printf 'EXIT=%s\n' "$result"
exit "$result"
