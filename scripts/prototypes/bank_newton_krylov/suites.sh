#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -u
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/bank-krylov-suites-XXXXXX")
finish() {
    result=$?
    rm -rvf "$scratch"
    printf 'EXIT=%s\n' "$result"
}
trap finish EXIT
export TMPDIR="$scratch" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
repo=$1
run=$2
base=$3
head=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$scratch/head"
git -C "$repo" archive "$base" nova scripts tests benchmarks pyproject.toml | tar -x -C "$scratch/base"
git -C "$repo" archive "$head" nova scripts tests benchmarks pyproject.toml | tar -x -C "$scratch/head"
run_arm() {
    arm=$1
    revision=$2
    tree="$scratch/$arm"
    outcome=0
    for name in ${TEST_FILES:-test_bank_newton_step test_bank_solve_cycle test_bank_solve_driver test_support_read test_bank_step_booking test_bank_step_booking_batch test_equilibrium_solve}; do
        log="$run/$arm-$name.log"
        command=("$python" -P -m pytest -p no:cacheprovider "--basetemp=$scratch/temp-$arm-$name" "$tree/tests/$name.py" -m "slow or not slow" -vv -s)
        printf -v command_text '%q ' "${command[@]}"
        printf '# revision: %s tree: %s command: %s\n' "$revision" "$tree" "$command_text" > "$log"
        PYTHONPATH="$tree" "$python" -P -c 'import nova, nova.equilibrium.support_read as m; from pathlib import Path; print("nova",nova.__file__,"module",m.__file__,"cwd",Path.cwd().resolve())' >> "$log" 2>&1
        PYTHONPATH="$tree" "${command[@]}" >> "$log" 2>&1
        code=$?
        printf 'EXIT=%s\n' "$code" >> "$log"
        if [ "$code" -ne 0 ]; then outcome=1; fi
    done
    return "$outcome"
}
base_pid=
if [ "${5:-both}" = both ]; then
    run_arm base "$base" &
    base_pid=$!
fi
run_arm head "$head" &
head_pid=$!
base_code=0
if [ -n "$base_pid" ]; then
    wait "$base_pid"
    base_code=$?
fi
wait "$head_pid"
head_code=$?
exit "$((base_code || head_code))"
