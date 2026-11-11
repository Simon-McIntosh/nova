#!/usr/bin/env bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
root=$1
run=$2
base=$3
head=$4
export TMPDIR=/tmp
export JAX_MEMORY_FITTING_LEVEL=O3 JAX_MEMORY_FITTING_EFFORT=1.0
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
scratch=$(mktemp -d "$TMPDIR/carried-timing-XXXXXX")
cleanup() {
    result=$?
    rm -rvf "$scratch"
    printf 'EXIT=%s\n' "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf '# H200, width 16 paired timing; job=%s\n' "$SLURM_JOB_ID"
printf '# memory fitting: level=%s effort=%s\n' "$JAX_MEMORY_FITTING_LEVEL" "$JAX_MEMORY_FITTING_EFFORT"
mkdir -p "$scratch/base" "$scratch/head"
git -C "$root" archive "$base" | tar -x -C "$scratch/base"
git -C "$root" archive "$head" | tar -x -C "$scratch/head"
mkdir -p "$scratch/base/scripts/prototypes/bank_carried_decision"
cp "$scratch/head/scripts/prototypes/bank_carried_decision/measure.py" "$scratch/base/scripts/prototypes/bank_carried_decision/measure.py"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
execute() {
    label=$1
    tree=$2
    revision=$3
    cache=$4
    shift 4
    (
        cd "$tree" || exit 2
        export PYTHONPATH="$tree" MEASURE_REVISION="$revision"
        printf '# revision: %s tree: %s wall_device: H200, width 16 command: %s -m scripts.prototypes.bank_carried_decision.measure %s\n' "$revision" "$tree" "$python" "$*"
        set +e
        bash "$scratch/head/scripts/prototypes/bank_booking_determinism/measurement.sh" "$cache" "$python" -m scripts.prototypes.bank_carried_decision.measure "$@"
        result=$?
        printf 'EXIT=%s\n' "$result"
        exit "$result"
    ) > "$run/$label.log" 2>&1
}
execute builder "$scratch/base" "$base" "$run/cache-base" --widths 16 --build --inputs "$run/common-inputs.npz" --output "$run/builder.json"
measure_width() {
    width=$1
    suffix=$2
    for process in warmup 0 1 2; do
        for arm in base head; do
            if [ "$arm" = base ]; then tree="$scratch/base"; revision=$base; else tree="$scratch/head"; revision=$head; fi
            extra=(--census)
            if [ "$process" = 1 ]; then extra=(--profile --census); fi
            execute "$arm$suffix-$process" "$tree" "$revision" "$run/cache-$arm" --widths "$width" --inputs "$run/common-inputs.npz" --output "$run/$arm$suffix-$process.json" "${extra[@]}"
        done
    done
}
measure_width 16 ''
printf 'MEASUREMENT_COMPLETE: builder and both revisions warmup,0,1,2 finished\n'
