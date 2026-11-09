#!/usr/bin/env bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -uo pipefail
root=$1
run=$2
base=$3
export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
scratch=$(mktemp -d "$TMPDIR/carried-timing-XXXXXX") || exit 2
trap 'rm -rvf "$scratch"' EXIT
mkdir -p "$scratch/base"
git -C "$root" archive "$base" | tar -x -C "$scratch/base"
mkdir -p "$scratch/base/scripts/prototypes/bank_carried_decision"
cp "$root/scripts/prototypes/bank_carried_decision/measure.py" "$scratch/base/scripts/prototypes/bank_carried_decision/measure.py"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
head=$(git -C "$root" rev-parse HEAD)
execute() {
    label=$1
    tree=$2
    revision=$3
    cache=$4
    shift 4
    (
        cd "$tree" || exit 2
        export PYTHONPATH="$tree" MEASURE_REVISION="$revision"
        printf '# revision: %s tree: %s command: %s -m scripts.prototypes.bank_carried_decision.measure %s\n' "$revision" "$tree" "$python" "$*"
        bash "$root/scripts/prototypes/bank_booking_determinism/measurement.sh" "$cache" "$python" -m scripts.prototypes.bank_carried_decision.measure "$@"
        result=$?
        printf 'EXIT=%s\n' "$result"
        exit "$result"
    ) > "$run/$label.log" 2>&1
}
execute builder "$scratch/base" "$base" "$run/cache-base" --build --inputs "$run/common-inputs.npz" --output "$run/builder.json" || exit 1
for process in 0 1 2; do
    for arm in base head; do
        if [ "$arm" = base ]; then tree="$scratch/base"; revision=$base; else tree=$root; revision=$head; fi
        extra=(--census)
        if [ "$process" = 1 ]; then extra=(--profile --census); fi
        execute "$arm-$process" "$tree" "$revision" "$run/cache-$arm" --inputs "$run/common-inputs.npz" --output "$run/$arm-$process.json" "${extra[@]}" || exit 1
    done
done
printf 'EXIT=0\n'
