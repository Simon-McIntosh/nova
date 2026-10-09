#!/usr/bin/env bash
set -u
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
root=$(pwd -P)
export PYTHONPATH="$root"
output=$1
cache=$2
control=$3
revision=$(git -C "$root" rev-parse HEAD)
export NOVA_MEASUREMENT_REVISION="$revision"
printf 'REVISION=%s TREE=%s COMMAND=bash %s %s %s %s\n' "$revision" "$root" "$0" "$output" "$cache" "$control"
mkdir -p "$output/rows" "$output/logs"
run_row() {
    kind=$1
    cells=$2
    log="$output/logs/$kind-$cells-$SLURM_JOB_ID.log"
    printf 'REVISION=%s TREE=%s COMMAND=/home/ITER/mcintos/Code/nova/.venv/bin/python %s/scripts/prototypes/support/measure.py --kind %s --cells %s --out %s/rows --cache %s --control %s\n' "$revision" "$root" "$root" "$kind" "$cells" "$output" "$cache" "$control" > "$log"
    /usr/bin/timeout 3600 /home/ITER/mcintos/Code/nova/.venv/bin/python -u "$root/scripts/prototypes/support/measure.py" --kind "$kind" --cells "$cells" --out "$output/rows" --cache "$cache" --control "$control" >> "$log" 2>&1
    code=$?
    printf 'EXIT=%s\n' "$code" >> "$log"
    printf 'ROW=%s/%s EXIT=%s LOG=%s\n' "$kind" "$cells" "$code" "$log"
    return "$code"
}
run_row diverted 550 || exit $?
run_row limited 550 || exit $?
run_row diverted 2000 || exit $?
run_row limited 2000 || exit $?
run_large() {
    kind=$1
    run_row "$kind" 5000
    code=$?
    if [ "$code" -eq 124 ] && ! grep -q 'STAGE_DONE machine' "$output/logs/$kind-5000-$SLURM_JOB_ID.log"; then
        printf 'MACHINE_BUDGET case=%s requested=5000 wall_limit=3600 fallback=3500\n' "$kind"
        run_row "$kind" 3500
        return $?
    fi
    return "$code"
}
run_large diverted || exit $?
run_large limited || exit $?
