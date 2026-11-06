#!/usr/bin/env bash
#SBATCH --job-name=bank-solve-trace
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -u
ROOT=$1
RUN=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=PYTHONPATH=%s %s -P %s/scripts/prototypes/bank_solve_trace/capture.py --run %s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY" "$ROOT" "$RUN"
export TMPDIR=/tmp
TASK_TEMP=$(mktemp -d "$TMPDIR/bank-trace-XXXXXX")
finish() {
    result=$?
    rm -rvf "$TASK_TEMP"
    printf 'EXIT=%s\n' "$result"
}
trap finish EXIT
export TMPDIR=$TASK_TEMP PYTHONPATH=$ROOT JAX_PLATFORMS=cuda MPLBACKEND=Agg
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
"$PY" -P "$ROOT/scripts/prototypes/bank_solve_trace/capture.py" --run "$RUN"
exit "$?"
