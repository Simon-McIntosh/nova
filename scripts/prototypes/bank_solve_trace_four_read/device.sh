#!/usr/bin/env bash
#SBATCH --job-name=solve-trace
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
set -u
ROOT=$1
RUN=$2
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf '# revision: %s tree: %s command: PYTHONPATH=%s %s -P -m scripts.prototypes.bank_solve_trace_four_read.device %s %s\n' "$(git -C "$ROOT" rev-parse HEAD)" "$ROOT" "$ROOT" "$PY" "$ROOT" "$RUN"
export TMPDIR=/tmp PYTHONPATH=$ROOT PYTHONDONTWRITEBYTECODE=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH XLA_FLAGS
"$PY" -P -m scripts.prototypes.bank_solve_trace_four_read.device "$ROOT" "$RUN"
result=$?
printf 'EXIT=%s\n' "$result"
exit "$result"
