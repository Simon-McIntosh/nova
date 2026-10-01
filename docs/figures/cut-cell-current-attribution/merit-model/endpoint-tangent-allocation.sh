#!/bin/bash
#SBATCH --job-name=tangent-endpoints
#SBATCH --partition=all_debug
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-342-cell-terminal-fails-the-tangent-check/scheduler.log
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-why-the-342-cell-terminal-fails-the-tangent-check
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
REPORT=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-342-cell-terminal-fails-the-tangent-check
cd "$PYTHONPATH"
log=$REPORT/endpoint-tangent-300.log
printf 'revision=%s tree=%s command=%s\n' "$(git rev-parse HEAD)" "$PWD" \
  "/home/ITER/mcintos/Code/nova/.venv/bin/python -u docs/figures/cut-cell-current-attribution/merit-model/endpoint_tangent.py --cells 300 --output $REPORT" > "$log"
/home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  docs/figures/cut-cell-current-attribution/merit-model/endpoint_tangent.py \
  --cells 300 --output "$REPORT" >> "$log" 2>&1
result=$?
printf 'ROW_EXIT cells=300 status=%s\n' "$result" >> "$log"
printf 'GATE_EXIT=%s\n' "$result"
exit "$result"