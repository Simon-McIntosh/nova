#!/bin/bash
#SBATCH --job-name=merit-model
#SBATCH --partition=all_debug
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction/allocation.log
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
cd /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction
printf 'revision=%s tree=%s command=%s\n' "$(git rev-parse HEAD)" "$PWD" "measure.py --cells 110 then --cells 300"
failed=0
for cells in 110 300; do
  log=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction/cells-$cells.log
  printf 'revision=%s tree=%s command=%s\n' "$(git rev-parse HEAD)" "$PWD" "/home/ITER/mcintos/Code/nova/.venv/bin/python docs/figures/cut-cell-current-attribution/merit-model/measure.py --cells $cells --output /home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction" > "$log"
  /home/ITER/mcintos/Code/nova/.venv/bin/python -u docs/figures/cut-cell-current-attribution/merit-model/measure.py --cells "$cells" --output /home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-merit-model-refuses-the-analytic-direction >> "$log" 2>&1
  result=$?
  printf 'ROW_EXIT cells=%s status=%s\n' "$cells" "$result" >> "$log"
  printf 'ROW_EXIT cells=%s status=%s\n' "$cells" "$result"
  if [ "$result" -ne 0 ]; then failed=1; fi
done
printf 'GATE_EXIT=%s\n' "$failed"
exit "$failed"

