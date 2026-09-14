#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=exterior-lowering
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior/lowering-%j.log

set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/msc-traced-exterior-one-compile-per-mesh
export HLO_RUN_DIRECTORY=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

echo "STAGE allocation started job=$SLURM_JOB_ID host=$(hostname)"
/home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/msc-traced-exterior-one-compile-per-mesh/docs/figures/millisecond-converged-solve/traced-exterior/run_lowering.py
status=$?
echo "STAGE allocation finished status=$status"
exit "$status"
