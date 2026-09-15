#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=profile-coefficients-lowering
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/profile-args/lowering-%j.log

set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/msc-profile-coefficients-recompile-audit
export HLO_RUN_DIRECTORY=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/profile-args
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

echo "STAGE allocation started job=$SLURM_JOB_ID host=$(hostname)"
/home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-review/msc-profile-coefficients-recompile-audit/benchmarks/profile_coefficients_recompile_audit.py
status=$?
echo "STAGE allocation finished status=$status"
exit "$status"
