#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=exterior-lowering
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior/lowering-%j.log

script_path="$(realpath -e -- "${BASH_SOURCE[0]}")"
repository_root="$(git -C "$(dirname "${script_path}")" rev-parse --show-toplevel)"

set -uo pipefail
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH=${repository_root}
export HLO_RUN_DIRECTORY=/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/traced-exterior
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH

echo "STAGE allocation started job=$SLURM_JOB_ID host=$(hostname)"
/home/ITER/mcintos/Code/nova/.venv/bin/python -u \
  ${repository_root}/docs/figures/millisecond-converged-solve/traced-exterior/run_lowering.py
status=$?
echo "STAGE allocation finished status=$status"
exit "$status"
