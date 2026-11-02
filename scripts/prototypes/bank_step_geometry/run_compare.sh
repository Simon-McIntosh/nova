#!/usr/bin/env bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -u
export TMPDIR=/tmp JAX_PLATFORMS=cuda
export PYTHONPATH=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
export BANK_CONVERGED_STATE=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261029T223038646111-bank-step-speed/converged-booking.npz
log="${1:?log path required}"
module="${2:-scripts.prototypes.bank_step_geometry.compare}"
if [ "$#" -ge 2 ]; then shift 2; else shift 1; fi
{
    printf 'revision=%s tree=%s command=%s\n' \
        "$(git -C "$PYTHONPATH" rev-parse HEAD)" "$PYTHONPATH" \
        "/home/ITER/mcintos/Code/nova/.venv/bin/python -m $module $*"
    /home/ITER/mcintos/Code/nova/.venv/bin/python -m "$module" "$@"
    result=$?
    echo "EXIT=$result"
} > "$log" 2>&1
exit "$result"
