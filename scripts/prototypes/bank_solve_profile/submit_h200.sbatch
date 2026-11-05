#!/usr/bin/env bash
# Submit one bank-solve receipt to the reserved H200.
#
#   SCRIPT=measure sbatch --output=<log> scripts/prototypes/bank_solve_profile/submit_h200.sbatch
#
# SCRIPT selects measure.py, fused.py or marginal.py; REPEATS passes through to
# the receipt. The job body is run_h200.sh beside this file, so the wrapper that
# produces a log is the wrapper in the tree that produced it.
#SBATCH --job-name=bank-solve-profile
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
export TMPDIR=/tmp
export SCRIPT=${SCRIPT:-measure}
# SLURM runs a copy of this file from /var/spool, so the tree comes from the
# directory sbatch was invoked in (the repository root), never from this path.
export NOVA_ROOT=${NOVA_ROOT:-$SLURM_SUBMIT_DIR}
exec bash "$NOVA_ROOT/scripts/prototypes/bank_solve_profile/run_h200.sh"