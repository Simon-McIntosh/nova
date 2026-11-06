#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -uo pipefail
export TMPDIR=/tmp
repo=$1
shift
/home/ITER/mcintos/Code/nova/.venv/bin/python "$repo/scripts/prototypes/bank_saddle_guards/suite.py" --repo "$repo" "$@"
result=$?
echo "EXIT=$result"
exit "$result"
