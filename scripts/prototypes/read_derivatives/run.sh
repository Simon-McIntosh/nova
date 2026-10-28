#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=read-derivatives
export TMPDIR=/tmp
# slurm runs a copy of this script outside the repository, so the worktree
W=${1:?usage: run.sh worktree-root}
export PYTHONPATH=$W
L=$W/scripts/prototypes/read_derivatives/logs
PY=${NOVA_PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
for kind in limited diverted; do
  $PY $W/scripts/prototypes/read_derivatives/measure.py --kind $kind --cells 550 --output $L/$kind-550.json >> $L/$kind-550.log 2>&1
  echo "EXIT=$? kind=$kind" >> $L/$kind-550.log
done
