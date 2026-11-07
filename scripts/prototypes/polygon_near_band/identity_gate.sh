#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=polygon-identity
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/identity-job-%j.log
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/identity_gate.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --timeout 3600 --log-root /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate
echo EXIT=$?