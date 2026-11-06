#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=polygon-near-band
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/job-%j.log
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/run_gate.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --log-root /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate
