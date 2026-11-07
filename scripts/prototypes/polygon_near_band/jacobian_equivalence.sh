#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=polygon-jacobian-equivalence
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/jacobian-equivalence-%j.log
export TMPDIR=/tmp
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export JAX_PLATFORMS=cpu
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/jacobian_equivalence.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --output /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/jacobian-equivalence.json
echo EXIT=$?