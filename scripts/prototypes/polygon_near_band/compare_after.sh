#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=polygon-compare-after
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/compare-after-job-%j.log
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$SLURM_SUBMIT_DIR"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/measure.py" --device cpu --compare --baseline-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --output /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/compare-after.json
echo EXIT=$?