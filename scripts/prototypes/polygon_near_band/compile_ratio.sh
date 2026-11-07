#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=polygon-compile-ratio
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/compile-ratio-job-%j.log
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/compile_ratio_arms.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --calls 3 --repeats 3 --log-root /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate
echo EXIT=$?