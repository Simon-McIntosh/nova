#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=01:04:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=polygon-identity-rows
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/identity-rows-%j.log
# Twelve processes in one allocation: the six rows at each of the two
# revisions, each with its own log and basetemp.
export TMPDIR=/tmp
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/identity_rows.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --log-root /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/identity-rows --timeout 3600 --workers 12
echo EXIT=$?