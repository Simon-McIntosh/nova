#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=01:04:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=polygon-weighted-files
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/weighted-files-%j.log
# Six processes in one allocation: two files at two revisions plus the
# truncation mutation of the vertex primal-JVP row.
export TMPDIR=/tmp
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/weighted_files_gate.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --log-root /home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/weighted-files --timeout 3600 --workers 6
echo EXIT=$?