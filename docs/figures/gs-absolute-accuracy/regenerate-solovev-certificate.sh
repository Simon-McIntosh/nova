#!/usr/bin/env bash
#SBATCH --job-name=nova-solovev-certificate
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --output=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/nia-certificate-raster-receipt/docs/figures/gs-absolute-accuracy/solovev-certificate-h200-%j.log

set -euo pipefail
export TMPDIR=/tmp
export PYTHONPATH=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/nia-certificate-raster-receipt
export JAX_PLATFORMS=cuda
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-labeller/nia-certificate-raster-receipt/benchmarks/solovev_certificate.py

for case_name in \
    weak-rotation-reactor-static \
    moderate-rotation-conventional-static \
    strong-rotation-compact-static \
    diverted-jump-bearing
do
    for requested_cells in -110 -300 -500 -1000
    do
        "$PYTHON" "$DRIVER" --case "$case_name" --requested-cells "$requested_cells"
    done
done

"$PYTHON" "$DRIVER" --aggregate --scheduler-job-id "$SLURM_JOB_ID"
