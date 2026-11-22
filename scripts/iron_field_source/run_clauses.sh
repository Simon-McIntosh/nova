#!/bin/bash
#SBATCH --job-name=icm-clauses
#SBATCH --partition=rigel
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
set -eu
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONPATH="$ICM_ROOT"
export PYTHONUNBUFFERED=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true
cd "$ICM_ROOT"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
"$python" scripts/iron_field_source/measure.py state uniform-0.08
"$python" scripts/iron_field_source/measure.py field uniform-0.08
"$python" scripts/iron_field_source/sampled.py uniform-0.08
echo "CLAUSES-COMPLETE $(date -u +%FT%TZ)"