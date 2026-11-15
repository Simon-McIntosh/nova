#!/bin/bash
#SBATCH --partition=rigel
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:25:00
set -eu
export TMPDIR=/tmp JAX_PLATFORMS=cpu IRON_MEASUREMENT_THREADS=16
exec bash "$PWD/scripts/iron_linear_magnetisation/run.sh" "$@"
