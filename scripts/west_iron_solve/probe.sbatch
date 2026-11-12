#!/usr/bin/env bash
#SBATCH --partition=sirius
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=west-mesh-probe
set -euo pipefail
repo="$1"
output="$2"
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH="$repo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d "$TMPDIR/west-mesh-${SLURM_JOB_ID}-XXXXXX")
finish() {
  code=$?
  rm -rvf "$scratch"
  printf 'EXIT=%s\n' "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export TMPDIR="$scratch"
printf '# revision: %s tree: %s command: bash scripts/west_iron_solve/probe.sbatch %q %q\n' "$(git -C "$repo" rev-parse HEAD)" "$repo" "$repo" "$output"
/home/ITER/mcintos/Code/nova/.venv/bin/python -u "$repo/scripts/west_iron_solve/probe.py" --output "$output"
