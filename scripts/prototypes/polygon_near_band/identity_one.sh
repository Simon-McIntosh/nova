#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=polygon-identity-one
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/identity-split/%x-%j.log
# One identity row at one arm, so the six rows of an arm run side by side
# instead of behind each other's half-hour.
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
arm="$1"
row="$2"
tag=$(printf '%s' "$row" | tr -c 'a-zA-Z0-9' '-')
log_root="/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/identity-split/$arm-$tag"
mkdir -p "$log_root"
/home/ITER/mcintos/Code/nova/.venv/bin/python "$SLURM_SUBMIT_DIR/scripts/prototypes/polygon_near_band/identity_gate.py" --base-revision cb42f656f76a7d7e56da8a53d4754c22d131bdc0 --arm "$arm" --only "tests/biot/test_edge_term_tangents.py::$row" --timeout 3600 --log-root "$log_root"
echo EXIT=$?