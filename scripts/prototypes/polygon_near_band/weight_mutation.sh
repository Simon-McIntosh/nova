#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=polygon-weight-mutation
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261106T084812488886-polygon-near-band-quadrature/gate/weight-mutation-job-%j.log
export TMPDIR=/tmp
export RECKON_ALLOW_NESTED_LAUNCH=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export NOVA_TANGENT_TRUNCATION=1
cd "$SLURM_SUBMIT_DIR" || exit 2
echo "# command: NOVA_TANGENT_TRUNCATION=1 /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -v --timeout=3600 --basetemp \$TMPDIR/pytest-weight-mutation $SLURM_SUBMIT_DIR/tests/biot/test_edge_term_tangents.py::test_tangent_matches_base_jvp[edge_terms]"
/home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -p no:cacheprovider -v --timeout=3600 --basetemp "$TMPDIR/pytest-weight-mutation" "$SLURM_SUBMIT_DIR/tests/biot/test_edge_term_tangents.py::test_tangent_matches_base_jvp[edge_terms]"
echo EXIT=$?