#!/bin/bash
# Run one CPU arm of the node's suite, recording revision, tree, module and command.
#
# usage: run_suite_arm.sh <tree> <name>
# Env: OUT (log root), TESTS (space-separated test paths), MEM, TIME (sbatch limits).
set -u

tree=${1:?tree}
name=${2:?name}
rev=${REV:-}
out=${OUT:-/home/ITER/mcintos/.cache/nova-kernel-compile-barrier/bounded-jets-build}
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
tests=${TESTS:-tests/test_fixed_points.py}
command="PYTHONPATH=. JAX_PLATFORMS=cpu $python -m pytest -p no:cacheprovider -q $tests"

mkdir -p "$out"

payload=$(cat <<PAYLOAD
cd $tree
export TMPDIR=/tmp
echo "revision=${rev:-\$(git -C $tree rev-parse HEAD 2>/dev/null)}"
echo "tree=\$(pwd -P)"
PYTHONPATH=. JAX_PLATFORMS=cpu $python -c "import nova.equilibrium.fixed_points as m, os; print('module=' + m.__file__); print('cwd=' + os.getcwd())"
echo "# command: $command"
$command
echo EXIT=\$?
PAYLOAD
)

export TMPDIR=/tmp
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
env -u SLURM_JOB_ID -u SLURM_JOBID sbatch \
  --partition=sirius \
  --account=iter \
  --cpus-per-task=4 \
  --mem="${MEM:-192G}" \
  --time="${TIME:-02:00:00}" \
  --job-name="nova-bounded-jets-$name" \
  --output="$out/$name-%j.log" \
  --wrap "$payload"