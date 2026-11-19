#!/bin/bash
# Run the polygon symmetry probe and capture a gate-shaped log.
#
# usage: run_probe.sh <log-path> [extra probe.py arguments...]
#
# The header names the tree the payload imports, so a log from a --tree arm does
# not claim the working tree's module.
set -u
log=$1
shift
python=/home/ITER/mcintos/Code/nova/.venv/bin/python

tree=.
expect_tree=0
for argument in "$@"; do
  if [ "$expect_tree" = 1 ]; then
    tree=$argument
    expect_tree=0
  fi
  if [ "$argument" = "--tree" ]; then
    expect_tree=1
  fi
done

{
  echo "# revision: $(git rev-parse HEAD)"
  echo "# cwd: $PWD"
  echo "# tree under test: $tree"
  echo "# command: JAX_PLATFORMS=cpu PYTHONPATH=$tree $python scripts/prototypes/polygon_symmetry_violation/probe.py $*"
} > "$log"

PYTHONPATH="$tree" "$python" -c 'import nova.biot.polygonanalytic as m, os
print("# module.__file__:", m.__file__)
print("# resolved cwd:", os.getcwd())' >> "$log" 2>&1

JAX_PLATFORMS=cpu PYTHONPATH="$tree" "$python" scripts/prototypes/polygon_symmetry_violation/probe.py "$@" >> "$log" 2>&1
status=$?
echo "EXIT=$status" >> "$log"
exit $status