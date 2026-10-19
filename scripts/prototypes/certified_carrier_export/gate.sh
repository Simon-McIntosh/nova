#!/usr/bin/env bash
# Run the certified-carrier-export round-trip gate in one fresh process.
#
# Usage:  gate.sh <log-path> [negate-psi]
#
# The optional second argument names the negative-control mutation, which
# corrupts the flux grid on its way to disk so the round-trip test has a state
# it must refuse. The log always opens with the revision, the tree, the date
# and the exact command, and always ends with an EXIT line, so a log can be
# read for its verdict without trusting the caller's exit status.
set -u

log=${1:?usage: gate.sh <log-path> [negate-psi]}
mutation=${2:-}
tree=$(git rev-parse --show-toplevel)
revision=$(git rev-parse HEAD)
# A log's revision describes its run only when nothing else sits in the tree:
# uncommitted edits make the revision a label for code the run never executed.
dirty=$(git status --porcelain -uno | wc -l | tr -d ' ')
python=${PYTHON:-/home/ITER/mcintos/Code/nova/.venv/bin/python}
targets="scripts/prototypes/certified_carrier_export/test_export_records.py"
command="TMPDIR=${TMPDIR:-/tmp} NOVA_CERTIFIED_EXPORT_MUTATION=$mutation JAX_PLATFORMS=cpu PYTHONPATH=$tree $python -m pytest -p no:cacheprovider -q -rf $targets"
nova_file=$(PYTHONPATH="$tree" "$python" -c 'import nova; print(nova.__file__)')

{
  echo "# revision: $revision; tree: $tree; command: $command"
  echo "# nova.__file__: $nova_file"
  echo "# cwd: $(pwd -P)"
  if [ "$dirty" = "0" ]; then
    echo "# worktree: clean"
  else
    echo "# worktree: $dirty uncommitted entries (the revision above does not describe this run)"
  fi
  echo "# tree: $tree"
  echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# mutation: ${mutation:-none}"
  echo "# command: $command"
} > "$log"

NOVA_CERTIFIED_EXPORT_MUTATION="$mutation" \
JAX_PLATFORMS=cpu PYTHONPATH="$tree" \
  "$python" -m pytest -p no:cacheprovider -q -rf "$targets" >> "$log" 2>&1
result=$?
echo "EXIT=$result" >> "$log"
exit "$result"
