#!/usr/bin/env bash
# Run the spectra sweep and reduce it, with a gate log a reader can re-run.
#
# usage: gate.sh <run|analyse> <log>
#
# The log's first line names the revision, the tree and the command; its last
# line is a bare EXIT=<code>, which is where promotion reads the status.  One
# arm is one imas-efit process over completion of the study's own `efit`
# subcommand, so the arms run concurrently here rather than as one array job.
set -o pipefail

command_name=${1:?run or analyse}
log=${2:?absolute log path}
workers=${WORKERS:-5}

python=/home/ITER/mcintos/Code/nova/.venv/bin/python
tree=$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)
revision=$(git -C "$tree" rev-parse HEAD)
driver="$tree/scripts/prototypes/drifting_sensor_sequences/spectrum_sweep/sweep.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export DRIFT_STUDY=${DRIFT_STUDY:?set DRIFT_STUDY to the sweep study directory}

{
  echo "# revision $revision"
  echo "# tree $tree"
  echo "# module $python -c 'import nova.imas.mast_efit_referee'"
  echo "# cwd $(pwd)"
  echo "# command bash $0 $command_name $log"
} > "$log"

status=0
if [ "$command_name" = run ]; then
  "$python" -c "import nova.imas.mast_efit_referee as m; print('# module-file ' + m.__file__)" >> "$log" 2>&1
  "$python" "$driver" run --workers "$workers" --python "$python" >> "$log" 2>&1
  status=$?
else
  "$python" -c "import nova.imas.mast_efit_referee as m; print('# module-file ' + m.__file__)" >> "$log" 2>&1
  "$python" "$driver" analyse >> "$log" 2>&1
  status=$?
fi
echo "EXIT=$status" >> "$log"
exit "$status"