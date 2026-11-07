#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=graded-residual-compile
#SBATCH --output=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery/gate/sirius-%j.log
set -u
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH
export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
R=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T134005364650-graded-residual-compile-recovery
G=$R/gate; mkdir -p "$G"
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
BASE=cb42f656f76a7d7e56da8a53d4754c22d131bdc0
TARGETS="tests/biot/test_edge_term_tangents.py tests/biot/test_vertex_term_tangents.py tests/biot/test_vertex_edge_tangent_recurrences.py tests/biot/test_graded_layer_near_corner.py"

cd "$W" || exit 1
HEAD=$(git rev-parse HEAD)

run_ratio() {  # $1 module $2 name $3 tag
  local tag=$3 log=$G/compile-ratio-$3.log
  { echo "# revision=$HEAD"; echo "# tree=$W"; \
    echo "# command=$PY scripts/prototypes/polygon_near_band/compile_ratio_arms.py --base-revision $BASE --module $1 --name $2 --calls 3 --repeats 3 --log-root $G/row-$3"; } > "$log"
  "$PY" scripts/prototypes/polygon_near_band/compile_ratio_arms.py \
    --base-revision "$BASE" --module "$1" --name "$2" --calls 3 --repeats 3 \
    --log-root "$G/row-$tag" >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
}

run_suite() {  # $1 tag
  local tag=$1 log=$G/pytest-$1.log
  { echo "# revision=$HEAD"; echo "# tree=$W"; \
    echo "# command=$PY -m pytest -p no:cacheprovider --timeout 3600 -q $TARGETS"; } > "$log"
  PYTHONPATH="$W" PYTHONPATH="$W" "$PY" -m pytest -p no:cacheprovider --timeout 3600 -q $TARGETS >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
}

run_ratio test_edge_term_tangents second_residual edge-second-residual
run_ratio test_vertex_edge_tangent_recurrences regularised vertex-edge-regularised
run_suite head

BT=$(mktemp -d /tmp/graded-base-XXXXXX)
git archive "$BASE" | tar -x -C "$BT"
{ echo "# revision=$BASE"; echo "# tree=$BT"; \
  echo "# command=$PY -m pytest -p no:cacheprovider --timeout 3600 -q $TARGETS"; } > "$G/pytest-base.log"
(cd "$BT" && PYTHONPATH="$BT" "$PY" -m pytest -p no:cacheprovider --timeout 3600 -q $TARGETS >> "$G/pytest-base.log" 2>&1)
echo "EXIT=$?" >> "$G/pytest-base.log"
"$PY" -c "import nova.biot.gradedresidual as m; print(m.__file__)" >> "$G/pytest-head.log" 2>&1
rm -rf "$BT"
echo ALLDONE
