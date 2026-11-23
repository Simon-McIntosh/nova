#!/bin/bash
set -uo pipefail
export TMPDIR=/tmp
WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="$WORKTREE/scripts/prototypes/hot_path_map"
OUT="${1:-$HOME/.cache/nova-s33/hot-path-map}"
REV="$(git -C "$WORKTREE" rev-parse HEAD)"
NEGATIVE_CONTROL="nova/utilities/ANSYS.py"
export PYTHONPATH="$WORKTREE"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
echo "# revision: $REV tree: $WORKTREE cwd: $(pwd) python: $PY"
echo "# command: bash scripts/prototypes/hot_path_map/job.sh $OUT"
cd "$WORKTREE"
"$PY" -c "import nova, sys; print('# nova:', nova.__file__); print('# python:', sys.version.split()[0])"

"$PY" -m coverage run --rcfile="$DRIVER/coverage.rc" \
  --data-file="$OUT/forward.coverage" \
  "$DRIVER/forward_route.py" --output "$OUT/forward" --revision "$REV"
FORWARD_STATUS=$?
echo "FORWARD_EXIT=$FORWARD_STATUS"
"$PY" -m coverage json --rcfile="$DRIVER/coverage.rc" \
  --data-file="$OUT/forward.coverage" -o "$OUT/forward.json"
echo "FORWARD_JSON_EXIT=$?"
export NOVA_MAGNETICS_FIT_EVIDENCE="$OUT/magnetics"
"$PY" -m coverage run --rcfile="$DRIVER/coverage.rc" \
  --data-file="$OUT/inverse.coverage" -m pytest \
  -o addopts='' -p no:cacheprovider --timeout=3600 -q \
  "tests/test_equilibrium_magnetics_fit.py::test_known_state_recovery[mast]"
INVERSE_STATUS=$?
echo "INVERSE_EXIT=$INVERSE_STATUS"
"$PY" -m coverage json --rcfile="$DRIVER/coverage.rc" \
  --data-file="$OUT/inverse.coverage" -o "$OUT/inverse.json"
echo "INVERSE_JSON_EXIT=$?"
"$PY" "$DRIVER/map.py" \
  --forward "$OUT/forward.json" --inverse "$OUT/inverse.json" \
  --root "$WORKTREE" --revision "$REV" --tree "$WORKTREE" \
  --negative-control "$NEGATIVE_CONTROL" \
  --output-json "$OUT/hot-path-map.json"
MAP_STATUS=$?
echo "MAP_EXIT=$MAP_STATUS"
STATUS=0
[ "$FORWARD_STATUS" -eq 0 ] || STATUS=1
[ "$INVERSE_STATUS" -eq 0 ] || STATUS=1
[ "$MAP_STATUS" -eq 0 ] || STATUS=1
echo "hot-path-map route exits forward=$FORWARD_STATUS inverse=$INVERSE_STATUS map=$MAP_STATUS"
exit $STATUS