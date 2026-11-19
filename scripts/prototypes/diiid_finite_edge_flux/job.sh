#!/bin/bash
# Solve the eight rows on the cards this allocation holds.
#
# Every shard is a worker process with its own device, so the rows share the
# allocation and its compile rather than being one job each. Each shard writes
# its own log and its own comparison file, so an expiry loses one row.
set -u
COUNT=${1:?shard count}
REPOSITORY=${2:?repository}
OUTPUT=${3:?output directory}
BASE=${4:?base revision}
HEAD=${5:?head revision}
PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python
DRIVER="$REPOSITORY/scripts/prototypes/diiid_finite_edge_flux/batch.py"
export TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu
pids=()
for index in $(seq 0 $((COUNT - 1))); do
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON" "$DRIVER" \
    --repository "$REPOSITORY" \
    --base "$BASE" --head "$HEAD" --output "$OUTPUT" \
    --shard "$index" "$COUNT" \
    >"$OUTPUT/shard-$index.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
echo "SHARDS_EXIT=$status"
exit "$status"