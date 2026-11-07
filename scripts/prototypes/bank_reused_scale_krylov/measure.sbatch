#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:45:00

# One allocation carries the re-measurement and the fail-closed control:
#   measure  the lane-sequential arm at 16 and 128 slices, then the batched
#            arm at 16 slices in two fresh processes, all with the corrected
#            shape assertion;
#   control  the batched arm at 128 slices, whose fused graph the allocator
#            refuses, which must make the driver exit nonzero. A control that
#            exits zero fails this job, because it would mean the driver
#            again certified a crashed arm.
set -u -o pipefail
export TMPDIR=/tmp
TREE=${TREE:?}
OUT=${OUT:?}

bash "$TREE/scripts/prototypes/bank_reused_scale_krylov/device.sh" \
  "$TREE" "$OUT/measure" map 16 128
map_status=$?
bash "$TREE/scripts/prototypes/bank_reused_scale_krylov/device.sh" \
  "$TREE" "$OUT/measure" vmap 16
vmap_status=$?
PASSES=1 bash "$TREE/scripts/prototypes/bank_reused_scale_krylov/device.sh" \
  "$TREE" "$OUT/control" vmap 128
control_status=$?

echo "# measure(map) EXIT=$map_status"
echo "# measure(vmap 16) EXIT=$vmap_status"
echo "# control(vmap 128) EXIT=$control_status (must be nonzero)"
echo "# control log: $OUT/control/vmap-128.log"
if [ "$map_status" -ne 0 ] || [ "$vmap_status" -ne 0 ] || [ "$control_status" -eq 0 ]; then
  echo "EXIT=1"
  exit 1
fi
echo "EXIT=0"