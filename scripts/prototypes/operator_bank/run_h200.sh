#!/usr/bin/env bash
# One H200 job measuring the operator bank at the diverted 550-cell certificate.
set -u
WT=$(cd "$(dirname "$0")/../../.." && pwd)
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261028T144924463262-operator-bank
LOG=$RUN/logs/operator_bank_h200.log
export TMPDIR=/tmp
D=$(mktemp -d /tmp/operator-bank-XXXX)
trap 'rm -rf "$D"' EXIT
export PYTHONPATH="$WT"
export JAX_PLATFORMS=cuda
cd "$WT" || exit 1
echo "# operator-bank H200 measure" > "$LOG"
echo "# revision $(git -C "$WT" rev-parse HEAD) tree $WT command:\
 scripts/prototypes/operator_bank/measure.py --cells 550 --batch 16" >> "$LOG"
/home/ITER/mcintos/Code/nova/.venv/bin/python scripts/prototypes/operator_bank/measure.py \
  --cells 550 --batch 16 \
  --store "$RUN/operator_bank_store" \
  --compile-cache "$D/jax-cache" >> "$LOG" 2>&1
status=$?
echo "EXIT=$status" >> "$LOG"
exit $status
