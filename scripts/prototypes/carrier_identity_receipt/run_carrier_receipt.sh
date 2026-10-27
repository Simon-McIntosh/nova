#!/bin/bash
# Build the certificate carriers in this tree and publish the carrier and
# reference-solve identity each build reads.
set -uo pipefail
export TMPDIR=/tmp
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
R=${CARRIER_RUN_DIR:?set CARRIER_RUN_DIR to the directory the receipts and logs go under}
LABEL=${1:-arm}
shift || true
export PYTHONPATH="$W"
export NOVA_REVISION=$(git -C "$W" rev-parse HEAD)
mkdir -p "$R/receipts"
export CARRIER_RECEIPT="$R/receipts/carrier-${LABEL}.json"
cd "$W" || exit 3
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
echo "# command: bash scripts/prototypes/carrier_identity_receipt/run_carrier_receipt.sh $LABEL $*"
echo "# revision: $NOVA_REVISION"
echo "# tree: $W"
echo "# receipt: $CARRIER_RECEIPT"
$PY -c "import nova, os; print('nova.__file__', nova.__file__); print('cwd', os.getcwd())"
$PY - <<'PY'
import jax
from nova.jax.config import configure_dtypes
configure_dtypes()
print('x64', jax.config.jax_enable_x64, 'backend', jax.default_backend(), 'devices', jax.devices())
PY
$PY scripts/prototypes/carrier_identity_receipt/measure_carrier.py "$@"
status=$?
echo "EXIT=$status"
exit $status