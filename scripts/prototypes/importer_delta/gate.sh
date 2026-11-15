#!/usr/bin/env bash
# Rebuild every device's delta table from the recorded arm receipts and refuse a
# table that does not answer for the rows it declares.
#
#   gate.sh <receipts-dir> <base-rows> <out-dir> <device-config>...
#
# A device config names the device, its receipt files (resolved against the
# receipts directory) and the test files that device declares. The base-rows
# argument names the receipt files whose rows are the declared set, comma
# separated; every file they name must carry a completed pair, and no row may
# add a failure id. A partial table is a partial measurement, not a clean one,
# so the gate fails rather than passing on the factorization that the unmeasured
# rows happen to be absent from the rows it looked at. verify.py reads the rows
# and prints every file that carries no pair by name, which is the list a reader
# needs to dispatch the remainder.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
receipts="$1"
base_rows="$2"
out="$3"
shift 3
PY="${PYTHON:-python3}"

echo "# command=bash $0 ${args[*]}"
echo "# interpreter=$PY"
echo "# receipts=$receipts"
echo "# base_rows=$base_rows"
echo "# out=$out"

status=0
deltas=()
for config in "$@"; do
  name=$(basename "$config" | sed 's/-config\.json$//')
  delta="$out/delta-$name.json"
  "$PY" "$here/assemble.py" --config "$config" --base-rows "$base_rows" \
    --receipts-dir "$receipts" --out "$delta" || status=1
  deltas+=("$delta")
done

"$PY" "$here/verify.py" "$base_rows" "${deltas[@]}"
verdict=$?
if [ "$status" -ne 0 ]; then verdict="$status"; fi
echo "EXIT=$verdict"
exit "$verdict"
