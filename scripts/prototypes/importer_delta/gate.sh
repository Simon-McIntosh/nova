#!/usr/bin/env bash
# Rebuild every device's delta table from the recorded arm receipts, check the
# comparison against known inputs, and refuse a table the gated device cannot
# answer for.
#
#   gate.sh <receipts-dir> <base-rows> <out-dir> <control-log> <device-config>...
#
# A device config names the device, its receipt files (resolved against the
# receipts directory), the test files that device declares and the revisions
# its arms must record. A config marked gated must answer for every declared
# file: a completed head arm at the declared revision, a completed base arm as
# well unless the head is the only arm the file has, and no added failure id. A
# config that is not gated is information: its table is built and printed and
# never enters the verdict, because a slow file measured once is evidence for a
# reader rather than a row the gate can pass.
#
# The controls run inside the gate: the base receipts compared with themselves,
# with one id removed and with one id planted, half of which would make the
# comparison meaningless. A gate that passed beside a failing control would be
# reporting an instrument nobody had checked.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=("$@")
receipts="$1"
base_rows="$2"
out="$3"
control_log="$4"
shift 4
PY="${PYTHON:-python3}"

echo "# command=bash $0 ${args[*]}"
echo "# interpreter=$PY"
echo "# receipts=$receipts"
echo "# base-rows=$base_rows"
echo "# out=$out"
echo "# control-log=$control_log"

status=0
deltas=()
for config in "$@"; do
  name=$(basename "$config" | sed 's/-config\.json$//')
  delta="$out/delta-$name.json"
  "$PY" "$here/assemble.py" --config "$config" --base-rows "$base_rows" \
    --receipts-dir "$receipts" --out "$delta" || status=1
  deltas+=("$delta")
done

"$PY" "$here/verify.py" "${deltas[@]}"
verdict=$?

"$PY" "$here/control.py" "$base_rows" "$here/compare.py" "$control_log"
controls=$?

if [ "$status" -ne 0 ]; then verdict="$status"; fi
if [ "$controls" -ne 0 ]; then
  verdict=1
  echo "CONTROLS=FAIL see $control_log"
else
  echo "CONTROLS=PASS"
fi
echo "EXIT=$verdict"
exit "$verdict"