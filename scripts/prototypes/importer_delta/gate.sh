#!/usr/bin/env bash
# Rebuild every device's delta table from the recorded arm receipts and refuse a
# table that does not answer for the rows it declares.
#
#   gate.sh <receipts-dir> <base-rows> <out-dir> <device-config>...
#
# A device config names the device, its receipt file (resolved against the
# receipts directory) and the test files that device declares. The gate fails when a table has no rows, when any row's head
# arm is missing or did not complete, and on any added failure id. A partial
# table is a partial measurement, not a clean one, so it fails rather than
# passing on the factorization that the unmeasured rows happen to be absent from
# the rows the gate looks at. Every unmeasured row is printed by name and state,
# which is the list a reader needs to dispatch the remainder.
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

"$PY" - "${deltas[@]}" <<'PY'
import json
import sys

problems = []
added = 0
unmeasured = 0
for path in sys.argv[1:]:
    data = json.loads(open(path).read())
    rows = data['rows']
    device = data['device']
    bad = data.get('unmeasured') or []
    print('DEVICE=%s ROWS=%d VERDICTS=%d UNMEASURED=%d ADDED=%d'
          % (device, len(rows), len(rows) - len(bad), len(bad),
             data.get('added_total', 0)))
    if not rows:
        problems.append('%s table is empty' % device)
    for row in bad:
        print('UNMEASURED %s %s %s' % (device, row['file'], row['state']))
    unmeasured += len(bad)
    for row in rows:
        for item in row['added']:
            print('ADDED %s %s %s' % (device, row['file'], item))
    added += data.get('added_total', 0)

if problems:
    print('GATE=FAIL empty=1 added=%d unmeasured=%d' % (added, unmeasured))
    for line in problems:
        print('PROBLEM ' + line)
    raise SystemExit(1)
if unmeasured or added:
    print('GATE=FAIL empty=0 added=%d unmeasured=%d' % (added, unmeasured))
    raise SystemExit(1)
print('GATE=PASS empty=0 added=0 unmeasured=0')
PY
verdict=$?
if [ "$status" -ne 0 ]; then verdict="$status"; fi
echo "EXIT=$verdict"
exit "$verdict"