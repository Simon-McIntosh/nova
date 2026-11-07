#!/bin/bash
# One H200 allocation: fast route and present route on certificate rows.
set -u
export TMPDIR=/tmp
W=$(git rev-parse --show-toplevel)
OUT=${1:?results directory required}
mkdir -p "$OUT"
cd "$W" || exit 1
for spec in \
  "diverted 550 --present" \
  "limited 550 --present" \
  "limited 2000 --present --present-cells -2500" \
  "diverted 2000 --present --present-cells -2500"
do
  set -- $spec
  kind=$1; cells=$2; shift 2
  bash scripts/prototypes/bank_route_rows/run.sh \
    "$OUT/$kind-$cells.log" "$OUT/$kind-$cells.json" \
    --kind "$kind" --cells "$cells" "$@"
done
printf 'ROWS_COMPLETE=%s\n' "$(date -u +%FT%TZ)"