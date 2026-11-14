#!/usr/bin/env bash
set -uo pipefail
TREE=${1:?worktree}
OUTPUT=${2:?reference output}
mkdir -p "$OUTPUT"
exec > "$OUTPUT/reference.log" 2>&1
REVISION=$(git -C "$TREE" rev-parse HEAD)
COMMAND=(/home/ITER/mcintos/Code/nova/.venv/bin/python
    "$TREE/scripts/prototypes/diiid_edge_current_refusal/measure.py"
    --tree "$TREE" --revision "$REVISION" --frame 146 --branch limited
    --output "$OUTPUT" --arm edge
    --trace-helper "$TREE/scripts/prototypes/diiid_hollow_current/trace.py")
printf '# revision: %s; tree: %s; command: ' "$REVISION" "$TREE"
printf '%q ' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"
RESULT=$?
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
