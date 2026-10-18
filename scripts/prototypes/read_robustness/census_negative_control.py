"""Fail while the exact-zero cell-membership step still loses the saddle.

Reads the per-seed census trace at the two adjacent offsets and asserts the
departure the repaired admission threshold removes: with the exact-zero test in
place the candidate set drops one cell across the pair, and the census loses the
hyperbolic wall-contact saddle with it. The script exits nonzero while that
departure is reproduced, so removing the repair is what makes this check fail;
a zero exit means the recorded trace no longer shows the loss.
"""

import json
import sys
from pathlib import Path


def main():
    trace = json.loads(Path(sys.argv[1]).read_text())
    rows = trace["rows"]
    if len(rows) != 2:
        print(f"expected two adjacent states, found {len(rows)}")
        return 2
    left, right = rows
    lc = left["stages"]["candidate_count"]
    rc = right["stages"]["candidate_count"]
    lv = sum(left["stages"]["valid"])
    rv = sum(right["stages"]["valid"])
    print(f"candidate cells {lc} -> {rc}; valid roots {lv} -> {rv}")
    print(f"left saddle valid {left['stages']['valid']}")
    print(f"right saddle valid {right['stages']['valid']}")
    if lc > rc and lv > rv:
        print("DEPARTURE REPRODUCED: the exact-zero admission step loses the "
              "hyperbolic saddle at the adjacent pair")
        return 1
    print("no departure in this trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
