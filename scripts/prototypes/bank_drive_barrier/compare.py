"""Compare the quantitative rows of two fresh barrier probes."""

import json
import math
import sys
from pathlib import Path


def rows(path):
    selected = []
    for line in Path(path).read_text().splitlines():
        label, payload = line.split(" ", 1)
        if label in {"REPLAY", "PROBE", "CONDITION", "DONE"}:
            selected.append((label, json.loads(payload)))
    return selected


def main():
    first, second = (rows(path) for path in sys.argv[1:3])
    if len(first) != 12:
        raise SystemExit("fresh processes disagree or a quantitative row is absent")
    for (left_label, left), (right_label, right) in zip(first, second, strict=True):
        if left_label != right_label or left.keys() != right.keys():
            raise SystemExit(f"fresh processes disagree on {left_label}")
        if left_label == "REPLAY":
            left = {**left, "reason": left["reason"].split(" with residual")[0]}
            right = {**right, "reason": right["reason"].split(" with residual")[0]}
        for key, value in left.items():
            other = right[key]
            agreed = (
                math.isclose(value, other, rel_tol=1e-6, abs_tol=0.0)
                if left_label == "CONDITION" and key not in {"drive", "valid"}
                else value == other
            )
            if not agreed:
                raise SystemExit(f"fresh processes disagree on {left_label}.{key}")
    residuals = [row[1]["residual_max"] for row in first if row[0] == "PROBE"]
    conditions = [row[1]["condition_number"] for row in first if row[0] == "CONDITION"]
    print(
        f"AGREE replay=1 probes=6 conditions={conditions} done=1 residuals={residuals}"
    )


if __name__ == "__main__":
    main()
