"""Compare the quantitative rows of two fresh barrier probes."""

import json
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
    if len(first) != 11 or first != second:
        raise SystemExit("fresh processes disagree or a quantitative row is absent")
    residuals = [row[1]["residual_max"] for row in first if row[0] == "PROBE"]
    conditions = [row[1]["condition_number"] for row in first if row[0] == "CONDITION"]
    print(
        f"AGREE replay=1 probes=6 conditions={conditions} done=1 residuals={residuals}"
    )


if __name__ == "__main__":
    main()
