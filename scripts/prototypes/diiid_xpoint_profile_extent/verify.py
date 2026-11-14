"""Check that each row landed and that each solve was handed its own arm's knots.

The measurement is a change of prescribed profiles, so the control is a reading
of the tables the solve actually received. A row whose arm was silently dropped
still solves, still writes a receipt, and still reports a distance; only the
inputs it recorded can tell the two apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = (
    (137, "diverted"),
    (146, "diverted"),
    (146, "limited"),
    (179, "limited"),
)
ARMS = ("base", "edge")
EXPECTED_KNOTS = {"base": 19, "edge": 29}
EXPECTED_BOUNDARY = {"base": 0.95, "edge": 1.0}
RECEIPTS = ("receipt.json", "core-analysis.json")


def tables(folder):
    return json.loads((folder / "source-inputs.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for frame, branch in CASES:
        case = f"frame-{frame}-{branch}"
        received = {}
        for arm in ARMS:
            folder = args.input / arm / case
            missing = [name for name in RECEIPTS if not (folder / name).is_file()]
            if missing:
                failures.append(f"{arm} {case}: missing {missing}")
                continue
            log = folder / "row.log"
            tail = log.read_text().splitlines()[-1] if log.is_file() else ""
            if tail != "EXIT=0":
                failures.append(f"{arm} {case}: row.log ends {tail!r}")
            record = tables(folder)
            received[arm] = record
            for name in ("p_prime", "ff_prime"):
                coordinate = record[name]["coordinate"]
                count = len(coordinate)
                boundary = round(coordinate[-1], 6)
                print(
                    f"{arm:4s} {case:20s} {name:8s} knots={count:3d} "
                    f"last={boundary:.3f} value={record[name]['values'][-1]:.6f}"
                )
                if count != EXPECTED_KNOTS[arm]:
                    failures.append(f"{arm} {case} {name}: {count} knots")
                if boundary != EXPECTED_BOUNDARY[arm]:
                    failures.append(f"{arm} {case} {name}: boundary {boundary}")
        if len(received) != len(ARMS):
            continue
        for name in ("p_prime", "ff_prime"):
            base = received["base"][name]
            edge = received["edge"][name]
            shared = base["coordinate"] == edge["coordinate"][: len(base["coordinate"])]
            difference = max(
                abs(one - other) for one, other in zip(base["values"], edge["values"])
            )
            print(
                f"     {case:20s} {name:8s} shared_coordinate={shared} "
                f"max_abs_difference={difference:.3e}"
            )
            if not shared:
                failures.append(f"{case} {name}: interior coordinates differ")
            if difference:
                failures.append(f"{case} {name}: interior values differ")
    if failures:
        for failure in failures:
            print("FAIL " + failure)
        raise SystemExit(1)
    print(f"PASS {len(CASES) * len(ARMS)} rows, {len(CASES)} arm pairs")


if __name__ == "__main__":
    main()
