"""Check that each fitted arm's solved current vector carries the fitted correction.

The solve records the conductor order and vector its adapter installed. Over the
shipped conductors, that vector must equal the measured arm's vector less the
row's fitted correction, entry by entry. A correction that reached the operator
can never leave the two vectors equal, so pairing the measured arm with itself
is the negative control: that comparison must redden, and a green fitted-arm
result then means the check can see a missing correction rather than merely
finding nothing to report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROWS = ((137, "diverted"), (179, "limited"), (146, "diverted"), (146, "limited"))


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def solved(path):
    record = json.loads(path.read_text())
    return dict(zip(record["shipped_names"], record["shipped_current_a_turn"]))


def correction_for(root, case, frame):
    path = root / "fits" / case / ("correction-%d.json" % frame)
    return json.loads(path.read_text())


def compare(measured, fitted, correction):
    """Per-conductor expected and actual, and the conductors that disagree."""

    rows = []
    mismatched = []
    for name, actual in fitted.items():
        measured_value = float(measured[name])
        error = float(correction.get(name, 0.0))
        expected = measured_value - error
        residual = abs(float(actual) - expected)
        scale = max(abs(measured_value), 1.0)
        rows.append(
            {
                "name": name,
                "measured_a_turn": measured_value,
                "correction_a_turn": error,
                "expected_a_turn": expected,
                "solved_a_turn": float(actual),
                "relative_residual": residual / scale,
            }
        )
        if not np.isclose(float(actual), expected, rtol=1e-9, atol=1e-3):
            mismatched.append(name)
    return rows, mismatched


def arm_pair(root, case, frame, control):
    measured = solved(root / "base" / case / "solve-currents.json")
    if control == "base-base":
        fitted = measured
    else:
        fitted = solved(root / "corrected" / case / "solve-currents.json")
    correction = correction_for(root, case, frame)
    return measured, fitted, correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control",
        choices=("fitted", "base-base"),
        default="fitted",
        help="pair the fitted arm with the measured one, or measured with itself",
    )
    args = parser.parse_args()
    report = {"control": args.control, "rows": {}, "confirmed": False}
    failures = 0
    for frame, branch in ROWS:
        case = case_name(frame, branch)
        measured_path = args.input / "base" / case / "solve-currents.json"
        fitted_path = args.input / "corrected" / case / "solve-currents.json"
        wanted = measured_path if args.control == "base-base" else fitted_path
        if not measured_path.is_file() or not wanted.is_file():
            report["rows"][case] = {
                "missing": str(wanted if not wanted.is_file() else measured_path)
            }
            failures += 1
            print("%s arm missing: %s" % (case, report["rows"][case]["missing"]))
            continue
        measured, fitted, correction = arm_pair(args.input, case, frame, args.control)
        rows, mismatched = compare(measured, fitted, correction)
        unchanged = [
            item["name"]
            for item in rows
            if item["correction_a_turn"] != 0.0
            and abs(item["solved_a_turn"] - item["measured_a_turn"]) <= 1e-3
        ]
        report["rows"][case] = {
            "conductors": len(rows),
            "mismatched": mismatched,
            "unchanged_under_a_fitted_correction": unchanged,
            "rows": rows,
        }
        failures += len(mismatched)
        print(
            "%s conductors=%d mismatched=%d unchanged=%d"
            % (case, len(rows), len(mismatched), len(unchanged)),
            flush=True,
        )
    report["confirmed"] = failures == 0 and all(
        not item.get("unchanged_under_a_fitted_correction")
        for item in report["rows"].values()
    )
    report["rows_present"] = sum(
        1 for item in report["rows"].values() if "conductors" in item
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("CONFIRMED" if report["confirmed"] else "NOT_CONFIRMED", flush=True)
    return 0 if report["confirmed"] else 1


if __name__ == "__main__":
    sys.exit(main())
