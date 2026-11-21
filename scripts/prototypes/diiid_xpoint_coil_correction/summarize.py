"""Build the two-arm comparison record from the row receipts.

Every row is solved twice on its own base state's fit: once with the measured
coil currents and once with those currents less the fitted equivalent forward
error. A fall in the lower X-point distance is only read as a share of the
offset on a row whose corrected state converged with its support retained,
because a terminal state that has moved to another plasma has a different
distance rather than a smaller error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conductor_correction  # noqa: E402

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
ARMS = ("base", "corrected")
THRESHOLD = 0.05

TERMINATION = {
    1: "CONVERGED",
    7: "ACTIVE_SET_CYCLE_DETECTED",
    8: "ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED",
    12: "ACTIVE_SET_STEP_REFUSED",
}


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def arm_row(root, arm, case):
    folder = root / arm / case
    record = json.loads((folder / "receipt.json").read_text())
    summary = json.loads((folder / "shadow-image.json").read_text())
    census = json.loads((folder / "core-analysis.json").read_text())
    topology = record["terminal_topology"]
    null = topology["lower_saddle_rz_m"]
    reference = summary["efit_x_rz_m"]
    offset = [100.0 * (null[0] - reference[0]), 100.0 * (null[1] - reference[1])]
    support = topology.get("support")
    return {
        "residual": float(record["residual"]),
        "converged": bool(record["converged"]),
        "termination": TERMINATION.get(
            int(summary["termination_reason"]), str(summary["termination_reason"])
        ),
        "support_cells": None if support is None else int(support),
        "maxima_count": int(census["maxima_count"]),
        "minima_count": int(census["minima_count"]),
        "axis_rz_m": [float(item) for item in record["axis_rz_m"]],
        "lower_x_rz_m": [float(item) for item in null],
        "efit_x_rz_m": [float(item) for item in reference],
        "offset_cm": offset,
        "distance_cm": 100.0 * float(topology["lower_saddle_distance_m"]),
        "shadow_node_count": int(summary["shadow_node_count"]),
    }


def read_only(row):
    """A fall is a share of the offset only if the terminal state is the same plasma."""

    corrected = row["corrected"]
    base = row["base"]
    if not corrected["converged"]:
        return None, f"corrected arm terminated {corrected['termination']}"
    if corrected["support_cells"] is None or base["support_cells"] is None:
        return None, "support not recorded"
    if corrected["support_cells"] != base["support_cells"]:
        return None, (
            f"support moved from {base['support_cells']} to "
            f"{corrected['support_cells']} cells"
        )
    if base["distance_cm"] == 0.0:
        return None, "base distance is zero"
    return (base["distance_cm"] - corrected["distance_cm"]) / base["distance_cm"], None


def measured_inputs(root, case):
    """The measured vector, from whichever arm recorded it.

    The corpus-column writer records the measured vector only on a corrected
    arm, because it is written where the shift is applied. A measured arm's
    vector is the same read of the same corpus row, so it is taken from the
    corrected arm when the measured arm records none.
    """

    for arm in ("base", "corrected"):
        path = root / arm / case / "current-inputs.json"
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(root / "base" / case / "current-inputs.json")


def coil_rows(root, case, frame):
    applied = measured_inputs(root, case)
    fit_path = root / "fits" / case / f"conductor-fit-{frame}.json"
    names = applied["names"]
    fit = json.loads(fit_path.read_text()) if fit_path.is_file() else None
    measured = applied["measured_a_turn"]
    equivalent = (
        [float(item) for item in fit["equivalent_error_a_turn"]]
        if fit is not None
        else [0.0] * len(names)
    )
    fractions = conductor_correction.relative_correction(measured, equivalent)
    rows = []
    for name, current, value, share in zip(names, measured, equivalent, fractions):
        rows.append(
            {
                "name": name,
                "measured_ka_turn": 1e-3 * current,
                "equivalent_error_ka_turn": 1e-3 * value,
                "relative_correction": abs(share),
                "implausible": abs(share) > THRESHOLD,
            }
        )
    return rows, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"rows": {}, "total": {}, "excluded": {}, "coils": {}, "fits": {}}
    read_ = []
    for frame, branch in ROWS:
        case = case_name(frame, branch)
        row = {}
        for arm in ARMS:
            folder = args.input / arm / case
            if not (folder / "shadow-image.json").is_file():
                row[arm] = None
                continue
            row[arm] = arm_row(args.input, arm, case)
        report["rows"][case] = row
        if row.get("base") and row.get("corrected"):
            share, reason = read_only(row)
            report["rows"][case]["share_closed"] = share
            if reason is None:
                read_.append(case)
            else:
                report["excluded"][case] = reason
        coils, fit = coil_rows(args.input, case, frame)
        report["coils"][case] = coils
        if fit is not None:
            report["fits"][case] = {
                "rank": fit["rank"],
                "scaled_condition": fit["scaled_condition"],
                "fitted_squared_fraction": fit["fitted_squared_fraction"],
                "positive_control_residual_fraction": fit[
                    "fit_positive_control_residual_fraction"
                ],
                "measured_offset_cm": fit["measured_offset_cm"],
            }
    if read_:
        fall = sum(
            report["rows"][case]["base"]["distance_cm"]
            - report["rows"][case]["corrected"]["distance_cm"]
            for case in read_
        )
        total = sum(report["rows"][case]["base"]["distance_cm"] for case in read_)
        report["total"] = {
            "rows": read_,
            "fall_cm": fall,
            "base_cm": total,
            "share": fall / total if total else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case, row in report["rows"].items():
        for arm in ARMS:
            item = row.get(arm)
            if item is None:
                print(f"{case:22s} {arm:10s} missing")
                continue
            print(
                f"{case:22s} {arm:10s} residual={item['residual']:.3e} "
                f"converged={item['converged']} term={item['termination']:38s} "
                f"support={item['support_cells']} "
                f"max/min={item['maxima_count']}/{item['minima_count']} "
                f"offset=({item['offset_cm'][0]:+.3f},{item['offset_cm'][1]:+.3f}) cm "
                f"distance={item['distance_cm']:.3f} cm"
            )
        share = row.get("share_closed")
        print(
            f"{case:22s} share_closed={share if share is None else round(share, 4)}"
            + (f" excluded={report['excluded'].get(case)}" if share is None else "")
        )


if __name__ == "__main__":
    main()
