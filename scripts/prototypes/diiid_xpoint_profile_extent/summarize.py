"""Reconcile the cohort receipts, close the offset shares and check the base arm.

The base arm re-solves the established tables, so its lower-X distances must
reproduce the converged cohort already recorded for these rows. That agreement
is the control that the two arms differ only in the prescribed tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = (
    (137, "diverted", "137D"),
    (146, "diverted", "146D"),
    (146, "limited", "146L"),
    (179, "limited", "179L"),
)
BANKED_BASE_DISTANCE_M = {
    "137D": 0.02360164757,
    "146D": 0.02960378735,
    "146L": 0.05129600751,
    "179L": 0.01442593043,
}


def read(root, arm, case):
    folder = root / arm / f"frame-{case[0]}-{case[1]}"
    receipt = json.loads((folder / "receipt.json").read_text())
    shadow = json.loads((folder / "shadow-image.json").read_text())
    return {
        "residual": float(receipt["residual"]),
        "converged": bool(receipt["converged"]),
        "reason": receipt.get("terminal_reason", shadow.get("termination_reason")),
        "maxima_count": int(shadow["maxima_count"]),
        "lower_x_distance_m": float(shadow["lower_x_distance_m"]),
        "lower_x_rz_m": shadow["lower_x_rz_m"],
        "efit_x_rz_m": shadow["efit_x_rz_m"],
        "support_cells": int(receipt.get("support_cells", 0)),
        "arms": json.loads((folder / "arm.json").read_text())["arm"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"rows": {}, "control": {}}
    for case in CASES:
        base = read(args.input, "base", case)
        edge = read(args.input, "edge", case)
        name = case[2]
        closed = base["lower_x_distance_m"] - edge["lower_x_distance_m"]
        report["rows"][name] = {
            "base": base,
            "edge": edge,
            "closed_m": closed,
            "share_closed": closed / base["lower_x_distance_m"],
            "edge_over_base_distance": (
                edge["lower_x_distance_m"] / base["lower_x_distance_m"]
            ),
        }
        banked = BANKED_BASE_DISTANCE_M[name]
        report["control"][name] = {
            "banked_m": banked,
            "reproduced_m": base["lower_x_distance_m"],
            "difference_m": base["lower_x_distance_m"] - banked,
            "reproduced": abs(base["lower_x_distance_m"] - banked) < 5.0e-4,
        }
        fraction = 100 * closed / base["lower_x_distance_m"]
        print(
            f"{name} base {base['lower_x_distance_m']:.5f} m "
            f"(banked {banked:.5f}) -> edge {edge['lower_x_distance_m']:.5f} m; "
            f"closed {100 * closed:.2f} cm ({fraction:.1f}%)"
        )
    report["all_bases_reproduced"] = all(
        item["reproduced"] for item in report["control"].values()
    )
    report["total_closed_m"] = sum(item["closed_m"] for item in report["rows"].values())
    report["total_base_m"] = sum(
        item["base"]["lower_x_distance_m"] for item in report["rows"].values()
    )
    report["total_share_closed"] = report["total_closed_m"] / report["total_base_m"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"TOTAL {100 * report['total_closed_m']:.2f} of "
        f"{100 * report['total_base_m']:.2f} cm "
        f"({100 * report['total_share_closed']:.1f}%)"
    )


if __name__ == "__main__":
    main()
