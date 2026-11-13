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
    census = json.loads((folder / "core-analysis.json").read_text())
    topology = receipt["terminal_topology"]
    return {
        "residual": float(receipt["residual"]),
        "converged": bool(receipt["converged"]),
        "reason": int(receipt["terminal_reason"]),
        "maxima_count": int(census["maxima_count"]),
        "minima_count": int(census["minima_count"]),
        "shadow_node_count": int(receipt["shadow_count"]),
        "support_cells": int(topology["support"]),
        "lower_x_distance_m": float(topology["lower_saddle_distance_m"]),
        "lower_x_rz_m": topology["lower_saddle_rz_m"],
        "efit_x_rz_m": receipt["efit_x_point_rz_m"],
        "arm": json.loads((folder / "arm.json").read_text())["arm"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"rows": {}, "control": {}, "missing": []}
    for case in CASES:
        name = case[2]
        if not all(
            (args.input / arm / f"frame-{case[0]}-{case[1]}" / "receipt.json").is_file()
            for arm in ("base", "edge")
        ):
            report["missing"].append(name)
            print(f"{name} MISSING a receipt")
            continue
        base = read(args.input, "base", case)
        edge = read(args.input, "edge", case)
        closed = base["lower_x_distance_m"] - edge["lower_x_distance_m"]
        components = {}
        for axis, index in (("radius", 0), ("height", 1)):
            start = base["lower_x_rz_m"][index] - base["efit_x_rz_m"][index]
            end = edge["lower_x_rz_m"][index] - edge["efit_x_rz_m"][index]
            components[axis] = {
                "base_m": start,
                "edge_m": end,
                "closed_m": start - end,
                "share_closed": (start - end) / start if start else None,
            }
        report["rows"][name] = {
            "base": base,
            "edge": edge,
            "closed_m": closed,
            "share_closed": closed / base["lower_x_distance_m"],
            "edge_over_base_distance": (
                edge["lower_x_distance_m"] / base["lower_x_distance_m"]
            ),
            "components": components,
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
            f"closed {100 * closed:.2f} cm ({fraction:.1f}%); "
            f"R {100 * components['radius']['base_m']:.4f}->"
            f"{100 * components['radius']['edge_m']:.4f} cm, Z "
            f"{100 * components['height']['base_m']:.4f}->"
            f"{100 * components['height']['edge_m']:.4f} cm"
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
