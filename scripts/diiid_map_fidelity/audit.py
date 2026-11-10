"""Audit captured fields without rerunning the solver or changing raw receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root


def stationary(field, radius, height, guess):
    spline = RectBivariateSpline(radius, height, field)
    answer = root(lambda x: [spline.ev(*x, dx=1), spline.ev(*x, dy=1)], guess)
    point = answer.x
    determinant = float(
        spline.ev(*point, dx=2) * spline.ev(*point, dy=2)
        - spline.ev(*point, dx=1, dy=1) ** 2
    )
    valid = bool(
        answer.success
        and np.linalg.norm(answer.fun) < 1e-7
        and radius[0] <= point[0] <= radius[-1]
        and height[0] <= point[1] <= height[-1]
        and determinant < 0
    )
    return point, valid, determinant, float(np.linalg.norm(answer.fun))


def continue_saddle(label, solved, radius, height, seed):
    """Continue a saddle through interpolated fields with bounded local hops."""
    guess = np.asarray(seed)
    trace = []
    connected = True
    for fraction in np.linspace(0, 1, 41):
        field = (1 - fraction) * label + fraction * solved
        point, valid, determinant, gradient = stationary(field, radius, height, guess)
        hop = float(np.linalg.norm(point - guess))
        connected &= valid and hop < 0.1
        trace.append(
            dict(
                fraction=float(fraction),
                rz_m=point.tolist(),
                valid=valid,
                displacement_from_previous_m=hop,
                hessian_determinant=determinant,
                gradient_norm=gradient,
            )
        )
        if not connected:
            break
        guess = point
    return guess, bool(connected), trace


def audit(row):
    data = dict(np.load(row / "fields.npz"))
    terminal = json.loads((row / "solve.json").read_text())
    reference = json.loads((row / "reference.json").read_text())
    guess, connected, trace = continue_saddle(
        data["label"],
        data["solved"],
        data["radius"],
        data["height"],
        data["reference_saddle"],
    )
    selected = np.asarray(terminal["selected_saddle_rz_m"])
    selected_matches = bool(connected and np.linalg.norm(guess - selected) < 1e-5)
    admitted_same_branch = terminal["topology"] == "diverted" and selected_matches
    result = dict(
        identity=row.name,
        branch_method=(
            "41 samples of linear interpolation between reference and "
            "terminal fields; saddle polishing from preceding root"
        ),
        continuation_completed=bool(connected),
        selected_matches_continuation=selected_matches,
        admitted_same_branch=bool(admitted_same_branch),
        admitted_xpoint_distance_m=float(
            np.linalg.norm(selected - data["reference_saddle"])
        )
        if admitted_same_branch
        else None,
        reference_saddle_rz_m=data["reference_saddle"].tolist(),
        selected_saddle_rz_m=selected.tolist(),
        continuation=trace,
        force_displacement_slope_n_per_m=(
            reference[2]["force_n"] - reference[1]["force_n"]
        )
        / 0.04,
        negative_shift_residual_ratio=reference[1]["relative"]
        / reference[0]["relative"],
        positive_shift_residual_ratio=reference[2]["relative"]
        / reference[0]["relative"],
        note=(
            "Field interpolation diagnoses root identity; it is distinct from "
            "the executed Newton trajectory and is not an exhaustive null census."
        ),
    )
    (row / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        "AUDIT",
        json.dumps(
            {key: value for key, value in result.items() if key != "continuation"}
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(path.parent) for path in sorted(args.input.glob("*/solve.json"))]
    assert len(rows) == 4, f"expected four complete rows, found {len(rows)}"
    mast = next(path for path in args.input.glob("mast-*/solve.json"))
    control = json.loads(mast.read_text())
    assert control["converged"], "MAST reference-seeded Newton did not converge"
    assert control["direct_terminal_map"]["relative"] < 1e-8, (
        "known converged MAST root is not small on the map instrument"
    )
    print(
        "AUDIT COMPLETE: four rows; MAST converged-root map positive control passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
