"""Sweep the Part V strip's error against the ring-integral reference.

Geometry is fixed once: a 0.10 m segment (the long edge of a thin cell) whose
centre sits at (1.5, 0.0) m, observed at the fixed target set -- half a cell
width, one cell width, one segment length and ten segment lengths, off the
segment's middle and off each end.

For every slope the strip's worst relative error over the target set is
recorded against the strip width ``eps``, which runs from ``1e-2`` to ``1e-8``
of the segment length. The curve has two branches -- the ``eps**2`` model error
and the rounding term that grows as the strip's two near-coincident long edges
cancel over ``eps`` -- so it turns over where they cross. The crossing is fitted
and reported with the best attainable error and the ``eps`` that gives it.

Run from the repository root:

    python scripts/edge_sheet/sweep.py --json <path>
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from scripts.edge_sheet.reference import (
    segment_frame,
    sheet_error,
    sheet_reference,
    strip_greens,
    target_set,
)

SLOPES = (0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0, math.pi / 2.0)
FRACTIONS = tuple(float(value) for value in np.logspace(-2, -8, 13))
CENTRE = np.array([1.5, 0.0])
LENGTH = 0.1
CELL_WIDTH = 0.02
DISTANCES = (0.5 * CELL_WIDTH, CELL_WIDTH, LENGTH, 10.0 * LENGTH)
TOLERANCE = 1e-10
MACHINE = float(np.finfo(np.float64).eps)
GEOMETRY = {
    "centre": [1.5, 0.0],
    "length": LENGTH,
    "cell_width": CELL_WIDTH,
    "distances": list(DISTANCES),
    "slopes": list(SLOPES),
    "fractions": list(FRACTIONS),
    "tolerance": TOLERANCE,
}


def _fit_branches(fractions: list[float], errors: list[float]) -> dict:
    """Fit ``error = a f**2 + b / f`` and return both terms at the minimum."""
    from scipy.optimize import curve_fit

    f = np.asarray(fractions)
    e = np.asarray(errors)
    best = int(np.argmin(e))
    start = (max(e[best], 1e-30) / f[best] ** 2, max(e[best], 1e-30) * f[best])
    try:
        (a, b), _ = curve_fit(
            lambda x, a, b: a * x**2 + b / x, f, e, p0=start, maxfev=20000
        )
    except RuntimeError:
        a, b = start
    return {
        "model_coefficient": float(a),
        "rounding_coefficient": float(b),
        "model_term_at_best": float(a * f[best] ** 2),
        "rounding_term_at_best": float(b / f[best]),
        # the rounding term is the machine epsilon inflated by the cancellation
        # between the strip's long edges: kappa is that inflation factor
        "cancellation_factor": float(b / (f[best] * MACHINE)),
    }


def run(only: tuple[float, ...] | None = None) -> dict:
    slopes = []
    for slope in SLOPES:
        if only is not None and not any(abs(slope - value) < 1e-9 for value in only):
            continue
        direction = np.array([math.cos(slope), math.sin(slope)])
        normal = np.array([direction[1], -direction[0]])
        start = CENTRE - 0.5 * LENGTH * direction
        end = CENTRE + 0.5 * LENGTH * direction
        centre, direction, normal, length = segment_frame(start, end)
        targets = target_set(
            centre,
            direction,
            normal,
            length,
            cell_width=CELL_WIDTH,
            distances=DISTANCES,
        )
        rows = []
        for target in targets:
            reference = sheet_reference(target, start, end)
            errors = []
            for fraction in FRACTIONS:
                strip = strip_greens(target, start, end, fraction=fraction)
                worst, detail = sheet_error(strip, reference)
                errors.append(worst)
            rows.append(
                {
                    "distance": target.distance,
                    "station": target.station,
                    "r": target.r,
                    "z": target.z,
                    "reference": {
                        "psi": reference.psi,
                        "bz": reference.bz,
                        "br": reference.br,
                    },
                    "reference_convergence": {
                        "converged": bool(reference.converged),
                        "rounds": reference.rounds,
                        "panels": reference.panels,
                        "achieved": reference.achieved,
                        "history": list(reference.history),
                    },
                    "errors": errors,
                }
            )
            print(
                "slope %.4f station %-6s d=%.3f ref %s err@1e-4 %.2e"
                % (
                    slope,
                    target.station,
                    target.distance,
                    "ok" if reference.converged else "NONCONVERGED",
                    errors[4],
                ),
                flush=True,
            )
        curve = [
            float(np.max([row["errors"][k] for row in rows]))
            for k in range(len(FRACTIONS))
        ]
        best_index = int(np.argmin(curve))
        fitted = _fit_branches(list(FRACTIONS), curve)
        slopes.append(
            {
                "slope": slope,
                "targets": rows,
                "curve": curve,
                "best_fraction": FRACTIONS[best_index],
                "best_error": curve[best_index],
                "all_references_converged": all(
                    row["reference_convergence"]["converged"] for row in rows
                ),
                "fit": fitted,
                "meets_tolerance_at_best": curve[best_index] <= TOLERANCE,
            }
        )
        print(
            "slope %.4f best %.3e at eps/L %.2e curve %s"
            % (
                slope,
                curve[best_index],
                FRACTIONS[best_index],
                " ".join("%.1e" % value for value in curve),
            ),
            flush=True,
        )
    return {"geometry": GEOMETRY, "slopes": slopes}


def combined(slopes: list[dict]) -> dict:
    """Worst error over every slope and target, and the widths meeting tolerance."""
    combined_curve = [
        float(np.max([slope["curve"][k] for slope in slopes]))
        for k in range(len(FRACTIONS))
    ]
    best_index = int(np.argmin(combined_curve))
    meeting = [
        FRACTIONS[k] for k, value in enumerate(combined_curve) if value <= TOLERANCE
    ]
    return {
        "curve": combined_curve,
        "best_fraction": FRACTIONS[best_index],
        "best_error": combined_curve[best_index],
        "fractions_meeting_tolerance": meeting,
        "single_fraction_meets_all_slopes": bool(meeting),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--only-slope",
        type=float,
        action="append",
        default=None,
        help="run this slope alone and splice it into --merge",
    )
    parser.add_argument("--merge", type=str, default=None)
    arguments = parser.parse_args()
    only = tuple(arguments.only_slope or ()) or None
    result = run(only)
    if arguments.merge:
        with open(arguments.merge) as handle:
            existing = json.load(handle)
        replaced = {round(slope["slope"], 6) for slope in result["slopes"]}
        kept = [
            slope
            for slope in existing["slopes"]
            if round(slope["slope"], 6) not in replaced
        ]
        result["slopes"] = sorted(kept + result["slopes"], key=lambda s: s["slope"])
        result["geometry"] = existing["geometry"]
    result["combined"] = combined(result["slopes"])
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(result, handle, indent=1)
        print("wrote", arguments.json)
    summary = result["combined"]
    print(
        "combined best %.3e at eps/L %.2e; fractions meeting 1e-10: %s"
        % (
            summary["best_error"],
            summary["best_fraction"],
            summary["fractions_meeting_tolerance"] or "none",
        )
    )
    failed = [
        slope["slope"]
        for slope in result["slopes"]
        if not slope["all_references_converged"]
    ]
    if failed:
        print("NON-CONVERGING REFERENCE at slopes", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
