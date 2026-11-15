"""Solve one finite-edge row and read its source-cutoff jump discriminator.

The solve, the seeding and the branch receipts are the profile-extent cohort's
edge arm, run with the DIII-D consumer on the booked finite-edge route. After
that row lands this driver reads the same built profile at the terminal state
and reports the current the booked route gives the cells that straddle the
level, beside the value-only route's assignment there.

The two routes are read on one operator, so the comparison differs only in how
the cell current is taken from the sampled density: the booked route integrates
the moment blocks the consumer supplies, and the value-only route evaluates the
point weight at the cell centre.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location(
    "extent_measure", ROOT / "scripts/prototypes/diiid_xpoint_profile_extent/measure.py"
)
extent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extent
SPEC.loader.exec_module(extent)

from benchmarks import diiid_forward_gs_match as bench  # noqa: E402

BUILT: dict[str, object] = {}
SOLVED: dict[str, object] = {}
BAND = 0.02


def capture_build(original):
    """Retain the profile the row built, so the read uses that one."""

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        BUILT["value"] = result
        return result

    return build


def capture_adapter(original):
    """Retain the operator the current adapter completed.

    The completed adapter's profile is the one the row goes on to solve, so it
    is retained here: the build call returns a six-item tuple, not a profile,
    and the discriminator needs the operator the adapter built.
    """

    def adapter(profile, **kwargs):
        result = original(profile, **kwargs)
        BUILT["profile"] = result.profile
        return result

    return adapter


def capture_solve(original):
    """Retain the equilibrium the row returned, so the read uses that one."""

    def solve(profile, *args, **kwargs):
        result = original(profile, *args, **kwargs)
        SOLVED["value"] = result
        return result

    return solve


def _optional(value):
    """Return a JSON-safe float, or None where the field is absent."""

    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    number = float(array.reshape(-1)[0])
    return number if np.isfinite(number) else None


def terminal(equilibrium) -> dict[str, object]:
    """Return the terminal convergence and current-pin readings of one solve."""

    fixed_point = equilibrium.fixed_point
    normalisation = equilibrium.normalisation
    return {
        "terminal_residual": _optional(fixed_point.residual),
        "converged": bool(np.asarray(fixed_point.converged)),
        "termination_reason": int(
            np.asarray(getattr(fixed_point, "termination_reason", -999)).reshape(-1)[0]
        ),
        "current_pin_amplitude": _optional(normalisation.amplitude),
        "current_pin_policy": normalisation.policy_name,
    }


def discriminator() -> dict[str, object]:
    """Return the booked and value-only current of the boundary-band cells."""

    profile = BUILT.get("profile")
    equilibrium = SOLVED.get("value")
    if profile is None or equilibrium is None:
        return {"status": "unavailable"}
    operator = profile.operator
    psi = equilibrium.flux
    topology = operator._fixed_design_read(psi)[1]
    axis = float(np.asarray(topology.axis_flux))
    boundary = float(np.asarray(topology.boundary_flux))
    if not (np.isfinite(axis) and np.isfinite(boundary)):
        return {"status": "unresolved boundary level"}
    # The state is ordered grid nodes, then wall nodes, then the sampling
    # nodes, while a cell current carries one entry per grid cell, so both
    # reads are restricted to the grid population.
    cells = operator.grid.node_number
    normalised = (np.asarray(psi)[:cells] - axis) / (boundary - axis)
    band = np.abs(normalised - 1.0) <= BAND
    if not band.any():
        return {"status": "empty band"}
    booked = np.asarray(operator.cell_current(psi))[:cells]
    point = np.asarray(
        replace(operator, use_linear_moments=False, sample=None).cell_current(psi)
    )[:cells]
    difference = np.abs(booked - point)[band]
    band_peak = float(np.abs(booked[band]).max())
    return {
        "status": "measured",
        "band_cells": int(band.sum()),
        "booked_band_current_sum_a": float(booked[band].sum()),
        "value_only_band_current_sum_a": float(point[band].sum()),
        "max_cell_difference_a": float(difference.max()),
        "band_peak_current_a": band_peak,
        "difference_relative_to_band_peak": (
            float(difference.max() / band_peak) if band_peak else None
        ),
        "raw_current_a": float(booked.sum()),
        "axis_flux_wb": axis,
        "boundary_flux_wb": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="edge")
    args = parser.parse_args()

    bench.build_profile = capture_build(bench.build_profile)
    bench.complete_profile_current_adapter = capture_adapter(
        bench.complete_profile_current_adapter
    )
    bench.ForwardProfile.solve = capture_solve(bench.ForwardProfile.solve)
    extent.main()
    receipt = discriminator()
    equilibrium = SOLVED.get("value")
    if equilibrium is not None:
        receipt.update(terminal(equilibrium))
    receipt.update(
        dict(
            frame=args.frame,
            branch=args.branch,
            revision=args.revision,
        )
    )
    (args.output / "finite-edge.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("FINITE_EDGE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
