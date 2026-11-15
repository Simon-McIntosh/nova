"""Solve one finite-edge row and read its source-cutoff jump discriminator.

The solve, the seeding and the branch receipts are the profile-extent cohort's
edge arm, run with the DIII-D consumer on the booked finite-edge route. After
that row lands this driver reads the same built profile at the terminal state
and reports the current the booked route gives the cells that straddle the
level, beside the value-only route's assignment there.
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


def discriminator() -> dict[str, object]:
    """Return the booked and value-only current of the boundary-band cells."""

    import jax.numpy as jnp

    built = BUILT.get("value")
    state = SOLVED.get("value")
    if built is None or state is None:
        return {"status": "unavailable"}
    operator = built[0].operator
    psi = jnp.asarray(np.asarray(state).reshape(-1))
    axis = float(np.asarray(operator.declared_axis_flux))
    boundary = float(np.asarray(operator.declared_boundary_flux))
    physical = np.asarray(psi)[: operator.grid.node_number]
    normalised = (physical - axis) / (boundary - axis)
    band = np.abs(normalised - 1.0) <= BAND
    if not band.any():
        return {"status": "empty band"}
    booked = np.asarray(operator.cell_current(psi))
    point = np.asarray(
        replace(operator, use_linear_moments=False, sample=None).cell_current(psi)
    )
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
    solve = bench.ForwardProfile.solve

    def capture_solve(profile, request=None, *a, **kw):
        result = solve(profile, request, *a, **kw)
        SOLVED["value"] = getattr(result, "state", result)
        return result

    bench.ForwardProfile.solve = capture_solve
    extent.main()
    receipt = discriminator()
    receipt.update(
        dict(
            frame=args.frame,
            branch=args.branch,
            revision=args.revision,
            moment_route=bench.FINITE_EDGE_MOMENT_ROUTE,
        )
    )
    (args.output / "finite-edge.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("FINITE_EDGE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()