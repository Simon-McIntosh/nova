"""Separate physical and numerical readings of an affine-regression third term.

The control reproduces the fixed-width normalised-flux bins used to recover
EFIT's flux functions.  The intervention sorts the same stored-LCFS-interior
cells by normalised flux and divides them into equal-population bins.  A
rotation-like physical term must retain a consistent coefficient sign and
magnitude; alternating signs identify a numerical regression degree of
freedom instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import zarr
from scipy.constants import mu_0

jax.config.update("jax_enable_x64", True)

from benchmarks.efit_affine_extraction import (  # noqa: E402
    DEFAULT_BIN_COUNT,
    DEFAULT_SHOT,
    DEFAULT_SLICE_INDEX,
    _fit,
    _flux_map,
    _stored_lcfs,
    _uniform_axis,
)
from nova.equilibrium.conservation import FluxLattice, delta_star  # noqa: E402
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR  # noqa: E402
from nova.equilibrium.wall_mask import inside_polygon  # noqa: E402
from nova.imas.mast_vacuum_cohort import SHOT_STORE  # noqa: E402


def _regression(
    indices: np.ndarray,
    psi_normalised: np.ndarray,
    radius: np.ndarray,
    current_density: np.ndarray,
    bin_index: int,
) -> dict[str, Any]:
    """Fit the two radial models on one declared cell population."""

    cell_radius = radius[indices]
    target = cell_radius * current_density[indices]
    two_term = _fit(np.column_stack((cell_radius**2, np.ones(indices.size))), target)
    three_term = _fit(
        np.column_stack((cell_radius**2, np.ones(indices.size), cell_radius**4)),
        target,
    )
    if two_term["rank"] != 2 or three_term["rank"] != 3:
        raise ValueError(f"bin {bin_index} regression is rank deficient")
    coefficient = float(three_term["coefficients"][2])
    standard_error = float(three_term["standard_error"][2])
    return {
        "bin_index": bin_index,
        "psi_min": float(np.min(psi_normalised[indices])),
        "psi_mean": float(np.mean(psi_normalised[indices])),
        "psi_max": float(np.max(psi_normalised[indices])),
        "cell_count": int(indices.size),
        "r4_coefficient": coefficient,
        "r4_standard_error": standard_error,
        "r4_standard_error_ratio": abs(coefficient) / standard_error,
        "r4_resolved_above_standard_error": abs(coefficient) > standard_error,
        "r4_sign": int(np.sign(coefficient)),
        "two_term_scatter_fraction": float(two_term["scatter_fraction"]),
        "two_term_r_squared": float(two_term["r_squared"]),
        "three_term_scatter": float(three_term["scatter"]),
        "three_term_scatter_reduction_fraction": float(
            (two_term["scatter"] - three_term["scatter"]) / two_term["scatter"]
        ),
    }


def _fixed_width_bins(
    plasma: np.ndarray,
    psi_normalised: np.ndarray,
    radius: np.ndarray,
    current_density: np.ndarray,
    bin_count: int,
) -> list[dict[str, Any]]:
    """Fit the fixed-width control partition over zero-to-one flux."""

    rows = []
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        upper_test = (
            psi_normalised <= upper
            if bin_index == bin_count - 1
            else psi_normalised < upper
        )
        selected = plasma & (psi_normalised >= lower) & upper_test
        indices = np.flatnonzero(selected)
        if indices.size < 4:
            raise ValueError(
                f"fixed-width bin {bin_index} has only {indices.size} cells"
            )
        row = _regression(indices, psi_normalised, radius, current_density, bin_index)
        row["declared_psi_lower"] = float(lower)
        row["declared_psi_upper"] = float(upper)
        rows.append(row)
    return rows


def _equal_population_bins(
    plasma: np.ndarray,
    psi_normalised: np.ndarray,
    radius: np.ndarray,
    current_density: np.ndarray,
    bin_count: int,
) -> list[dict[str, Any]]:
    """Fit a deterministic stable-sort partition with equal cell populations."""

    plasma_indices = np.flatnonzero(plasma)
    order = np.argsort(psi_normalised[plasma_indices], kind="stable")
    populations = np.array_split(plasma_indices[order], bin_count)
    return [
        _regression(indices, psi_normalised, radius, current_density, bin_index)
        for bin_index, indices in enumerate(populations)
    ]


def _distribution(values: list[float]) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(sample, (0.0, 0.25, 0.5, 0.75, 1.0))
    return dict(zip(("minimum", "q25", "median", "q75", "maximum"), quantiles.tolist()))


def separate_third_term(
    store: Path = SHOT_STORE,
    shot: int = DEFAULT_SHOT,
    slice_index: int = DEFAULT_SLICE_INDEX,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    """Measure third-term stability under fixed-width and equal-population bins."""

    if bin_count < 20:
        raise ValueError("at least 20 bins are required")
    group = zarr.open_group((store / f"{shot}.zarr").as_posix(), mode="r")["efm"]
    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius_axis = _uniform_axis(stored_radius, "gridr")
    height_axis = _uniform_axis(stored_height, "gridz")
    mesh = FluxLattice(radius_axis, height_axis)
    psi_per_radian = _flux_map(group, slice_index, stored_radius).T.reshape(-1)
    total_flux = TOTAL_FLUX_FACTOR * psi_per_radian
    current_density = -np.asarray(delta_star(mesh, total_flux)) / (
        TOTAL_FLUX_FACTOR * mu_0 * mesh.node_radius
    )
    psi_axis = float(group["psi_axis"][slice_index])
    psi_boundary = float(group["psi_boundary"][slice_index])
    psi_normalised = (psi_per_radian - psi_axis) / (psi_boundary - psi_axis)
    lcfs_radius, lcfs_height = _stored_lcfs(group, slice_index)
    plasma = (
        np.asarray(mesh.interior())
        & np.isfinite(current_density)
        & np.isfinite(psi_normalised)
        & inside_polygon(
            mesh.coordinate[:, 0],
            mesh.coordinate[:, 1],
            lcfs_radius,
            lcfs_height,
        )
    )

    fixed = _fixed_width_bins(
        plasma, psi_normalised, mesh.node_radius, current_density, bin_count
    )
    equal = _equal_population_bins(
        plasma, psi_normalised, mesh.node_radius, current_density, bin_count
    )
    fixed_resolved = [row for row in fixed if row["r4_resolved_above_standard_error"]]
    equal_resolved = [row for row in equal if row["r4_resolved_above_standard_error"]]
    equal_positive = sum(row["r4_sign"] > 0 for row in equal)
    equal_negative = sum(row["r4_sign"] < 0 for row in equal)
    resolved_positive = sum(row["r4_sign"] > 0 for row in equal_resolved)
    resolved_negative = sum(row["r4_sign"] < 0 for row in equal_resolved)
    coefficient_sign_consistent = equal_positive == 0 or equal_negative == 0
    significant_sign_consistent = resolved_positive == 0 or resolved_negative == 0
    significance_count_fell = len(equal_resolved) < len(fixed_resolved)
    physical = (
        not significance_count_fell
        and coefficient_sign_consistent
        and significant_sign_consistent
    )
    population_floor = min(row["cell_count"] for row in equal)

    return {
        "source": {
            "shot": shot,
            "slice_index": slice_index,
            "time_s": float(group["time"][slice_index]),
            "lcfs_vertex_count": int(lcfs_radius.size),
            "plasma_cell_count": int(np.count_nonzero(plasma)),
        },
        "method": {
            "bin_count": bin_count,
            "control": "equal width over 0 <= psi_normalised <= 1",
            "intervention": (
                "stable sort by psi_normalised followed by numpy.array_split"
            ),
            "significance_rule": "abs(R^4 coefficient) > its standard error",
        },
        "fixed_width_bins": fixed,
        "fixed_width_resolved_bins": fixed_resolved,
        "equal_population_bins": equal,
        "summary": {
            "fixed_width_resolved_count": len(fixed_resolved),
            "equal_population_resolved_count": len(equal_resolved),
            "significance_count_fell": significance_count_fell,
            "equal_population_cell_count_distribution": _distribution(
                [row["cell_count"] for row in equal]
            ),
            "equal_population_positive_count": equal_positive,
            "equal_population_negative_count": equal_negative,
            "equal_population_resolved_positive_count": resolved_positive,
            "equal_population_resolved_negative_count": resolved_negative,
            "coefficient_sign_consistent": coefficient_sign_consistent,
            "significant_coefficient_sign_consistent": significant_sign_consistent,
            "fixed_width_two_term_scatter_fraction_distribution": _distribution(
                [row["two_term_scatter_fraction"] for row in fixed]
            ),
            "equal_population_r4_coefficient_distribution": _distribution(
                [row["r4_coefficient"] for row in equal]
            ),
        },
        "verdict": {
            "reading": "physical" if physical else "numerical",
            "basis": (
                "equal-population coefficients retain one sign and formal "
                "significance does not fall"
                if physical
                else (
                    f"equal-population coefficients alternate sign "
                    f"({equal_positive} positive, {equal_negative} negative; "
                    f"significant subset {resolved_positive} positive, "
                    f"{resolved_negative} negative)"
                )
            ),
            "third_forward_input": {
                "kind": "candidate_rotation_profile",
                "values": [row["r4_coefficient"] for row in equal],
            }
            if physical
            else None,
            "two_term_model_stands": not physical,
            "minimum_cells_for_third_term_interpretation": population_floor,
            "population_floor_qualification": (
                "below the smallest equal-population bin, a third coefficient "
                "is not interpreted; at the floor, significance still requires "
                "cross-bin sign consistency"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--slice-index", type=int, default=DEFAULT_SLICE_INDEX)
    parser.add_argument("--bins", type=int, default=DEFAULT_BIN_COUNT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = separate_third_term(args.store, args.shot, args.slice_index, args.bins)
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
