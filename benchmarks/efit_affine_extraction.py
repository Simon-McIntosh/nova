"""Recover EFIT flux-function profiles from a stored poloidal-flux map.

The stored EFIT map is flux per toroidal radian.  It is converted to total
flux before the package conservation operator is applied, so Ampere's law is
read as ``mu0 * j_phi = -delta_star(Phi) / (2 pi R)``.  Fixed normalised-flux
bins then expose the two coefficients of
``R * j_phi = R**2 * p_prime + ff_prime / mu0``.

The command writes strict JSON to standard output.  In addition to regression
diagnostics, ``profiles`` contains bin-centred arrays that can be consumed
directly by a forward-input adapter.
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

from nova.equilibrium.conservation import FluxLattice  # noqa: E402
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR  # noqa: E402
from nova.equilibrium.map_extraction import (  # noqa: E402
    extract_flux_functions as extract_map_flux_functions,
)
from nova.equilibrium.wall_mask import inside_polygon  # noqa: E402
from nova.imas.mast_vacuum_cohort import SHOT_STORE  # noqa: E402

DEFAULT_SHOT = 21978
DEFAULT_SLICE_INDEX = 46
DEFAULT_BIN_COUNT = 24
MINIMUM_FITTED_BINS = 20
FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
THIRD_TERM_CELL_FLOOR = 39


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Return an endpoint-preserving float64 axis after checking uniformity."""

    stored = np.asarray(values)
    resolution = np.finfo(stored.dtype).eps
    values = np.asarray(stored, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate axis")
    expected = np.linspace(values[0], values[-1], values.size)
    tolerance = 16.0 * resolution * max(1.0, abs(values[-1]))
    if not np.allclose(values, expected, rtol=0.0, atol=tolerance):
        deviation = float(np.max(np.abs(values - expected)))
        raise ValueError(f"{name} is not uniform; maximum deviation is {deviation:.6g}")
    return expected


def _flux_map(group: zarr.Group, slice_index: int, radius: np.ndarray) -> np.ndarray:
    """Return one finite EFIT flux raster in ``(height, radius)`` order."""

    raw = np.asarray(group["psirz"][slice_index], dtype=np.float64)
    finite_columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if finite_columns.size != radius.size:
        raise ValueError(
            "the selected psirz slice does not carry exactly one finite column "
            f"per radial coordinate ({finite_columns.size} != {radius.size})"
        )
    if "profile_r" in group:
        profile_radius = np.asarray(group["profile_r"][:], dtype=np.float64)
        if not np.allclose(
            profile_radius[finite_columns], radius, rtol=2e-7, atol=1e-8
        ):
            raise ValueError("finite psirz columns do not match the EFIT radial grid")
    return raw[:, finite_columns]


def _radial_correction_diagnostic(
    radius: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    """Return the extra radial-dependence diagnostic beyond the production pair."""

    matrix = np.column_stack((radius**2, np.ones(radius.size), radius**4))
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    fitted = matrix @ coefficients
    residual = target - fitted
    residual_sum_squares = float(residual @ residual)
    target_offset = target - np.mean(target)
    total_sum_squares = float(target_offset @ target_offset)
    scatter = float(np.sqrt(residual_sum_squares / target.size))
    fitted_magnitude = float(np.sqrt(np.mean(fitted**2)))
    degrees_of_freedom = target.size - matrix.shape[1]
    if degrees_of_freedom > 0 and rank == matrix.shape[1]:
        variance = residual_sum_squares / degrees_of_freedom
        covariance = variance * np.linalg.pinv(matrix.T @ matrix)
        standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        standard_error = np.full(matrix.shape[1], np.nan)
    return {
        "coefficients": coefficients,
        "standard_error": standard_error,
        "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "scatter": scatter,
        "fitted_magnitude": fitted_magnitude,
        "scatter_fraction": scatter / fitted_magnitude
        if fitted_magnitude > 0.0
        else np.nan,
        "r_squared": 1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else np.nan,
    }


def _distribution(values: list[float]) -> dict[str, float]:
    """Return fixed quantiles for a finite scalar sample."""

    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("a reported distribution must be non-empty and finite")
    quantiles = np.quantile(sample, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("minimum", "q25", "median", "q75", "maximum"), quantiles.tolist()))


def _profile_comparison(
    group: zarr.Group,
    slice_index: int,
    centers: list[float],
    extracted: list[float],
    field: str,
) -> dict[str, Any]:
    """Compare an extracted bin-centred profile with a stored EFIT profile."""

    if field not in group or "psi_norm" not in group:
        return {"present": False, "reason": f"{field} or psi_norm is absent"}
    stored_grid = np.asarray(group["psi_norm"][:], dtype=np.float64)
    stored_profile = np.asarray(group[field][slice_index], dtype=np.float64)
    valid = np.isfinite(stored_grid) & np.isfinite(stored_profile)
    if np.count_nonzero(valid) < 2:
        return {"present": False, "reason": f"{field} has fewer than two finite points"}
    order = np.argsort(stored_grid[valid])
    reference = np.interp(
        np.asarray(centers), stored_grid[valid][order], stored_profile[valid][order]
    )
    extracted_array = np.asarray(extracted)
    scale = max(float(np.max(np.abs(reference))), np.finfo(np.float64).tiny)
    denominator = np.maximum(np.abs(reference), scale * 1.0e-12)
    signed_relative = (extracted_array - reference) / denominator
    return {
        "present": True,
        "stored_units": group[field].attrs.get("units"),
        "stored_at_bin_centers": reference.tolist(),
        "signed_difference": (extracted_array - reference).tolist(),
        "signed_relative_difference": signed_relative.tolist(),
        "signed_relative_difference_distribution": _distribution(
            signed_relative.tolist()
        ),
        "relative_denominator_floor": scale * 1.0e-12,
    }


def _stored_lcfs(group: zarr.Group, slice_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite stored EFIT LCFS vertices for one slice."""

    required = ("lcfs_r", "lcfs_z", "lcfsn_c")
    missing = [field for field in required if field not in group]
    if missing:
        raise ValueError(f"stored EFIT LCFS fields are absent: {', '.join(missing)}")
    count_value = float(group["lcfsn_c"][slice_index])
    if not np.isfinite(count_value) or count_value != int(count_value):
        raise ValueError(f"stored EFIT LCFS point count is invalid: {count_value}")
    count = int(count_value)
    radius = np.asarray(group["lcfs_r"][slice_index], dtype=np.float64)
    height = np.asarray(group["lcfs_z"][slice_index], dtype=np.float64)
    if count < 3 or count > radius.size or count > height.size:
        raise ValueError(
            f"stored EFIT LCFS count {count} does not fit coordinate arrays"
        )
    radius = radius[:count]
    height = height[:count]
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
        raise ValueError("stored EFIT LCFS contains non-finite vertices")
    return radius, height


def extract_efit_flux_functions(
    store: Path = SHOT_STORE,
    shot: int = DEFAULT_SHOT,
    slice_index: int = DEFAULT_SLICE_INDEX,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    """Extract bin-centred EFIT flux functions and regression diagnostics."""

    if bin_count < MINIMUM_FITTED_BINS:
        raise ValueError(f"bin_count must be at least {MINIMUM_FITTED_BINS}")
    shot_path = store / f"{shot}.zarr"
    group = zarr.open_group(shot_path.as_posix(), mode="r")["efm"]
    time = np.asarray(group["time"][:], dtype=np.float64)
    if slice_index < 0 or slice_index >= time.size:
        raise IndexError(f"slice index {slice_index} is outside 0..{time.size - 1}")

    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius = _uniform_axis(stored_radius, "gridr")
    height = _uniform_axis(stored_height, "gridz")
    psi_per_radian = _flux_map(group, slice_index, stored_radius)
    if psi_per_radian.shape != (height.size, radius.size):
        raise ValueError(
            "psirz finite raster shape does not match the EFIT grid: "
            f"{psi_per_radian.shape} != {(height.size, radius.size)}"
        )

    mesh = FluxLattice(radius, height)
    psi_flat = psi_per_radian.T.reshape(-1)
    psi_axis = float(group["psi_axis"][slice_index])
    psi_boundary = float(group["psi_boundary"][slice_index])
    flux_span = psi_boundary - psi_axis
    if not np.isfinite(flux_span) or flux_span == 0.0:
        raise ValueError("the selected slice has no finite non-zero plasma flux span")
    psi_normalised = (psi_flat - psi_axis) / flux_span
    lcfs_radius, lcfs_height = _stored_lcfs(group, slice_index)
    lcfs_interior = inside_polygon(
        mesh.coordinate[:, 0],
        mesh.coordinate[:, 1],
        lcfs_radius,
        lcfs_height,
    )
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    operator_interior = np.asarray(mesh.interior(), dtype=bool)
    plasma_seed = operator_interior & np.isfinite(psi_normalised) & lcfs_interior
    extraction = extract_map_flux_functions(
        radius,
        height,
        (TOTAL_FLUX_FACTOR * psi_flat).reshape(mesh.shape),
        psi_normalised.reshape(mesh.shape),
        surfaces=bin_centers,
        plasma_mask=plasma_seed.reshape(mesh.shape),
        min_samples=4,
        maximum_condition=np.inf,
        maximum_inflation=np.inf,
    )
    node_radius = mesh.node_radius
    current_density = extraction.current.toroidal_current_density.reshape(-1)
    operator_valid = (
        extraction.current.valid.reshape(-1)
        & np.isfinite(current_density)
        & np.isfinite(psi_normalised)
    )
    plasma = operator_valid & lcfs_interior
    exterior = operator_valid & ~lcfs_interior

    bins: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        upper_test = (
            psi_normalised <= upper
            if index == bin_count - 1
            else psi_normalised < upper
        )
        selected = plasma & (psi_normalised >= lower) & upper_test
        cell_count = int(np.count_nonzero(selected))
        if cell_count < 4:
            exclusions.append(
                {
                    "bin_index": index,
                    "lower": float(lower),
                    "upper": float(upper),
                    "cell_count": cell_count,
                    "reason": "fewer than four cells for the three-term fit",
                }
            )
            continue
        radial = node_radius[selected]
        target = radial * current_density[selected]
        three_term = _radial_correction_diagnostic(radial, target)
        extracted_pprime = -TOTAL_FLUX_FACTOR * extraction.p_prime[index]
        extracted_ffprime = -TOTAL_FLUX_FACTOR * extraction.ff_prime[index]
        if (
            not np.isfinite(extracted_pprime)
            or not np.isfinite(extracted_ffprime)
            or three_term["rank"] != 3
        ):
            exclusions.append(
                {
                    "bin_index": index,
                    "lower": float(lower),
                    "upper": float(upper),
                    "cell_count": cell_count,
                    "reason": "regression design matrix is rank deficient",
                }
            )
            continue
        fitted = radial**2 * extracted_pprime + extracted_ffprime / mu_0
        residual = target - fitted
        residual_sum_squares = float(residual @ residual)
        target_offset = target - np.mean(target)
        total_sum_squares = float(target_offset @ target_offset)
        scatter = float(extraction.projection_rms[index])
        fitted_magnitude = float(np.sqrt(np.mean(fitted**2)))
        scatter_fraction = (
            scatter / fitted_magnitude if fitted_magnitude > 0.0 else np.nan
        )
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else np.nan
        )
        r4_coefficient = float(three_term["coefficients"][2])
        r4_standard_error = float(three_term["standard_error"][2])
        scatter_reduction = (
            (scatter - three_term["scatter"]) / scatter if scatter > 0.0 else 0.0
        )
        bins.append(
            {
                "bin_index": index,
                "lower": float(lower),
                "upper": float(upper),
                "center": float((lower + upper) / 2.0),
                "cell_count": cell_count,
                "pprime": float(extracted_pprime),
                "intercept_ffprime_over_mu0": float(extracted_ffprime / mu_0),
                "ffprime": float(extracted_ffprime),
                "residual_scatter": scatter,
                "fitted_r_jphi_magnitude": fitted_magnitude,
                "residual_scatter_fraction": float(scatter_fraction),
                "r_squared": float(r_squared),
                "three_term_residual_scatter": float(three_term["scatter"]),
                "three_term_scatter_reduction_fraction": float(scatter_reduction),
                "r4_coefficient": r4_coefficient,
                "r4_standard_error": r4_standard_error,
                "r4_standard_error_ratio": abs(r4_coefficient) / r4_standard_error
                if r4_standard_error > 0.0
                else None,
                "r4_resolved_above_standard_error": abs(r4_coefficient)
                > r4_standard_error,
            }
        )

    if len(bins) < MINIMUM_FITTED_BINS:
        raise ValueError(
            f"only {len(bins)} of {bin_count} fixed bins could be fitted; "
            f"at least {MINIMUM_FITTED_BINS} are required"
        )

    centers = [row["center"] for row in bins]
    pprime = [row["pprime"] for row in bins]
    ffprime = [row["ffprime"] for row in bins]
    scatter_fraction = [row["residual_scatter_fraction"] for row in bins]
    scatter_reduction = [row["three_term_scatter_reduction_fraction"] for row in bins]
    r4_resolved = [row["r4_resolved_above_standard_error"] for row in bins]
    scatter_fraction_distribution = _distribution(scatter_fraction)
    scatter_reduction_distribution = _distribution(scatter_reduction)
    resolved_count = int(np.count_nonzero(r4_resolved))

    derived_current = float(np.sum(current_density[plasma] * mesh.cell_area[plasma]))
    exterior_current = float(
        np.sum(current_density[exterior] * mesh.cell_area[exterior])
    )
    exterior_absolute_current = float(
        np.sum(np.abs(current_density[exterior]) * mesh.cell_area[exterior])
    )
    stored_current_field = (
        "plasma_current_c" if "plasma_current_c" in group else "plasma_current_x"
    )
    stored_current = float(group[stored_current_field][slice_index])
    current_relative_difference = (derived_current - stored_current) / abs(
        stored_current
    )

    stored_density_comparison: dict[str, Any]
    if "plasma_current_rz" in group:
        stored_density = np.asarray(
            group["plasma_current_rz"][slice_index], dtype=np.float64
        ).T.reshape(-1)
        compare = plasma & np.isfinite(stored_density)
        difference = current_density[compare] - stored_density[compare]
        scale = float(np.max(np.abs(stored_density[compare])))
        stored_density_comparison = {
            "present": True,
            "cell_count": int(np.count_nonzero(compare)),
            "normalised_rms_difference": float(np.sqrt(np.mean(difference**2)) / scale),
            "correlation": float(
                np.corrcoef(current_density[compare], stored_density[compare])[0, 1]
            ),
        }
    else:
        stored_density_comparison = {"present": False}

    return {
        "source": {
            "shot_path": shot_path.as_posix(),
            "group": "efm",
            "shot": shot,
            "slice_index": slice_index,
            "time_s": float(time[slice_index]),
            "psirz_units": group["psirz"].attrs.get("units"),
            "psi_axis_per_radian": psi_axis,
            "psi_boundary_per_radian": psi_boundary,
        },
        "method": {
            "operator": "nova.equilibrium.conservation.delta_star",
            "current_relation": "mu0*jphi=-delta_star(Phi)/(2*pi*R)",
            "stored_flux_to_total_flux_factor": TOTAL_FLUX_FACTOR,
            "raster_input_order": "height,radius",
            "operator_order": "radius,height",
            "bin_edges": bin_edges.tolist(),
            "declared_bin_count": bin_count,
            "operator_stencil_margin_cells": 2,
            "plasma_region": "cell centers inside stored EFIT LCFS polygon",
            "lcfs_source_fields": ["lcfs_r", "lcfs_z", "lcfsn_c"],
            "lcfs_vertex_count": int(lcfs_radius.size),
        },
        "profiles": {
            "psi_normalised": centers,
            "pprime": pprime,
            "pprime_units": "Pa-rad/Wb",
            "ffprime": ffprime,
            "ffprime_units": group["ffprime"].attrs.get("units")
            if "ffprime" in group
            else "T-rad",
            "representation": "piecewise values at fixed bin centers",
        },
        "bins": bins,
        "excluded_bins": exclusions,
        "summary": {
            "fitted_bin_count": len(bins),
            "excluded_bin_count": len(exclusions),
            "residual_scatter_fraction_distribution": scatter_fraction_distribution,
            "three_term_scatter_reduction_fraction_distribution": (
                scatter_reduction_distribution
            ),
            "r4_resolved_above_standard_error_count": resolved_count,
            "r4_resolved_above_standard_error_fraction": float(np.mean(r4_resolved)),
            "two_term_sufficiency": {
                "verdict": "not sufficient across all fixed bins"
                if resolved_count > 0
                else "no additional radial dependence resolved",
                "basis": (
                    f"R^4 exceeds its standard error in {resolved_count}/{len(bins)} "
                    "bins; two-term scatter fractions and three-term reductions are "
                    "reported by fixed quantiles"
                ),
            },
        },
        "stored_profile_comparison": {
            "pprime": _profile_comparison(
                group, slice_index, centers, pprime, "pprime"
            ),
            "ffprime": _profile_comparison(
                group, slice_index, centers, ffprime, "ffprime"
            ),
        },
        "current_integral": {
            "plasma_cell_count": int(np.count_nonzero(plasma)),
            "derived_current_a": derived_current,
            "stored_current_field": stored_current_field,
            "stored_current_a": stored_current,
            "signed_relative_difference": current_relative_difference,
            "measured_current_a": float(group["plasma_current_x"][slice_index])
            if "plasma_current_x" in group
            else None,
        },
        "exterior_current": {
            "cell_count": int(np.count_nonzero(exterior)),
            "signed_current_a": exterior_current,
            "absolute_current_a": exterior_absolute_current,
            "region": "operator-stencil interior outside the stored EFIT LCFS",
        },
        "stored_current_density_comparison": stored_density_comparison,
    }


def _cohort_profile_summary(
    comparison: dict[str, Any], control: dict[str, float]
) -> dict[str, Any]:
    """Summarise one pointwise profile comparison against the control spread."""

    distribution = comparison["signed_relative_difference_distribution"]
    q25 = float(distribution["q25"])
    median = float(distribution["median"])
    q75 = float(distribution["q75"])
    width = q75 - q25
    control_magnitude = max(abs(control["median"]), np.finfo(np.float64).tiny)
    control_width = control["q75"] - control["q25"]
    return {
        "signed_relative_median": median,
        "signed_relative_q25": q25,
        "signed_relative_q75": q75,
        "signed_relative_iqr_width": width,
        "median_absolute_magnitude_factor_vs_control": abs(median) / control_magnitude,
        "iqr_width_factor_vs_control": width / control_width,
        "median_within_control_iqr": control["q25"] <= median <= control["q75"],
    }


def extract_affine_cohort(
    store: Path = SHOT_STORE,
    shots: tuple[int, ...] = FROZEN_SHOTS,
    slice_index: int = DEFAULT_SLICE_INDEX,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    """Run the landed affine extraction on one common slice per frozen shot."""

    reports = [
        extract_efit_flux_functions(store, shot, slice_index, bin_count)
        for shot in shots
    ]
    control_report = next(
        (report for report in reports if report["source"]["shot"] == DEFAULT_SHOT),
        None,
    )
    if control_report is None:
        raise ValueError(f"cohort must contain control shot {DEFAULT_SHOT}")
    control_profiles = {
        field: control_report["stored_profile_comparison"][field][
            "signed_relative_difference_distribution"
        ]
        for field in ("pprime", "ffprime")
    }

    rows = []
    departures = []
    for report in reports:
        shot = int(report["source"]["shot"])
        populations = [int(row["cell_count"]) for row in report["bins"]]
        below_floor = [
            int(row["bin_index"])
            for row in report["bins"]
            if row["cell_count"] < THIRD_TERM_CELL_FLOOR
        ]
        profiles = {
            field: _cohort_profile_summary(
                report["stored_profile_comparison"][field],
                control_profiles[field],
            )
            for field in ("pprime", "ffprime")
        }
        row = {
            "shot": shot,
            "slice_index": int(report["source"]["slice_index"]),
            "time_s": float(report["source"]["time_s"]),
            "is_control": shot == DEFAULT_SHOT,
            "interior_cell_count": int(report["current_integral"]["plasma_cell_count"]),
            "fitted_bin_count": len(report["bins"]),
            "excluded_bin_count": len(report["excluded_bins"]),
            "bin_population": _distribution(populations),
            "bins_below_third_term_cell_floor": below_floor,
            "bin_count_below_third_term_cell_floor": len(below_floor),
            "profiles": profiles,
        }
        rows.append(row)
        if shot != DEFAULT_SHOT:
            for field, summary in profiles.items():
                if not summary["median_within_control_iqr"]:
                    departures.append(
                        {
                            "shot": shot,
                            "profile": field,
                            "median": summary["signed_relative_median"],
                            "control_q25": control_profiles[field]["q25"],
                            "control_q75": control_profiles[field]["q75"],
                            "median_absolute_magnitude_factor_vs_control": summary[
                                "median_absolute_magnitude_factor_vs_control"
                            ],
                            "iqr_width_factor_vs_control": summary[
                                "iqr_width_factor_vs_control"
                            ],
                        }
                    )

    return {
        "method": {
            "shots": list(shots),
            "common_slice_index": slice_index,
            "control_shot": DEFAULT_SHOT,
            "bin_count": bin_count,
            "binning": "fixed width over 0 <= psi_normalised <= 1",
            "plasma_mask": "stored EFIT LCFS interior",
            "third_term_cell_floor": THIRD_TERM_CELL_FLOOR,
            "departure_rule": "shot median lies outside the control IQR",
        },
        "shots": rows,
        "cohort": {
            "shot_count": len(rows),
            "agreement_consistent_with_control_spread": not departures,
            "departures": departures,
            "shots_with_bins_below_third_term_cell_floor": [
                row["shot"]
                for row in rows
                if row["bin_count_below_third_term_cell_floor"] > 0
            ],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--shots", type=int, nargs="+", default=FROZEN_SHOTS)
    parser.add_argument("--slice-index", type=int, default=DEFAULT_SLICE_INDEX)
    parser.add_argument("--bins", type=int, default=DEFAULT_BIN_COUNT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = extract_affine_cohort(
        args.store, tuple(args.shots), args.slice_index, args.bins
    )
    json.dump(report, fp=sys.stdout, indent=2, allow_nan=False)
    print()


if __name__ == "__main__":
    main()
