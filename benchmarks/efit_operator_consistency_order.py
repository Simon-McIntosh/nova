"""Measure grid convergence of the Green-coupling/delta-star operator pair.

The benchmark repeats one LCFS-masked EFIT plasma-current round trip on a
geometrically fixed sequence of Nova grids.  It reports the finite-cell Green
coupling followed by conservation ``delta_star`` separately from the extra
field created by remasking recovered current to the LCFS.  Consequently a
resolution-dependent change in admitted LCFS cells cannot masquerade as
operator convergence.

The stored-to-Nova-to-stored bilinear interpolation receipt is also reported
for every grid, but is not included in either operator contribution or its
fitted convergence order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy import stats

from benchmarks.efit_flux_decomposition import (
    DEFAULT_SHOT,
    _density_from_flux,
    _evaluate_on_grid,
    _flux_interpolation_receipt,
    _interpolator,
    _lcfs_mask,
    _plasma_flux,
    _read_stored_slice,
)
from nova.equilibrium.conservation import FluxLattice
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_solve_inputs import SHOT_STORE

CONTROL_GRID = (33, 49)
CONTROL_ERROR_FRACTION = 5.152548851379959e-3
GRID_SEQUENCE = ((17, 25), (25, 37), CONTROL_GRID, (41, 61))


def _sup_fraction(field: np.ndarray, span: float) -> float:
    """Return the sup norm of a field divided by the stored flux span."""

    return float(np.max(np.abs(field)) / span)


def _fit_power_order(
    cell_sizes_m: np.ndarray, error_fractions: np.ndarray
) -> dict[str, float | list[float]]:
    """Fit ``error = coefficient * cell_size**order`` in log space."""

    fit = stats.linregress(np.log(cell_sizes_m), np.log(error_fractions))
    predicted_log = fit.intercept + fit.slope * np.log(cell_sizes_m)
    residual = np.log(error_fractions) - predicted_log
    degrees_of_freedom = cell_sizes_m.size - 2
    confidence_multiplier = float(stats.t.ppf(0.975, degrees_of_freedom))
    order_interval = [
        float(fit.slope - confidence_multiplier * fit.stderr),
        float(fit.slope + confidence_multiplier * fit.stderr),
    ]
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "order_95_percent_interval": order_interval,
        "coefficient": float(np.exp(fit.intercept)),
        "log_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "fraction_residual_rms": float(
            np.sqrt(np.mean((error_fractions - np.exp(predicted_log)) ** 2))
        ),
        "r_squared": float(fit.rvalue**2),
    }


def _measure_resolution(
    *,
    shot: int,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
    stored: Any,
    stored_density: np.ndarray,
    span: float,
    radial_points: int,
    vertical_points: int,
) -> dict[str, Any]:
    """Measure the exact pair/remask vector decomposition on one Nova grid."""

    chain = build_mast_parity_chain(
        shot,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
        radial_points=radial_points,
        vertical_points=vertical_points,
    )
    profile = chain.profile_solver
    radius = np.asarray(profile.grid_r, dtype=float)
    height = np.asarray(profile.grid_z, dtype=float)
    lattice = FluxLattice(radius, height)
    lcfs = _lcfs_mask(
        radius,
        height,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    valid = np.asarray(lattice.interior(margin=2), dtype=bool).reshape(lattice.shape)

    mapped_density = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored_density.T),
        radius,
        height,
    ).T
    mapped_density = np.where(lcfs, mapped_density, 0.0)
    plasma_flux = _plasma_flux(profile, lattice, mapped_density)

    recovered_density, recovered_valid = _density_from_flux(lattice, plasma_flux)
    recovered_complete = np.where(recovered_valid, recovered_density, 0.0)
    recovered_masked = np.where(recovered_valid & lcfs, recovered_density, 0.0)
    complete_reconstruction = _plasma_flux(profile, lattice, recovered_complete)
    masked_reconstruction = _plasma_flux(profile, lattice, recovered_masked)

    pair_error = plasma_flux - complete_reconstruction
    remask_error = complete_reconstruction - masked_reconstruction
    masked_total_error = plasma_flux - masked_reconstruction
    closure = masked_total_error - pair_error - remask_error

    mapped_flux = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored.total_flux_wb),
        radius,
        height,
    )
    interpolation = _flux_interpolation_receipt(stored, radius, height, mapped_flux)
    radial_cell_size = float(radius[1] - radius[0])
    vertical_cell_size = float(height[1] - height[0])
    characteristic_cell_size = float(np.sqrt(radial_cell_size * vertical_cell_size))
    control = (radial_points, vertical_points) == CONTROL_GRID
    total_fraction = _sup_fraction(masked_total_error, span)
    return {
        "radial_points": radial_points,
        "vertical_points": vertical_points,
        "cell_count": radial_points * vertical_points,
        "radial_cell_size_m": radial_cell_size,
        "vertical_cell_size_m": vertical_cell_size,
        "characteristic_cell_size_m": characteristic_cell_size,
        "linear_point_count": float(np.sqrt(radial_points * vertical_points)),
        "is_control_grid": control,
        "lcfs_admitted_cells": int(np.count_nonzero(lcfs)),
        "operator_valid_lcfs_cells": int(np.count_nonzero(valid & lcfs)),
        "lcfs_admitted_fraction": float(np.mean(lcfs)),
        "green_delta_pair_error_fraction": _sup_fraction(pair_error, span),
        "lcfs_remask_error_fraction": _sup_fraction(remask_error, span),
        "masked_round_trip_error_fraction": total_fraction,
        "vector_closure_fraction": _sup_fraction(closure, span),
        "control_reference_error_fraction": (
            CONTROL_ERROR_FRACTION if control else None
        ),
        "control_reproduction_difference": (
            float(total_fraction - CONTROL_ERROR_FRACTION) if control else None
        ),
        "interpolation": interpolation,
    }


def measure_consistency_order(
    *,
    shot: int,
    slice_index: int,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
) -> dict[str, Any]:
    """Measure and classify Green/delta-star consistency under refinement."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    requested_time = float(group["time"][slice_index])
    stored = _read_stored_slice(group, requested_time)
    if stored.index != slice_index:
        raise ValueError(
            f"requested slice {slice_index} resolved to usable slice {stored.index}"
        )
    span = float(np.ptp(stored.total_flux_wb))
    stored_lattice = FluxLattice(stored.radius_m, stored.height_m)
    stored_density_total, stored_valid = _density_from_flux(
        stored_lattice, stored.total_flux_wb
    )
    stored_lcfs = _lcfs_mask(
        stored.radius_m,
        stored.height_m,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    stored_density = np.where(stored_valid & stored_lcfs, stored_density_total, 0.0)

    resolutions = [
        _measure_resolution(
            shot=shot,
            store=store,
            artifact_cache=artifact_cache,
            artifact_digest=artifact_digest,
            stored=stored,
            stored_density=stored_density,
            span=span,
            radial_points=radial_points,
            vertical_points=vertical_points,
        )
        for radial_points, vertical_points in GRID_SEQUENCE
    ]
    cell_sizes = np.asarray(
        [row["characteristic_cell_size_m"] for row in resolutions], dtype=float
    )
    pair_errors = np.asarray(
        [row["green_delta_pair_error_fraction"] for row in resolutions],
        dtype=float,
    )
    masked_errors = np.asarray(
        [row["masked_round_trip_error_fraction"] for row in resolutions],
        dtype=float,
    )
    pair_fit = _fit_power_order(cell_sizes, pair_errors)
    masked_fit = _fit_power_order(cell_sizes, masked_errors)
    pair_monotone = bool(np.all(np.diff(pair_errors) < 0.0))
    positive_order = bool(pair_fit["order_95_percent_interval"][0] > 0.0)
    discretisation = pair_monotone and positive_order
    finest_pair_error = float(pair_errors[-1])
    plateau = float(np.mean(pair_errors[-2:]))

    return {
        "source": {
            "shot": shot,
            "slice_index": stored.index,
            "time_s": stored.time_s,
            "stored_flux_span_wb": span,
            "stored_grid_shape_zr": list(stored.total_flux_wb.shape),
        },
        "grid_series": {
            "fixed_extent": (
                "MAST limiter bounding box plus the production 0.02 m margin"
            ),
            "resolution_count": len(resolutions),
            "linear_point_count_factor": float(
                resolutions[-1]["linear_point_count"]
                / resolutions[0]["linear_point_count"]
            ),
            "resolutions": resolutions,
        },
        "convergence": {
            "quantity_fitted": (
                "Green-coupling followed by conservation delta-star sup error; "
                "LCFS remasking and interpolation excluded"
            ),
            "green_delta_pair": pair_fit,
            "masked_round_trip_for_context": masked_fit,
            "pair_error_falls_monotonically": pair_monotone,
            "pair_order_statistically_positive": positive_order,
            "finest_pair_error_fraction": finest_pair_error,
            "verdict": "DISCRETISATION" if discretisation else "DEFECT",
            "extrapolated_zero_cell_size_limit_fraction": (
                0.0 if discretisation else None
            ),
            "plateau_fraction_finest_two": None if discretisation else plateau,
            "interpretation": (
                "the pair error falls monotonically at a statistically positive "
                "order and the fitted power law extrapolates to zero"
                if discretisation
                else "the pair error does not show statistically positive, "
                "monotone convergence and is read as a nonzero-grid defect"
            ),
        },
        "control": {
            "grid_shape_rz": list(CONTROL_GRID),
            "required_reference_fraction": CONTROL_ERROR_FRACTION,
            "reproduced_fraction": next(
                row["masked_round_trip_error_fraction"]
                for row in resolutions
                if row["is_control_grid"]
            ),
        },
        "policy": (
            "measurement only: no bound is registered, moved, or applied; "
            "the existing operator identity requirement remains unchanged"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--slice", type=int, default=46)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    return parser


def main() -> None:
    """Print the convergence receipt as stable JSON."""

    arguments = _parser().parse_args()
    result = measure_consistency_order(
        shot=arguments.shot,
        slice_index=arguments.slice,
        store=arguments.store,
        artifact_cache=arguments.artifact_cache,
        artifact_digest=arguments.artifact_digest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
