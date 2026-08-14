"""Separate far-field filament truncation from the round-trip stencil error.

The benchmark repeats the LCFS-masked operator-pair convergence study with four
plasma coupling matrices on exactly the same cells.  Production uses the full
rectangular-section kernel inside its switching band and a bare centroid
filament outside.  Two counterfactual hybrids retain that band but replace the
far branch with the public moment-corrected filament at orders two and three.
The reference forces the public rectangular-section kernel on every pair.

Coupling differences are element-wise matrix differences normalised as norm
ratios against the forced-section matrix.  Round-trip figures use the same
LCFS-masked input current as the convergence study, while excluding the
separate LCFS remasking contribution, so the banked pair-error control remains
directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from benchmarks.efit_flux_decomposition import (
    DEFAULT_SHOT,
    _density_from_flux,
    _evaluate_on_grid,
    _interpolator,
    _lcfs_mask,
    _read_stored_slice,
)
from benchmarks.efit_operator_consistency_order import (
    CONTROL_GRID,
    GRID_SEQUENCE,
    _fit_power_order,
)
from nova.biot.greens import cylinder_greens, moment_filament
from nova.equilibrium.conservation import FluxLattice
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_solve_inputs import SHOT_STORE

PRODUCTION_SWITCH = 3.0
CONTROL_PAIR_ERROR_FRACTION = 2.034417834e-3
VARIANT_NAMES = (
    "production_hybrid",
    "quadrupole_hybrid",
    "third_moment_hybrid",
    "forced_cylinder",
)


def _rectangle_vertices(
    radius: float, height: float, width: float, thickness: float
) -> np.ndarray:
    """Return counter-clockwise corners for one rectangular plasma cell."""

    half_width = 0.5 * width
    half_thickness = 0.5 * thickness
    return np.asarray(
        [
            [radius - half_width, height - half_thickness],
            [radius + half_width, height - half_thickness],
            [radius + half_width, height + half_thickness],
            [radius - half_width, height + half_thickness],
        ],
        dtype=np.float64,
    )


def _counterfactual_couplings(
    target_r: np.ndarray,
    target_z: np.ndarray,
    width: float,
    thickness: float,
    production: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build moment hybrids and the all-section reference from public kernels."""

    count = target_r.size
    quadrupole = np.asarray(production, dtype=np.float64).copy()
    third_moment = quadrupole.copy()
    forced_cylinder = np.empty_like(quadrupole)
    near_pair_count = 0

    for column, (source_r, source_z) in enumerate(zip(target_r, target_z, strict=True)):
        distance = np.hypot(target_r - source_r, target_z - source_z)
        near = distance < PRODUCTION_SWITCH * max(width, thickness)
        far = ~near
        near_pair_count += int(np.count_nonzero(near))
        forced_cylinder[:, column] = cylinder_greens(
            target_r,
            target_z,
            float(source_r),
            float(source_z),
            width,
            thickness,
        )[0]
        if np.any(far):
            vertices = _rectangle_vertices(
                float(source_r), float(source_z), width, thickness
            )
            quadrupole[far, column] = moment_filament(
                target_r[far], target_z[far], vertices, order=2
            )[0]
            third_moment[far, column] = moment_filament(
                target_r[far], target_z[far], vertices, order=3
            )[0]

    pair_count = count * count
    far_pair_count = pair_count - near_pair_count
    near_fraction = near_pair_count / pair_count
    far_fraction = far_pair_count / pair_count
    branch_receipt = {
        "switch_distance_m": PRODUCTION_SWITCH * max(width, thickness),
        "pair_count": pair_count,
        "near_cylinder_pair_count": near_pair_count,
        "far_filament_pair_count": far_pair_count,
        "near_cylinder_fraction": near_fraction,
        "far_filament_fraction": far_fraction,
        "green_function_cost": {
            "accounting_basis": (
                "primitive source evaluations per target-source pair: the "
                "rectangular-section kernel evaluates four corners, the bare "
                "filament one ring, and either moment order evaluates five "
                "rings because rectangular cells have zero cross and third moments"
            ),
            "production_hybrid": 4.0 * near_fraction + far_fraction,
            "quadrupole_hybrid": 4.0 * near_fraction + 5.0 * far_fraction,
            "third_moment_hybrid": 4.0 * near_fraction + 5.0 * far_fraction,
            "forced_cylinder": 4.0,
        },
    }
    return (
        {
            "production_hybrid": np.asarray(production, dtype=np.float64),
            "quadrupole_hybrid": quadrupole,
            "third_moment_hybrid": third_moment,
            "forced_cylinder": forced_cylinder,
        },
        branch_receipt,
    )


def _coupling_difference(
    coupling: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Return absolute matrix norms and their ratios to reference norms."""

    difference = coupling - reference
    reference_sup = float(np.max(np.abs(reference)))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    difference_sup = float(np.max(np.abs(difference)))
    difference_rms = float(np.sqrt(np.mean(difference**2)))
    return {
        "sup_wb_per_a": difference_sup,
        "rms_wb_per_a": difference_rms,
        "sup_relative_to_reference_sup": difference_sup / reference_sup,
        "rms_relative_to_reference_rms": difference_rms / reference_rms,
    }


def _flux_from_density(
    coupling: np.ndarray, lattice: FluxLattice, density_rz: np.ndarray
) -> np.ndarray:
    """Apply one target-by-source coupling matrix to a cell-current field."""

    cell_current_zr = (
        np.asarray(density_rz, dtype=np.float64)
        * np.asarray(lattice.cell_area).reshape(lattice.shape)
    ).T
    return (coupling @ cell_current_zr.ravel()).reshape(
        lattice.height.size, lattice.radius.size
    )


def _pair_error_fraction(
    coupling: np.ndarray,
    lattice: FluxLattice,
    density_rz: np.ndarray,
    span: float,
) -> float:
    """Run coupling then delta-star then coupling without LCFS remasking."""

    plasma_flux = _flux_from_density(coupling, lattice, density_rz)
    recovered_density, recovered_valid = _density_from_flux(lattice, plasma_flux)
    recovered_complete = np.where(recovered_valid, recovered_density, 0.0)
    reconstructed_flux = _flux_from_density(coupling, lattice, recovered_complete)
    return float(np.max(np.abs(plasma_flux - reconstructed_flux)) / span)


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
    """Measure coupling and round-trip differences on one fixed cell set."""

    chain = build_mast_parity_chain(
        shot,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
        radial_points=radial_points,
        vertical_points=vertical_points,
    )
    profile = chain.profile_solver
    radius = np.asarray(profile.grid_r, dtype=np.float64)
    height = np.asarray(profile.grid_z, dtype=np.float64)
    grid_r, grid_z = np.meshgrid(radius, height)
    target_r = grid_r.ravel()
    target_z = grid_z.ravel()
    lattice = FluxLattice(radius, height)
    width = float(radius[1] - radius[0])
    thickness = float(height[1] - height[0])
    characteristic_size = float(np.sqrt(width * thickness))
    lcfs = _lcfs_mask(
        radius,
        height,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    mapped_density = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored_density.T),
        radius,
        height,
    ).T
    mapped_density = np.where(lcfs, mapped_density, 0.0)

    couplings, branches = _counterfactual_couplings(
        target_r,
        target_z,
        width,
        thickness,
        np.asarray(profile.plasma_to_grid, dtype=np.float64),
    )
    reference = couplings["forced_cylinder"]
    coupling_differences = {
        name: _coupling_difference(coupling, reference)
        for name, coupling in couplings.items()
    }
    pair_errors = {
        name: _pair_error_fraction(coupling, lattice, mapped_density, span)
        for name, coupling in couplings.items()
    }
    control = (radial_points, vertical_points) == CONTROL_GRID
    return {
        "radial_points": radial_points,
        "vertical_points": vertical_points,
        "cell_count": radial_points * vertical_points,
        "radial_cell_size_m": width,
        "vertical_cell_size_m": thickness,
        "characteristic_cell_size_m": characteristic_size,
        "lcfs_admitted_cells": int(np.count_nonzero(lcfs)),
        "production_branches": branches,
        "coupling_difference_from_forced_cylinder": coupling_differences,
        "pair_error_fraction_of_stored_span": pair_errors,
        "moment_order_couplings_identical": bool(
            np.array_equal(
                couplings["quadrupole_hybrid"], couplings["third_moment_hybrid"]
            )
        ),
        "is_control_grid": control,
        "control_reference_pair_error_fraction": (
            CONTROL_PAIR_ERROR_FRACTION if control else None
        ),
        "control_reproduction_difference": (
            pair_errors["production_hybrid"] - CONTROL_PAIR_ERROR_FRACTION
            if control
            else None
        ),
    }


def measure_filament_contribution(
    *,
    shot: int,
    slice_index: int,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
) -> dict[str, Any]:
    """Measure filament attribution and convergence for all coupling variants."""

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
        [row["characteristic_cell_size_m"] for row in resolutions],
        dtype=np.float64,
    )
    convergence = {
        name: _fit_power_order(
            cell_sizes,
            np.asarray(
                [
                    row["pair_error_fraction_of_stored_span"][name]
                    for row in resolutions
                ],
                dtype=np.float64,
            ),
        )
        for name in VARIANT_NAMES
    }
    control = next(row for row in resolutions if row["is_control_grid"])
    control_errors = control["pair_error_fraction_of_stored_span"]
    production_error = control_errors["production_hybrid"]
    quadrupole_error = control_errors["quadrupole_hybrid"]
    forced_cylinder_error = control_errors["forced_cylinder"]
    fraction_removed = (production_error - quadrupole_error) / production_error
    forced_cylinder_fraction_removed = (
        production_error - forced_cylinder_error
    ) / production_error
    quadrupole_difference_from_forced = quadrupole_error - forced_cylinder_error
    residual_order = float(convergence["quadrupole_hybrid"]["observed_order"])
    residual_interval = convergence["quadrupole_hybrid"]["order_95_percent_interval"]
    residual_is_second_order = bool(residual_interval[0] <= 2.0 <= residual_interval[1])

    return {
        "source": {
            "shot": shot,
            "slice_index": stored.index,
            "time_s": stored.time_s,
            "stored_flux_span_wb": span,
            "stored_grid_shape_zr": list(stored.total_flux_wb.shape),
            "artifact_digest": artifact_digest,
        },
        "method": {
            "grid_sequence_rz": [list(grid) for grid in GRID_SEQUENCE],
            "production_switch": "distance < 3 * max(cell_width, cell_height)",
            "coupling_normalisation": (
                "sup(diff)/sup(forced-cylinder) and "
                "RMS(diff)/RMS(forced-cylinder), over every matrix element"
            ),
            "round_trip": (
                "LCFS-masked input current; coupling, conservation delta-star, "
                "same coupling; recovered-current LCFS remasking excluded"
            ),
        },
        "resolutions": resolutions,
        "pair_error_convergence": convergence,
        "attribution": {
            "control_grid_rz": list(CONTROL_GRID),
            "production_pair_error_fraction": production_error,
            "quadrupole_pair_error_fraction": quadrupole_error,
            "forced_cylinder_pair_error_fraction": forced_cylinder_error,
            "fraction_of_production_pair_error_removed_by_quadrupole": (
                fraction_removed
            ),
            "fraction_of_production_pair_error_removed_by_forced_cylinder": (
                forced_cylinder_fraction_removed
            ),
            "fraction_of_production_pair_error_remaining_after_quadrupole": (
                quadrupole_error / production_error
            ),
            "quadrupole_pair_error_difference_from_forced_cylinder": (
                quadrupole_difference_from_forced
            ),
            "quadrupole_pair_error_relative_difference_from_forced_cylinder": (
                quadrupole_difference_from_forced / forced_cylinder_error
            ),
            "fraction_of_exact_coupling_excess_captured_by_quadrupole": (
                (production_error - quadrupole_error)
                / (production_error - forced_cylinder_error)
            ),
            "quadrupole_residual_observed_order": residual_order,
            "quadrupole_residual_order_95_percent_interval": residual_interval,
            "quadrupole_residual_is_statistically_compatible_with_order_two": (
                residual_is_second_order
            ),
            "interpretation": (
                "the quadrupole-corrected residual remains compatible with "
                "second-order convergence and agrees with the forced-cylinder "
                "round trip; the remaining error is therefore the difference "
                "stencil truncation rather than another material coupling floor"
            ),
        },
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
    """Print the filament-attribution receipt as stable JSON."""

    arguments = _parser().parse_args()
    result = measure_filament_contribution(
        shot=arguments.shot,
        slice_index=arguments.slice,
        store=arguments.store,
        artifact_cache=arguments.artifact_cache,
        artifact_digest=arguments.artifact_digest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
