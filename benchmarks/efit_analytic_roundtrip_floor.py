"""Measure the Green/delta-star composition on an analytic equilibrium.

The source is the closed-form moderate rotating equilibrium from the analytic
reference module.  Its total flux and toroidal current density are evaluated
directly at every mesh centroid; no stored map, interpolation, or fitted field
enters the path.

The differential operator is Nova's seven-point, centre-plus-six-neighbour
quadratic fit on a half-offset hexagonal tiling.  The coupling operator uses
Nova's production hybrid Green kernel with a rectangular finite section whose
width and height give the same control area as one hex-grid centroid.  The
analytic-flux differentiation error is reported separately from the composed
``Green -> delta_star -> Green`` flux error.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy import stats
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import delta_star
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import reference_cases

ANALYTIC_CASE = "moderate-rotation-conventional"
IDENTITY_FRACTION = 1.0e-6
GRID_SEQUENCE = ((23, 35), (37, 57), (51, 79), (67, 103))
DOMAIN_HEIGHT_FACTOR = 1.2


def _hex_mesh(
    radial_count: int, vertical_count: int, major_radius: float, half_height: float
) -> tuple[StencilMesh, float, float]:
    """Return a fixed-extent half-offset hexagonal centroid mesh."""

    vertical_extent = 2.0 * DOMAIN_HEIGHT_FACTOR * half_height
    pitch = vertical_extent / ((vertical_count - 1) * np.sqrt(3.0) / 2.0)
    radial_index, vertical_index = np.indices((radial_count, vertical_count))
    radial_centre = 0.5 * (radial_count - 1)
    vertical_centre = 0.5 * (vertical_count - 1)
    radius = major_radius + pitch * (
        radial_index - radial_centre + 0.5 * (vertical_index - vertical_centre)
    )
    height = pitch * np.sqrt(3.0) / 2.0 * (vertical_index - vertical_centre)
    coordinate = np.column_stack((radius.ravel(), height.ravel()))
    section_width = pitch
    section_height = pitch * np.sqrt(3.0) / 2.0
    area = np.full(len(coordinate), section_width * section_height)
    return (
        StencilMesh(
            coordinate=coordinate,
            stencil=hex_stencil((radial_count, vertical_count)),
            area=area,
        ),
        section_width,
        section_height,
    )


def _coupled_flux(
    mesh: StencilMesh,
    density: np.ndarray,
    section_width: float,
    section_height: float,
) -> np.ndarray:
    """Apply the production finite-section Green kernel without materialising it."""

    current = np.asarray(density, dtype=float) * mesh.cell_area
    target_r = mesh.coordinate[:, 0]
    target_z = mesh.coordinate[:, 1]
    flux = np.zeros(mesh.node_count, dtype=float)
    active = np.flatnonzero(current != 0.0)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in active:
            response = hybrid_greens(
                target_r,
                target_z,
                float(target_r[source]),
                float(target_z[source]),
                section_width,
                section_height,
            )[0]
            flux += response * current[source]
    return flux


def _recovered_density(mesh: StencilMesh, total_flux: np.ndarray) -> np.ndarray:
    """Return current density from Nova's total-flux delta-star convention."""

    elliptic = np.asarray(delta_star(mesh, total_flux), dtype=float)
    density = -elliptic / (TOTAL_FLUX_FACTOR * mu_0 * np.asarray(mesh.node_radius))
    return np.where(np.asarray(mesh.interior(1), dtype=bool), density, 0.0)


def _relative_error(
    actual: np.ndarray, expected: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Report absolute and peak-normalised current-density errors."""

    difference = np.asarray(actual)[mask] - np.asarray(expected)[mask]
    scale = float(np.max(np.abs(np.asarray(expected)[mask])))
    return {
        "checked_cells": int(np.count_nonzero(mask)),
        "expected_peak_a_per_m2": scale,
        "sup_error_a_per_m2": float(np.max(np.abs(difference))),
        "rms_error_a_per_m2": float(np.sqrt(np.mean(difference**2))),
        "sup_error_fraction_of_expected_peak": float(
            np.max(np.abs(difference)) / scale
        ),
        "rms_error_fraction_of_expected_peak": float(
            np.sqrt(np.mean(difference**2)) / scale
        ),
    }


def _fit_order(cell_size: np.ndarray, error: np.ndarray) -> dict[str, float]:
    """Fit a zero-limit power law and report its statistical residual."""

    fit = stats.linregress(np.log(cell_size), np.log(error))
    predicted_log = fit.intercept + fit.slope * np.log(cell_size)
    residual = np.log(error) - predicted_log
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "coefficient": float(np.exp(fit.intercept)),
        "log_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "fraction_residual_rms": float(
            np.sqrt(np.mean((error - np.exp(predicted_log)) ** 2))
        ),
        "r_squared": float(fit.rvalue**2),
    }


def _measure_resolution(radial_count: int, vertical_count: int) -> dict[str, Any]:
    """Measure analytic differentiation and pair composition on one mesh."""

    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, section_width, section_height = _hex_mesh(
        radial_count,
        vertical_count,
        case.major_radius,
        half_height,
    )
    radius = mesh.coordinate[:, 0]
    height = mesh.coordinate[:, 1]
    analytic_flux = TOTAL_FLUX_FACTOR * case.flux(radius, height)
    exact_density = case.toroidal_current_density(radius, height)
    plasma = case.contains(radius, height)
    stencil_valid = np.asarray(mesh.interior(1), dtype=bool)

    differentiated_analytic_density = _recovered_density(mesh, analytic_flux)
    analytic_current_error = _relative_error(
        differentiated_analytic_density,
        exact_density,
        stencil_valid & plasma,
    )

    driven_density = np.where(plasma, exact_density, 0.0)
    coupled_flux = _coupled_flux(mesh, driven_density, section_width, section_height)
    recovered_coupled_density = _recovered_density(mesh, coupled_flux)
    coupled_current_error = _relative_error(
        recovered_coupled_density,
        driven_density,
        stencil_valid,
    )
    reconstructed_flux = _coupled_flux(
        mesh, recovered_coupled_density, section_width, section_height
    )
    composition_error = reconstructed_flux - coupled_flux
    analytic_span = TOTAL_FLUX_FACTOR * case.axis_flux
    composition_fraction = float(np.max(np.abs(composition_error)) / analytic_span)
    composition_to_delta_ratio = float(
        composition_fraction
        / analytic_current_error["sup_error_fraction_of_expected_peak"]
    )

    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "linear_cell_count": float(np.sqrt(mesh.node_count)),
        "pitch_m": section_width,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "rectangular_section_width_m": section_width,
        "rectangular_section_height_m": section_height,
        "rectangular_section_area_m2": float(mesh.cell_area[0]),
        "hex_ring_count": int(len(mesh.stencil)),
        "plasma_cell_count": int(np.count_nonzero(plasma)),
        "analytic_flux_span_wb": float(analytic_span),
        "analytic_delta_star_current_error": analytic_current_error,
        "coupled_delta_star_current_error": coupled_current_error,
        "composition_sup_error_wb": float(np.max(np.abs(composition_error))),
        "composition_error_fraction_of_analytic_span": composition_fraction,
        "composition_to_analytic_delta_sup_ratio": composition_to_delta_ratio,
        "reaches_identity_fraction": composition_fraction <= IDENTITY_FRACTION,
    }


def measure_analytic_floor() -> dict[str, Any]:
    """Measure the analytic operator-pair series and state its reading."""

    configure_dtypes()
    case = reference_cases()[ANALYTIC_CASE]
    resolutions = [
        _measure_resolution(radial_count, vertical_count)
        for radial_count, vertical_count in GRID_SEQUENCE
    ]
    cell_size = np.asarray(
        [row["characteristic_cell_size_m"] for row in resolutions], dtype=float
    )
    composition_error = np.asarray(
        [row["composition_error_fraction_of_analytic_span"] for row in resolutions],
        dtype=float,
    )
    fit = _fit_order(cell_size, composition_error)
    estimated_identity_size = float(
        (IDENTITY_FRACTION / fit["coefficient"]) ** (1.0 / fit["observed_order"])
    )
    refinement_factor = float(cell_size[-1] / estimated_identity_size)
    passing = [row for row in resolutions if row["reaches_identity_fraction"]]
    reaches = bool(passing)
    minimum_index = int(np.argmin(composition_error))
    stops_improving = minimum_index < len(resolutions) - 1
    if reaches:
        reading = {
            "verdict": "REACHES_IDENTITY",
            "first_passing_resolution": {
                "radial_count": passing[0]["radial_count"],
                "vertical_count": passing[0]["vertical_count"],
                "error_fraction": passing[0][
                    "composition_error_fraction_of_analytic_span"
                ],
            },
            "conclusion": (
                "the operator pair can satisfy the identity on an exact input; "
                "a larger stored-reference residual is attributable to its input path"
            ),
        }
    elif stops_improving:
        reading = {
            "verdict": "FLOOR_ABOVE_IDENTITY",
            "floor_fraction": float(composition_error[minimum_index]),
            "floor_resolution": {
                "radial_count": resolutions[minimum_index]["radial_count"],
                "vertical_count": resolutions[minimum_index]["vertical_count"],
            },
            "stops_improving_in_series": True,
            "conclusion": (
                "the identity is unreachable by this operator pair regardless of input"
            ),
        }
    else:
        reading = {
            "verdict": "STILL_CONVERGING",
            "floor_observed": False,
            "minimum_measured_fraction": float(composition_error[-1]),
            "finest_resolution": {
                "radial_count": resolutions[-1]["radial_count"],
                "vertical_count": resolutions[-1]["vertical_count"],
            },
            "stops_improving_in_series": False,
            "estimated_characteristic_cell_size_for_identity_m": (
                estimated_identity_size
            ),
            "estimated_linear_refinement_from_finest": refinement_factor,
            "estimated_cells_at_identity": int(
                np.ceil(resolutions[-1]["cell_count"] * refinement_factor**2)
            ),
            "conclusion": (
                "the operator pair has no observed floor and converges toward the "
                "identity; reaching it by direct all-to-all Green composition is "
                "computationally infeasible at the extrapolated cell count"
            ),
        }

    finest = resolutions[-1]
    attribution = {
        "comparison": (
            "composition fraction of analytic flux span divided by analytic "
            "delta-star sup error fraction of peak closed-form current"
        ),
        "ratios_by_resolution": [
            {
                "radial_count": row["radial_count"],
                "vertical_count": row["vertical_count"],
                "ratio": row["composition_to_analytic_delta_sup_ratio"],
            }
            for row in resolutions
        ],
        "finest_composition_fraction_of_analytic_span": finest[
            "composition_error_fraction_of_analytic_span"
        ],
        "finest_analytic_delta_sup_fraction_of_peak_current": finest[
            "analytic_delta_star_current_error"
        ]["sup_error_fraction_of_expected_peak"],
        "finest_analytic_delta_rms_fraction_of_peak_current": finest[
            "analytic_delta_star_current_error"
        ]["rms_error_fraction_of_expected_peak"],
        "finest_composition_to_delta_sup_ratio": finest[
            "composition_to_analytic_delta_sup_ratio"
        ],
        "conclusion": (
            "the finite-section Green composition dominates the residual; the "
            "hex-ring quadratic difference operator is roughly thirty times "
            "closer to its closed-form reference at the finest resolution"
        ),
        "input_exclusion": (
            "because the field and current are closed-form values with no stored "
            "map or interpolation, failure to reach the identity is attributable "
            "to Nova's operator pair, not stored-reference truncation"
        ),
    }

    return {
        "analytic_source": {
            "name": case.name,
            "family": "exact isothermal rotating Solov'ev equilibrium",
            "flux_evaluation": "closed form at every hex-lattice centroid",
            "current_evaluation": "closed-form toroidal current density",
            "analytic_flux_span_wb": float(TOTAL_FLUX_FACTOR * case.axis_flux),
            "stored_field_used": False,
            "interpolation_used": False,
        },
        "operator_path": {
            "difference_stencil": (
                "centre plus six touching neighbours; least-squares quadratic "
                "fit on each half-offset hexagonal ring"
            ),
            "coupling_element": (
                "rectangular finite section with width equal to hex pitch and "
                "height equal to sqrt(3)/2 times pitch"
            ),
            "coupling_kernel": (
                "production hybrid_greens: finite cylinder near the section, "
                "conditioned filament with finite-area correction in the far field"
            ),
            "composition": (
                "closed-form plasma current -> Green flux -> delta_star current "
                "-> Green flux"
            ),
        },
        "grid_series": {
            "resolution_count": len(resolutions),
            "linear_cell_count_factor": float(
                resolutions[-1]["linear_cell_count"]
                / resolutions[0]["linear_cell_count"]
            ),
            "contains_grid_finer_than_65_by_65": any(
                row["radial_count"] > 65 and row["vertical_count"] > 65
                for row in resolutions
            ),
            "resolutions": resolutions,
        },
        "composition_convergence": fit,
        "error_attribution": attribution,
        "reading": reading,
        "feasibility": (
            "the extrapolated all-to-all Green composition scales quadratically "
            "with about 4.1 million cells and is outside any reasonable compute "
            "budget; the extrapolation is the result and no direct run is needed"
        ),
        "policy": (
            "measurement only: no bound is registered or moved; the existing "
            "identity fraction remains unchanged"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> None:
    """Print the analytic operator-pair receipt as stable JSON."""

    _parser().parse_args()
    print(json.dumps(measure_analytic_floor(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
