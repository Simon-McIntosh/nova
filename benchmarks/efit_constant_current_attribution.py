"""Attribute analytic coupling error to cell-current quadrature.

Two plasma-flux assemblies use the same rectangular-section
``cylinder_greens`` response for every source-target pair.  The first gives
each source section its centroid current density times its area.  The second
integrates the closed-form current density over the part of that section
inside the analytic plasma.  Their difference therefore changes source
quadrature without changing the kernel, grid, equilibrium, or target points.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from typing import Any

import numpy as np
from scipy import integrate, stats
from scipy.constants import mu_0

from nova.biot.greens import cylinder_greens
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

ANALYTIC_CASE = "moderate-rotation-conventional"
GRID_SEQUENCE = ((23, 35), (37, 57), (51, 79), (67, 103))
REFERENCE_COMPOSITION_FRACTIONS = (
    1.320394289e-2,
    3.819717157e-3,
    2.005611454e-3,
    1.176524998e-3,
)
DOMAIN_HEIGHT_FACTOR = 1.2
CURRENT_QUADRATURE_ABSOLUTE_TOLERANCE_A = 1.0e-7
CURRENT_QUADRATURE_RELATIVE_TOLERANCE = 2.0e-12


def _hex_mesh(
    radial_count: int, vertical_count: int, major_radius: float, half_height: float
) -> tuple[StencilMesh, float, float]:
    """Return the fixed-extent mesh used by the analytic round trip."""

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


def _current_antiderivative(case: RotatingEquilibrium, radius: float) -> float:
    """Integrate the closed-form toroidal current density with respect to R."""

    theta = case.rotation_parameter
    if theta == 0.0:
        pressure_term = 2.0 * case.pressure_coefficient * radius**2 / mu_0
    else:
        label = radius**2 - case.major_radius**2
        pressure_term = (
            2.0 * case.pressure_coefficient * np.exp(theta * label) / (mu_0 * theta)
        )
    field_term = 2.0 * case.field_coefficient * np.log(radius) / mu_0
    return float(pressure_term + field_term)


def _plasma_half_height(case: RotatingEquilibrium, radius: float) -> float:
    """Return the analytic plasma half-height at one major radius."""

    midplane_flux = float(case.flux(radius, 0.0))
    if midplane_flux <= 0.0:
        return 0.0
    return float(np.sqrt(midplane_flux / case.field_coefficient))


def _vertical_overlap(
    case: RotatingEquilibrium, radius: float, lower: float, upper: float
) -> float:
    """Return the section height inside the closed-form plasma boundary."""

    half_height = _plasma_half_height(case, radius)
    return max(0.0, min(upper, half_height) - max(lower, -half_height))


def _integrated_cell_current(
    case: RotatingEquilibrium,
    radius: float,
    height: float,
    width: float,
    vertical_extent: float,
) -> tuple[float, float, str]:
    """Return plasma current in one rectangle and its quadrature error estimate."""

    radial_lower = radius - 0.5 * width
    radial_upper = radius + 0.5 * width
    vertical_lower = height - 0.5 * vertical_extent
    vertical_upper = height + 0.5 * vertical_extent
    radial_candidates = [radial_lower, radial_upper]
    integration_points: list[float] = []
    if radial_lower < case.major_radius < radial_upper:
        radial_candidates.append(case.major_radius)
        integration_points.append(case.major_radius)
    half_heights = [
        _plasma_half_height(case, candidate) for candidate in radial_candidates
    ]
    minimum_required_half_height = max(0.0, vertical_lower, -vertical_upper)
    full_section_half_height = max(vertical_upper, -vertical_lower)
    if max(half_heights) <= minimum_required_half_height:
        return 0.0, 0.0, "outside"
    if min(half_heights) >= full_section_half_height:
        current = vertical_extent * (
            _current_antiderivative(case, radial_upper)
            - _current_antiderivative(case, radial_lower)
        )
        return current, 0.0, "inside"

    def integrand(source_radius: float) -> float:
        overlap = _vertical_overlap(case, source_radius, vertical_lower, vertical_upper)
        if overlap == 0.0:
            return 0.0
        density = float(case.toroidal_current_density(source_radius, 0.0))
        return density * overlap

    current, absolute_error = integrate.quad(
        integrand,
        radial_lower,
        radial_upper,
        epsabs=CURRENT_QUADRATURE_ABSOLUTE_TOLERANCE_A,
        epsrel=CURRENT_QUADRATURE_RELATIVE_TOLERANCE,
        limit=100,
        points=integration_points or None,
    )
    return float(current), float(absolute_error), "boundary"


def _cell_currents(
    case: RotatingEquilibrium,
    mesh: StencilMesh,
    section_width: float,
    section_height: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return centroid and integrated currents with a quadrature receipt."""

    radius = mesh.coordinate[:, 0]
    height = mesh.coordinate[:, 1]
    density = case.toroidal_current_density(radius, height)
    centroid_inside = case.contains(radius, height)
    centroid_current = np.where(centroid_inside, density * mesh.cell_area, 0.0)
    integrated_rows = [
        _integrated_cell_current(
            case,
            float(source_radius),
            float(source_height),
            section_width,
            section_height,
        )
        for source_radius, source_height in mesh.coordinate
    ]
    integrated_current = np.asarray([row[0] for row in integrated_rows])
    error_estimate = np.asarray([row[1] for row in integrated_rows])
    classes = np.asarray([row[2] for row in integrated_rows])
    return (
        centroid_current,
        integrated_current,
        {
            "closed_form_full_cell_count": int(np.count_nonzero(classes == "inside")),
            "adaptive_boundary_cell_count": int(
                np.count_nonzero(classes == "boundary")
            ),
            "zero_outside_cell_count": int(np.count_nonzero(classes == "outside")),
            "maximum_boundary_quadrature_error_estimate_a": float(
                np.max(error_estimate)
            ),
            "sum_boundary_quadrature_error_estimates_a": float(np.sum(error_estimate)),
        },
    )


def _flux_chunk(
    sources: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
    centroid_current: np.ndarray,
    integrated_current: np.ndarray,
    section_width: float,
    section_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble both source representations with one identical kernel pass."""

    centroid_flux = np.zeros_like(target_r)
    integrated_flux = np.zeros_like(target_r)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in sources:
            response = cylinder_greens(
                target_r,
                target_z,
                float(target_r[source]),
                float(target_z[source]),
                section_width,
                section_height,
            )[0]
            centroid_flux += response * centroid_current[source]
            integrated_flux += response * integrated_current[source]
    return centroid_flux, integrated_flux


def _assemble_fluxes(
    mesh: StencilMesh,
    centroid_current: np.ndarray,
    integrated_current: np.ndarray,
    section_width: float,
    section_height: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assemble both exact-kernel flux maps, optionally by source chunks."""

    target_r = np.asarray(mesh.coordinate[:, 0])
    target_z = np.asarray(mesh.coordinate[:, 1])
    active = np.flatnonzero((centroid_current != 0.0) | (integrated_current != 0.0))
    chunks = [chunk for chunk in np.array_split(active, workers) if len(chunk)]
    arguments = (
        target_r,
        target_z,
        centroid_current,
        integrated_current,
        section_width,
        section_height,
    )
    if len(chunks) == 1:
        rows = [_flux_chunk(chunks[0], *arguments)]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_flux_chunk, chunk, *arguments) for chunk in chunks
            ]
            rows = [future.result() for future in futures]
    centroid_flux = np.sum([row[0] for row in rows], axis=0)
    integrated_flux = np.sum([row[1] for row in rows], axis=0)
    if not np.all(np.isfinite(centroid_flux)) or not np.all(
        np.isfinite(integrated_flux)
    ):
        raise FloatingPointError("exact-kernel assembly produced a non-finite flux")
    return centroid_flux, integrated_flux, len(active)


def _current_difference(
    centroid_current: np.ndarray, integrated_current: np.ndarray
) -> dict[str, Any]:
    """Report source-quadrature error without conflating it with flux coupling."""

    active = (centroid_current != 0.0) | (integrated_current != 0.0)
    exact_nonzero = integrated_current != 0.0
    difference = centroid_current[active] - integrated_current[active]
    exact = integrated_current[active]
    per_cell_relative = np.abs(
        (centroid_current[exact_nonzero] - integrated_current[exact_nonzero])
        / integrated_current[exact_nonzero]
    )
    return {
        "active_cell_count": int(np.count_nonzero(active)),
        "centroid_nonzero_cell_count": int(np.count_nonzero(centroid_current)),
        "integrated_nonzero_cell_count": int(np.count_nonzero(integrated_current)),
        "sup_difference_a": float(np.max(np.abs(difference))),
        "rms_difference_a": float(np.sqrt(np.mean(difference**2))),
        "sup_difference_fraction_of_peak_integrated_cell_current": float(
            np.max(np.abs(difference)) / np.max(np.abs(exact))
        ),
        "rms_difference_fraction_of_rms_integrated_cell_current": float(
            np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(exact**2))
        ),
        "absolute_difference_fraction_of_absolute_integrated_current": float(
            np.sum(np.abs(difference)) / np.sum(np.abs(exact))
        ),
        "signed_total_difference_fraction_of_integrated_current": float(
            np.sum(difference) / np.sum(exact)
        ),
        "per_nonzero_integrated_cell_relative_difference": {
            "sup": float(np.max(per_cell_relative)),
            "rms": float(np.sqrt(np.mean(per_cell_relative**2))),
            "median": float(np.median(per_cell_relative)),
        },
    }


def _fit_order(cell_size: np.ndarray, error: np.ndarray) -> dict[str, float]:
    """Fit a zero-limit power law to one error series."""

    fit = stats.linregress(np.log(cell_size), np.log(error))
    predicted_log = fit.intercept + fit.slope * np.log(cell_size)
    residual = np.log(error) - predicted_log
    return {
        "observed_order": float(fit.slope),
        "order_standard_error": float(fit.stderr),
        "coefficient": float(np.exp(fit.intercept)),
        "log_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "r_squared": float(fit.rvalue**2),
    }


def _measure_resolution(
    radial_count: int,
    vertical_count: int,
    reference_composition_fraction: float,
    workers: int,
) -> dict[str, Any]:
    """Measure one same-grid constant-current attribution row."""

    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, section_width, section_height = _hex_mesh(
        radial_count, vertical_count, case.major_radius, half_height
    )
    centroid_current, integrated_current, quadrature = _cell_currents(
        case, mesh, section_width, section_height
    )
    centroid_flux, integrated_flux, active_source_count = _assemble_fluxes(
        mesh,
        centroid_current,
        integrated_current,
        section_width,
        section_height,
        workers,
    )
    difference = centroid_flux - integrated_flux
    analytic_span = float(TOTAL_FLUX_FACTOR * case.axis_flux)
    sup_fraction = float(np.max(np.abs(difference)) / analytic_span)
    rms_fraction = float(np.sqrt(np.mean(difference**2)) / analytic_span)
    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "analytic_flux_span_wb": analytic_span,
        "active_source_count": active_source_count,
        "centroid_vs_integrated_flux": {
            "sup_difference_wb": float(np.max(np.abs(difference))),
            "rms_difference_wb": float(np.sqrt(np.mean(difference**2))),
            "sup_difference_fraction_of_analytic_span": sup_fraction,
            "rms_difference_fraction_of_analytic_span": rms_fraction,
        },
        "reference_composition_error_fraction_of_analytic_span": (
            reference_composition_fraction
        ),
        "constant_current_fraction_of_reference_composition_error": float(
            sup_fraction / reference_composition_fraction
        ),
        "centroid_vs_integrated_cell_current": _current_difference(
            centroid_current, integrated_current
        ),
        "integrated_current_quadrature": quadrature,
    }


def measure_constant_current_attribution(workers: int) -> dict[str, Any]:
    """Measure the four-grid source-quadrature attribution series."""

    configure_dtypes()
    rows = [
        _measure_resolution(radial_count, vertical_count, composition, workers)
        for (radial_count, vertical_count), composition in zip(
            GRID_SEQUENCE, REFERENCE_COMPOSITION_FRACTIONS, strict=True
        )
    ]
    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in rows])
    sup_error = np.asarray(
        [
            row["centroid_vs_integrated_flux"][
                "sup_difference_fraction_of_analytic_span"
            ]
            for row in rows
        ]
    )
    rms_error = np.asarray(
        [
            row["centroid_vs_integrated_flux"][
                "rms_difference_fraction_of_analytic_span"
            ]
            for row in rows
        ]
    )
    accounted = np.asarray(
        [
            row["constant_current_fraction_of_reference_composition_error"]
            for row in rows
        ]
    )
    median_accounted = float(np.median(accounted))
    if median_accounted >= 1.0:
        conclusion = (
            "the isolated cell-current quadrature effect exceeds the reference "
            "composition residual, so the round trip contains cancellation rather "
            "than additive positive error; changing the kernel alone cannot remove "
            "this source term"
        )
    elif median_accounted >= 0.5:
        conclusion = (
            "cell-current quadrature accounts for most of the reference composition "
            "error, so changing the kernel alone cannot remove its dominant term"
        )
    elif median_accounted >= 0.1:
        conclusion = (
            "cell-current quadrature is a material but not majority share of the "
            "reference composition error; kernel-only work leaves this term intact"
        )
    else:
        conclusion = (
            "cell-current quadrature is a minor share of the reference composition "
            "error, leaving the kernel or another coupling detail as the larger lever"
        )
    return {
        "analytic_source": {
            "name": ANALYTIC_CASE,
            "flux_and_current": "closed form at every evaluation point",
            "stored_field_used": False,
            "interpolation_used": False,
        },
        "held_fixed": {
            "grid_sequence": [list(grid) for grid in GRID_SEQUENCE],
            "kernel": "cylinder_greens for every source-target pair",
            "filament_branch_used": False,
            "section_shape": "rectangle matching the analytic round-trip cell area",
        },
        "source_representations": {
            "centroid": (
                "closed-form centroid density times full cell area when the "
                "centroid is inside"
            ),
            "integrated": (
                "closed-form density integrated over the cell intersection "
                "with the analytic plasma"
            ),
        },
        "worker_processes": workers,
        "resolutions": rows,
        "flux_difference_convergence": {
            "sup_fraction_of_analytic_span": _fit_order(cell_size, sup_error),
            "rms_fraction_of_analytic_span": _fit_order(cell_size, rms_error),
        },
        "attribution": {
            "minimum_fraction_of_reference_composition_error": float(np.min(accounted)),
            "maximum_fraction_of_reference_composition_error": float(np.max(accounted)),
            "median_fraction_of_reference_composition_error": median_accounted,
            "conclusion": conclusion,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="source chunks evaluated concurrently",
    )
    return parser


def main() -> None:
    """Print the source-quadrature attribution as stable JSON."""

    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    print(
        json.dumps(
            measure_constant_current_attribution(args.workers),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
