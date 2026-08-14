"""Separate plasma-cell section shape from polygon residual quadrature order.

The source equilibrium, half-offset hexagonal centroid meshes, plasma mask, and
cell currents are exactly those of :mod:`benchmarks.efit_analytic_roundtrip_floor`.
That benchmark represented each hexagonal control area by an axis-aligned
rectangle whose width is the centroid pitch and whose height is
``sqrt(3) / 2`` times the pitch.

This benchmark changes only the finite section inside the same near/far routing:
the rectangle route is Nova's production ``hybrid_greens`` and the comparison
route replaces each near response with the Urankar Part V closed-form response
of the true regular-hexagon Voronoi cell.  Both routes use the same cell current
and the same filament response outside the production standoff, so their flux
difference isolates the section-shape substitution.

The Part V reduction leaves two regularised ``arsinh`` residuals numerical.  A
separate sweep drives its public ``nodes`` argument without changing geometry or
routing.  The current Part V default is 128 nodes; 785 is also included because
older rectangle-reference documentation names that order, although it is not
the polygon kernel's default and an odd request is realised as 784 nodes by the
two equal panels.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from benchmarks.efit_analytic_roundtrip_floor import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    IDENTITY_FRACTION,
    _hex_mesh,
)
from nova.biot.greens import hybrid_greens, second_moments
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import reference_cases

POLYGON_DEFAULT_NODES = 128
QUADRATURE_NODE_SEQUENCE = (32, 64, 128, 256, 785, 1570)
PRODUCTION_STANDOFF = 3.0
COMPOSITION_ERROR_FRACTIONS = {
    (23, 35): 1.320394289e-2,
    (37, 57): 3.819717157e-3,
    (51, 79): 2.005611454e-3,
    (67, 103): 1.176524998e-3,
}


def _regular_hexagon(radius: float, height: float, pitch: float) -> np.ndarray:
    """Return the regular Voronoi hexagon of the centroid lattice."""

    radial_half = 0.5 * pitch
    vertical_half = 0.5 * pitch / np.sqrt(3.0)
    vertical_tip = pitch / np.sqrt(3.0)
    return np.array(
        [
            (radius + radial_half, height + vertical_half),
            (radius, height + vertical_tip),
            (radius - radial_half, height + vertical_half),
            (radius - radial_half, height - vertical_half),
            (radius, height - vertical_tip),
            (radius + radial_half, height - vertical_half),
        ],
        dtype=np.float64,
    )


def _rectangle(radius: float, height: float, width: float, depth: float) -> np.ndarray:
    """Return the rectangle used by the analytic round-trip coupling."""

    return np.array(
        [
            (radius - 0.5 * width, height - 0.5 * depth),
            (radius + 0.5 * width, height - 0.5 * depth),
            (radius + 0.5 * width, height + 0.5 * depth),
            (radius - 0.5 * width, height + 0.5 * depth),
        ],
        dtype=np.float64,
    )


def _fractional_difference(
    left: np.ndarray, right: np.ndarray, span: float
) -> dict[str, float]:
    """Return sup and RMS differences as fractions of an analytic flux span."""

    difference = np.asarray(left) - np.asarray(right)
    return {
        "sup_wb": float(np.max(np.abs(difference))),
        "rms_wb": float(np.sqrt(np.mean(difference**2))),
        "sup_fraction_of_analytic_span": float(np.max(np.abs(difference)) / span),
        "rms_fraction_of_analytic_span": float(np.sqrt(np.mean(difference**2)) / span),
    }


def _section_geometry(pitch: float) -> dict[str, Any]:
    """Quantify the equal-area polygon and rectangular section geometries."""

    depth = pitch * np.sqrt(3.0) / 2.0
    hexagon = _regular_hexagon(0.0, 0.0, pitch)
    rectangle = _rectangle(0.0, 0.0, pitch, depth)
    hex_moments = second_moments(hexagon)
    rectangle_moments = second_moments(rectangle)
    return {
        "pitch_m": pitch,
        "hexagon": {
            "area_m2": float(np.sqrt(3.0) * pitch**2 / 2.0),
            "radial_second_moment_m2": hex_moments[0],
            "vertical_second_moment_m2": hex_moments[1],
            "cross_second_moment_m2": hex_moments[2],
            "radial_second_moment_over_pitch_squared": hex_moments[0] / pitch**2,
            "vertical_second_moment_over_pitch_squared": hex_moments[1] / pitch**2,
            "cross_second_moment_over_pitch_squared": hex_moments[2] / pitch**2,
        },
        "rectangle": {
            "width_m": pitch,
            "height_m": depth,
            "area_m2": float(pitch * depth),
            "radial_second_moment_m2": rectangle_moments[0],
            "vertical_second_moment_m2": rectangle_moments[1],
            "cross_second_moment_m2": rectangle_moments[2],
            "radial_second_moment_over_pitch_squared": (
                rectangle_moments[0] / pitch**2
            ),
            "vertical_second_moment_over_pitch_squared": (
                rectangle_moments[1] / pitch**2
            ),
            "cross_second_moment_over_pitch_squared": (rectangle_moments[2] / pitch**2),
        },
    }


def _coupled_flux_by_order(
    coordinate: np.ndarray,
    current: np.ndarray,
    section_width: float,
    section_height: float,
    orders: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Apply rectangle and hexagon kernels with identical near/far routing."""

    target_r = coordinate[:, 0]
    target_z = coordinate[:, 1]
    rectangle_flux = np.zeros(len(coordinate), dtype=np.float64)
    polygon_flux = {
        order: np.zeros(len(coordinate), dtype=np.float64) for order in orders
    }
    evaluated_pairs = 0
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in np.flatnonzero(current != 0.0):
            source_r = float(target_r[source])
            source_z = float(target_z[source])
            rectangle_response = hybrid_greens(
                target_r,
                target_z,
                source_r,
                source_z,
                section_width,
                section_height,
                switch=PRODUCTION_STANDOFF,
            )[0]
            rectangle_flux += rectangle_response * current[source]
            near = np.hypot(
                target_r - source_r, target_z - source_z
            ) < PRODUCTION_STANDOFF * max(section_width, section_height)
            evaluated_pairs += int(np.count_nonzero(near))
            vertices = _regular_hexagon(source_r, source_z, section_width)
            for order in orders:
                response = rectangle_response.copy()
                response[near] = polygon_analytic_greens(
                    target_r[near], target_z[near], vertices, nodes=order
                )[0]
                polygon_flux[order] += response * current[source]
    return rectangle_flux, polygon_flux, evaluated_pairs


def _measure_resolution(
    radial_count: int, vertical_count: int, orders: tuple[int, ...]
) -> dict[str, Any]:
    """Measure section shape and residual-quadrature convergence on one mesh."""

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
    density = np.where(
        case.contains(radius, height),
        case.toroidal_current_density(radius, height),
        0.0,
    )
    current = np.asarray(density, dtype=np.float64) * mesh.cell_area
    analytic_span = float(TOTAL_FLUX_FACTOR * case.axis_flux)
    rectangle_flux, polygon_flux, evaluated_pairs = _coupled_flux_by_order(
        mesh.coordinate,
        current,
        section_width,
        section_height,
        orders,
    )

    default_flux = polygon_flux[POLYGON_DEFAULT_NODES]
    shape_effect = _fractional_difference(default_flux, rectangle_flux, analytic_span)
    composition_fraction = COMPOSITION_ERROR_FRACTIONS[(radial_count, vertical_count)]
    shape_effect["fraction_of_composition_sup_error"] = (
        shape_effect["sup_fraction_of_analytic_span"] / composition_fraction
    )

    consecutive = []
    for lower, upper in zip(orders, orders[1:], strict=True):
        consecutive.append(
            {
                "lower_requested_nodes": lower,
                "upper_requested_nodes": upper,
                "lower_realised_nodes": 2 * (lower // 2),
                "upper_realised_nodes": 2 * (upper // 2),
                **_fractional_difference(
                    polygon_flux[upper], polygon_flux[lower], analytic_span
                ),
            }
        )

    reference_order = orders[-1]
    polygon_default_residual = _fractional_difference(
        polygon_flux[reference_order], default_flux, analytic_span
    )
    documented_order = 785
    documented_order_residual = _fractional_difference(
        polygon_flux[reference_order], polygon_flux[documented_order], analytic_span
    )
    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "plasma_cell_count": int(np.count_nonzero(current)),
        "polygon_near_pairs_per_order": evaluated_pairs,
        "analytic_flux_span_wb": analytic_span,
        "composition_error_fraction_of_analytic_span": composition_fraction,
        "geometry": _section_geometry(section_width),
        "section_shape_flux_difference": shape_effect,
        "quadrature_consecutive_changes": consecutive,
        "polygon_default_residual_against_high_order": polygon_default_residual,
        "documented_785_residual_against_high_order": documented_order_residual,
        "polygon_default_converged_at_identity_level": (
            polygon_default_residual["sup_fraction_of_analytic_span"]
            <= IDENTITY_FRACTION
        ),
        "documented_785_converged_at_identity_level": (
            documented_order_residual["sup_fraction_of_analytic_span"]
            <= IDENTITY_FRACTION
        ),
        "polygon_default_residual_below_section_shape_term": (
            polygon_default_residual["sup_fraction_of_analytic_span"]
            < shape_effect["sup_fraction_of_analytic_span"]
        ),
        "documented_785_residual_below_section_shape_term": (
            documented_order_residual["sup_fraction_of_analytic_span"]
            < shape_effect["sup_fraction_of_analytic_span"]
        ),
    }


def measure_section_shape_and_order() -> dict[str, Any]:
    """Return the full section-shape and polygon-order evidence receipt."""

    configure_dtypes()
    orders = QUADRATURE_NODE_SEQUENCE
    resolutions = [
        _measure_resolution(radial_count, vertical_count, orders)
        for radial_count, vertical_count in GRID_SEQUENCE
    ]
    default_residual = max(
        row["polygon_default_residual_against_high_order"][
            "sup_fraction_of_analytic_span"
        ]
        for row in resolutions
    )
    documented_residual = max(
        row["documented_785_residual_against_high_order"][
            "sup_fraction_of_analytic_span"
        ]
        for row in resolutions
    )
    minimum_shape = min(
        row["section_shape_flux_difference"]["sup_fraction_of_analytic_span"]
        for row in resolutions
    )
    return {
        "analytic_source": {
            "name": reference_cases()[ANALYTIC_CASE].name,
            "stored_field_used": False,
            "interpolation_used": False,
            "grid_sequence": [list(shape) for shape in GRID_SEQUENCE],
        },
        "routing": {
            "rectangle": (
                "production hybrid_greens with width=pitch and height=sqrt(3)/2*pitch"
            ),
            "hexagon": (
                "same filament far field and standoff; each finite-section near "
                "response replaced by polygon_analytic_greens on the regular "
                "hexagonal Voronoi cell"
            ),
            "standoff_in_pitch": PRODUCTION_STANDOFF,
            "same_current_and_equal_area": True,
        },
        "quadrature": {
            "part_v_actual_default_requested_nodes": POLYGON_DEFAULT_NODES,
            "requested_node_sequence": list(orders),
            "realised_node_sequence": [2 * (order // 2) for order in orders],
            "reference_requested_nodes": orders[-1],
            "documentation_note": (
                "785 describes the separate rectangle-reference zeta rule in "
                "polygon.py; Part V currently defaults to 128, and its two equal "
                "panels realise an odd 785 request as 784 nodes"
            ),
            "maximum_default_residual_fraction_of_analytic_span": default_residual,
            "maximum_785_residual_fraction_of_analytic_span": documented_residual,
            "default_converged_at_identity_level": (
                default_residual <= IDENTITY_FRACTION
            ),
            "documented_785_converged_at_identity_level": (
                documented_residual <= IDENTITY_FRACTION
            ),
            "default_residual_below_every_section_shape_term": (
                default_residual < minimum_shape
            ),
            "documented_785_residual_below_every_section_shape_term": (
                documented_residual < minimum_shape
            ),
        },
        "resolutions": resolutions,
        "policy": (
            "measurement only: no production kernel or identity bound is changed"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> None:
    """Print a stable JSON evidence receipt."""

    _parser().parse_args()
    print(json.dumps(measure_section_shape_and_order(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
