"""Verify that separatrix-clipped cell patches conserve poloidal area.

The first receipt copies the four-corner cell clip used by the current-rule
comparison.  The repaired receipt splits every horizontal cell edge at the
neighbouring row's T-junction, so both cells interpolate each shared atomic
edge from identical flux endpoints.  Patch areas and the piecewise-linear
boundary polygon are then independent shoelace sums over the same crossings.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy import integrate
from shapely.geometry import Polygon, box

from benchmarks.efit_constant_current_attribution import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    _fit_order,
    _hex_mesh,
)
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

CONSERVATION_RELATIVE_TOLERANCE = 1.0e-12
COORDINATE_ROUND_DIGITS = 14


def _signed_polygon_area(vertices: np.ndarray) -> float:
    """Return the signed shoelace area of an ordered polygon."""

    if len(vertices) < 3:
        return 0.0
    radius = vertices[:, 0]
    height = vertices[:, 1]
    return float(
        0.5 * np.sum(radius * np.roll(height, -1) - np.roll(radius, -1) * height)
    )


def _rectangle_vertices(
    radius: float,
    height: float,
    width: float,
    vertical_extent: float,
    *,
    split_horizontal_edges: bool,
) -> np.ndarray:
    """Return counter-clockwise cell vertices, optionally split at T-junctions."""

    left = radius - 0.5 * width
    right = radius + 0.5 * width
    lower = height - 0.5 * vertical_extent
    upper = height + 0.5 * vertical_extent
    if split_horizontal_edges:
        return np.asarray(
            [
                (left, lower),
                (radius, lower),
                (right, lower),
                (right, upper),
                (radius, upper),
                (left, upper),
            ]
        )
    return np.asarray([(left, lower), (right, lower), (right, upper), (left, upper)])


def _clip_polygon(
    case: RotatingEquilibrium, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clip one cell by linearly interpolating its zero-flux edge crossings."""

    flux = np.asarray(case.flux(vertices[:, 0], vertices[:, 1]), dtype=float)
    polygon: list[np.ndarray] = []
    crossings: list[np.ndarray] = []
    for index, start in enumerate(vertices):
        end_index = (index + 1) % len(vertices)
        end = vertices[end_index]
        start_flux = float(flux[index])
        end_flux = float(flux[end_index])
        if start_flux > 0.0:
            polygon.append(start)
        if (start_flux > 0.0) != (end_flux > 0.0):
            fraction = start_flux / (start_flux - end_flux)
            crossing = start + fraction * (end - start)
            polygon.append(crossing)
            crossings.append(crossing)
    polygon_array = np.asarray(polygon)
    if len(polygon_array) < 3:
        polygon_array = np.empty((0, 2))
    crossing_array = np.asarray(crossings)
    if crossing_array.size == 0:
        crossing_array = np.empty((0, 2))
    return polygon_array, crossing_array


def _ordered_unique_crossings(crossings: list[np.ndarray]) -> np.ndarray:
    """Return one angularly ordered copy of each shared contour crossing."""

    points = np.vstack([row for row in crossings if len(row)])
    rounded = np.round(points, COORDINATE_ROUND_DIGITS)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique = points[np.sort(unique_indices)]
    centre = np.mean(unique, axis=0)
    angle = np.arctan2(unique[:, 1] - centre[1], unique[:, 0] - centre[0])
    return unique[np.argsort(angle)]


def _horizontal_crossing_mismatch(
    case: RotatingEquilibrium,
    coordinates: np.ndarray,
    width: float,
    vertical_extent: float,
) -> dict[str, float | int]:
    """Measure disagreement where unsplit horizontal edges share an interface."""

    groups: dict[tuple[float, str], list[float]] = {}
    for radius, height in coordinates:
        vertices = _rectangle_vertices(
            float(radius),
            float(height),
            width,
            vertical_extent,
            split_horizontal_edges=False,
        )
        flux = np.asarray(case.flux(vertices[:, 0], vertices[:, 1]), dtype=float)
        for start_index in (0, 2):
            end_index = (start_index + 1) % len(vertices)
            start_flux = float(flux[start_index])
            end_flux = float(flux[end_index])
            if (start_flux > 0.0) == (end_flux > 0.0):
                continue
            fraction = start_flux / (start_flux - end_flux)
            crossing = vertices[start_index] + fraction * (
                vertices[end_index] - vertices[start_index]
            )
            side = "inboard" if crossing[0] < case.major_radius else "outboard"
            key = (round(float(crossing[1]), COORDINATE_ROUND_DIGITS), side)
            groups.setdefault(key, []).append(float(crossing[0]))
    spreads = [
        max(values) - min(values) for values in groups.values() if len(values) > 1
    ]
    return {
        "shared_interface_count": len(spreads),
        "maximum_radial_crossing_disagreement_m": float(max(spreads, default=0.0)),
        "rms_radial_crossing_disagreement_m": float(
            np.sqrt(np.mean(np.asarray(spreads) ** 2)) if spreads else 0.0
        ),
    }


def _analytic_plasma_area(case: RotatingEquilibrium) -> tuple[float, float]:
    """Return analytic LCFS area and its adaptive-quadrature error estimate."""

    inboard, outboard = case.boundary_midplane_radii()

    def full_height(radius: float) -> float:
        midplane_flux = max(float(case.flux(radius, 0.0)), 0.0)
        return 2.0 * np.sqrt(midplane_flux / case.field_coefficient)

    area, error = integrate.quad(
        full_height,
        inboard,
        outboard,
        epsabs=2.0e-14,
        epsrel=2.0e-13,
        limit=300,
        points=[case.major_radius],
    )
    return float(area), float(error)


def _patch_receipt(
    case: RotatingEquilibrium,
    coordinates: np.ndarray,
    width: float,
    vertical_extent: float,
    *,
    split_horizontal_edges: bool,
) -> dict[str, Any]:
    """Return patch and contour areas for one clipping construction."""

    cell_area = width * vertical_extent
    patch_areas = np.zeros(len(coordinates))
    crossings: list[np.ndarray] = []
    inside_count = 0
    boundary_count = 0
    outside_count = 0
    boundary_area = 0.0
    for index, (radius, height) in enumerate(coordinates):
        vertices = _rectangle_vertices(
            float(radius),
            float(height),
            width,
            vertical_extent,
            split_horizontal_edges=split_horizontal_edges,
        )
        polygon, cell_crossings = _clip_polygon(case, vertices)
        crossings.append(cell_crossings)
        area = abs(_signed_polygon_area(polygon))
        if len(cell_crossings) > 0:
            boundary_count += 1
            boundary_area += area
        elif bool(case.contains(float(radius), float(height))):
            inside_count += 1
            area = cell_area
        else:
            outside_count += 1
            area = 0.0
        patch_areas[index] = area

    contour = _ordered_unique_crossings(crossings)
    contour_signed_area = _signed_polygon_area(contour)
    if contour_signed_area < 0.0:
        contour = contour[::-1]
        contour_signed_area = -contour_signed_area
    patch_sum = float(np.sum(patch_areas))
    relative_residual = abs(patch_sum - contour_signed_area) / contour_signed_area

    contour_polygon = Polygon(contour)
    if not contour_polygon.is_valid:
        raise ValueError("ordered contour crossings do not form a valid polygon")
    cell_residuals = []
    for (radius, height), patch_area in zip(coordinates, patch_areas, strict=True):
        rectangle = box(
            radius - 0.5 * width,
            height - 0.5 * vertical_extent,
            radius + 0.5 * width,
            height + 0.5 * vertical_extent,
        )
        cell_residuals.append(
            abs(float(contour_polygon.intersection(rectangle).area) - patch_area)
        )
    return {
        "inside_cell_count": inside_count,
        "boundary_cell_count": boundary_count,
        "outside_cell_count": outside_count,
        "full_cell_area_m2": cell_area,
        "sum_full_inside_areas_m2": inside_count * cell_area,
        "sum_clipped_boundary_areas_m2": boundary_area,
        "sum_patch_areas_m2": patch_sum,
        "piecewise_linear_lcfs_area_m2": contour_signed_area,
        "absolute_conservation_residual_m2": abs(patch_sum - contour_signed_area),
        "relative_conservation_residual": relative_residual,
        "largest_single_cell_area_residual_m2": float(max(cell_residuals)),
        "piecewise_linear_lcfs_vertex_count": len(contour),
    }


def _measure_resolution(radial_count: int, vertical_count: int) -> dict[str, Any]:
    """Measure the copied and shared-edge clipping constructions on one grid."""

    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, width, vertical_extent = _hex_mesh(
        radial_count, vertical_count, case.major_radius, half_height
    )
    copied = _patch_receipt(
        case,
        mesh.coordinate,
        width,
        vertical_extent,
        split_horizontal_edges=False,
    )
    repaired = _patch_receipt(
        case,
        mesh.coordinate,
        width,
        vertical_extent,
        split_horizontal_edges=True,
    )
    if repaired["relative_conservation_residual"] > CONSERVATION_RELATIVE_TOLERANCE:
        raise AssertionError(
            "shared-edge patch areas do not conserve the piecewise-linear LCFS area: "
            f"{repaired['relative_conservation_residual']:.12e}"
        )
    analytic_area, analytic_error = _analytic_plasma_area(case)
    signed_geometry_error = (
        repaired["piecewise_linear_lcfs_area_m2"] - analytic_area
    ) / analytic_area
    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "copied_four_corner_clip": copied,
        "copied_clip_shared_edge_diagnostic": _horizontal_crossing_mismatch(
            case, mesh.coordinate, width, vertical_extent
        ),
        "repaired_shared_edge_clip": repaired,
        "analytic_lcfs_area_m2": analytic_area,
        "analytic_area_quadrature_error_estimate_m2": analytic_error,
        "piecewise_linear_vs_analytic_signed_relative_area_difference": (
            signed_geometry_error
        ),
        "conservation_assertion_passed": True,
    }


def measure_area_conservation() -> dict[str, Any]:
    """Return four-grid area conservation and geometric convergence receipts."""

    configure_dtypes()
    rows = [_measure_resolution(radial, vertical) for radial, vertical in GRID_SEQUENCE]
    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in rows])
    geometric_error = np.abs(
        np.asarray(
            [
                row["piecewise_linear_vs_analytic_signed_relative_area_difference"]
                for row in rows
            ]
        )
    )
    copied_fails = any(
        row["copied_four_corner_clip"]["relative_conservation_residual"]
        > CONSERVATION_RELATIVE_TOLERANCE
        for row in rows
    )
    return {
        "analytic_source": ANALYTIC_CASE,
        "grid_sequence": [list(grid) for grid in GRID_SEQUENCE],
        "conservation_relative_tolerance": CONSERVATION_RELATIVE_TOLERANCE,
        "first_measurement_failed": copied_fails,
        "named_defect": (
            "horizontal interfaces are T-junctions: adjacent unsplit cell edges "
            "interpolate the same contour crossing from different endpoint pairs"
            if copied_fails
            else "none"
        ),
        "repair": (
            "split every horizontal edge at the neighbouring row boundary and "
            "interpolate each shared atomic segment from identical endpoint fluxes"
        ),
        "resolutions": rows,
        "geometric_area_convergence": _fit_order(cell_size, geometric_error),
        "all_repaired_conservation_assertions_passed": all(
            row["conservation_assertion_passed"] for row in rows
        ),
    }


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main() -> None:
    """Print the conservation receipt as stable JSON."""

    _parser().parse_args()
    print(json.dumps(measure_area_conservation(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
