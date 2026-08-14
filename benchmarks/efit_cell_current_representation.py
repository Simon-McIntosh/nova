"""Rank cell-current quadratures against an integrated analytic reference.

Every representation changes only the source-current vector.  One forced
``cylinder_greens`` pass supplies the common rectangular-section coupling for
all variants, including separately accumulated interior- and boundary-cell
errors.  The zero-flux contour is the analytic equivalent of ``psi_N = 1``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks.efit_constant_current_attribution import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    _fit_order,
    _hex_mesh,
    _integrated_cell_current,
)
from nova.biot.greens import cylinder_greens
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

ANALYTIC_SPAN_WB = 1.3812146953272917
BANKED_CENTROID_SUP_FRACTIONS = (
    5.850582681e-2,
    2.105271145e-2,
    1.199637781e-2,
    9.102053091e-3,
)
GAUSS_ORDERS = (2, 4)
POLYGON_QUADRATURE_ORDER = 4
VARIANT_NAMES = ("centroid", "corners", "gauss_2", "gauss_4", "clipped")


def _masked_density(
    case: RotatingEquilibrium, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    """Return analytic current density inside the plasma and zero outside."""

    return np.where(
        case.contains(radius, height),
        case.toroidal_current_density(radius, height),
        0.0,
    )


def _tensor_current(
    case: RotatingEquilibrium,
    radius: np.ndarray,
    height: np.ndarray,
    width: float,
    vertical_extent: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Integrate masked point samples over every full rectangle."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    current = np.zeros_like(radius)
    radial_moment = np.zeros_like(radius)
    vertical_moment = np.zeros_like(radius)
    for radial_node, radial_weight in zip(nodes, weights, strict=True):
        source_r = radius + 0.5 * width * radial_node
        for vertical_node, vertical_weight in zip(nodes, weights, strict=True):
            source_z = height + 0.5 * vertical_extent * vertical_node
            weighted_density = (
                radial_weight
                * vertical_weight
                * _masked_density(case, source_r, source_z)
            )
            current += weighted_density
            radial_moment += weighted_density * source_r
            vertical_moment += weighted_density * source_z
    scale = 0.25 * width * vertical_extent
    current *= scale
    radial_moment *= scale
    vertical_moment *= scale
    centroid = np.column_stack((radius, height))
    nonzero = current != 0.0
    centroid[nonzero, 0] = radial_moment[nonzero] / current[nonzero]
    centroid[nonzero, 1] = vertical_moment[nonzero] / current[nonzero]
    evaluations = order**2 * len(radius)
    return current, centroid, evaluations, evaluations


def _corner_current(
    case: RotatingEquilibrium,
    radius: np.ndarray,
    height: np.ndarray,
    width: float,
    vertical_extent: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return the equal-weight four-corner current rule."""

    density = np.zeros_like(radius)
    radial_moment = np.zeros_like(radius)
    vertical_moment = np.zeros_like(radius)
    for radial_sign, vertical_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        source_r = radius + 0.5 * radial_sign * width
        source_z = height + 0.5 * vertical_sign * vertical_extent
        sample = _masked_density(
            case,
            source_r,
            source_z,
        )
        density += sample
        radial_moment += sample * source_r
        vertical_moment += sample * source_z
    scale = 0.25 * width * vertical_extent
    current = density * scale
    centroid = np.column_stack((radius, height))
    nonzero = current != 0.0
    centroid[nonzero, 0] = radial_moment[nonzero] / density[nonzero]
    centroid[nonzero, 1] = vertical_moment[nonzero] / density[nonzero]
    evaluations = 4 * len(radius)
    return current, centroid, evaluations, evaluations


def _clip_rectangle(
    case: RotatingEquilibrium,
    radius: float,
    height: float,
    width: float,
    vertical_extent: float,
) -> tuple[np.ndarray, int]:
    """Clip one rectangle by linearly interpolating flux on its edges."""

    vertices = np.asarray(
        [
            (radius - 0.5 * width, height - 0.5 * vertical_extent),
            (radius + 0.5 * width, height - 0.5 * vertical_extent),
            (radius + 0.5 * width, height + 0.5 * vertical_extent),
            (radius - 0.5 * width, height + 0.5 * vertical_extent),
        ]
    )
    flux = np.asarray(case.flux(vertices[:, 0], vertices[:, 1]))
    polygon: list[np.ndarray] = []
    for index, start in enumerate(vertices):
        end_index = (index + 1) % len(vertices)
        end = vertices[end_index]
        start_flux = float(flux[index])
        end_flux = float(flux[end_index])
        if start_flux > 0.0:
            polygon.append(start)
        if (start_flux > 0.0) != (end_flux > 0.0):
            fraction = start_flux / (start_flux - end_flux)
            polygon.append(start + fraction * (end - start))
    if len(polygon) < 3:
        return np.empty((0, 2)), 4
    return np.asarray(polygon), 4


def _triangle_current_moment(
    case: RotatingEquilibrium, triangle: np.ndarray
) -> tuple[float, np.ndarray, int]:
    """Integrate current and its first spatial moment over one triangle."""

    nodes, weights = np.polynomial.legendre.leggauss(POLYGON_QUADRATURE_ORDER)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    first, second, third = triangle
    first_edge = second - first
    second_edge = third - first
    double_area = abs(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
    current = 0.0
    moment = np.zeros(2)
    evaluations = 0
    for radial_node, radial_weight in zip(unit_nodes, unit_weights, strict=True):
        for edge_node, edge_weight in zip(unit_nodes, unit_weights, strict=True):
            point = first + radial_node * (
                (1.0 - edge_node) * (second - first) + edge_node * (third - first)
            )
            density = float(case.toroidal_current_density(point[0], point[1]))
            weighted = density * double_area * radial_node * radial_weight * edge_weight
            current += weighted
            moment += weighted * point
            evaluations += 1
    return current, moment, evaluations


def _clipped_currents(
    case: RotatingEquilibrium,
    radius: np.ndarray,
    height: np.ndarray,
    width: float,
    vertical_extent: float,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, int]]:
    """Integrate current over linearly clipped cell polygons."""

    current = np.zeros_like(radius)
    centroid = np.column_stack((radius, height))
    current_evaluations = 0
    flux_evaluations = 0
    evaluations_by_class = {name: 0 for name in ("inside", "boundary", "outside")}
    for index, (cell_r, cell_z) in enumerate(zip(radius, height, strict=True)):
        polygon, cell_flux_evaluations = _clip_rectangle(
            case, float(cell_r), float(cell_z), width, vertical_extent
        )
        flux_evaluations += cell_flux_evaluations
        if len(polygon) < 3:
            continue
        moment = np.zeros(2)
        cell_evaluations = 0
        for triangle_index in range(1, len(polygon) - 1):
            triangle = polygon[[0, triangle_index, triangle_index + 1]]
            triangle_current, triangle_moment, evaluations = _triangle_current_moment(
                case, triangle
            )
            current[index] += triangle_current
            moment += triangle_moment
            current_evaluations += evaluations
            cell_evaluations += evaluations
        evaluations_by_class[str(classes[index])] += cell_evaluations
        if current[index] != 0.0:
            centroid[index] = moment / current[index]
    return (
        current,
        centroid,
        current_evaluations,
        flux_evaluations,
        evaluations_by_class,
    )


def _exact_currents_and_classes(
    case: RotatingEquilibrium,
    mesh: StencilMesh,
    width: float,
    vertical_extent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact cell totals and geometric cell classes."""

    rows = [
        _integrated_cell_current(
            case, float(radius), float(height), width, vertical_extent
        )
        for radius, height in mesh.coordinate
    ]
    return (
        np.asarray([row[0] for row in rows]),
        np.asarray([row[2] for row in rows]),
    )


def _flux_chunk(
    sources: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
    current_vectors: np.ndarray,
    width: float,
    vertical_extent: float,
) -> np.ndarray:
    """Couple every current vector through one exact response per source."""

    flux = np.zeros((len(target_r), current_vectors.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in sources:
            response = cylinder_greens(
                target_r,
                target_z,
                float(target_r[source]),
                float(target_z[source]),
                width,
                vertical_extent,
            )[0]
            flux += response[:, None] * current_vectors[source]
    return flux


def _couple_vectors(
    mesh: StencilMesh,
    current_vectors: np.ndarray,
    width: float,
    vertical_extent: float,
    workers: int,
) -> np.ndarray:
    """Apply the shared exact coupling to all current representations."""

    target_r = np.asarray(mesh.coordinate[:, 0])
    target_z = np.asarray(mesh.coordinate[:, 1])
    active = np.flatnonzero(np.any(current_vectors != 0.0, axis=1))
    chunks = [chunk for chunk in np.array_split(active, workers) if len(chunk)]
    arguments = (target_r, target_z, current_vectors, width, vertical_extent)
    if len(chunks) == 1:
        rows = [_flux_chunk(chunks[0], *arguments)]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_flux_chunk, chunk, *arguments) for chunk in chunks
            ]
            rows = [future.result() for future in futures]
    flux = np.sum(rows, axis=0)
    if not np.all(np.isfinite(flux)):
        raise FloatingPointError("exact-kernel coupling produced non-finite flux")
    return flux


def _flux_error(field: np.ndarray) -> dict[str, float]:
    """Return sup and RMS error fractions of the analytic span."""

    return {
        "sup_fraction_of_analytic_span": float(
            np.max(np.abs(field)) / ANALYTIC_SPAN_WB
        ),
        "rms_fraction_of_analytic_span": float(
            np.sqrt(np.mean(field**2)) / ANALYTIC_SPAN_WB
        ),
    }


def _variant_currents(
    case: RotatingEquilibrium,
    mesh: StencilMesh,
    width: float,
    vertical_extent: float,
    classes: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, float | int]],
    dict[str, np.ndarray],
]:
    """Build each candidate current vector and its setup-cost receipt."""

    radius = np.asarray(mesh.coordinate[:, 0])
    height = np.asarray(mesh.coordinate[:, 1])
    area = width * vertical_extent
    currents: dict[str, np.ndarray] = {}
    centroids: dict[str, np.ndarray] = {"centroid": np.column_stack((radius, height))}
    cost: dict[str, dict[str, float | int]] = {}

    started = perf_counter()
    currents["centroid"] = _masked_density(case, radius, height) * area
    cost["centroid"] = {
        "generation_seconds": perf_counter() - started,
        "current_density_evaluations": len(radius),
        "normalized_flux_evaluations": len(radius),
    }

    started = perf_counter()
    (
        currents["corners"],
        centroids["corners"],
        current_count,
        flux_count,
    ) = _corner_current(case, radius, height, width, vertical_extent)
    cost["corners"] = {
        "generation_seconds": perf_counter() - started,
        "current_density_evaluations": current_count,
        "normalized_flux_evaluations": flux_count,
    }

    for order in GAUSS_ORDERS:
        name = f"gauss_{order}"
        started = perf_counter()
        (
            currents[name],
            centroids[name],
            current_count,
            flux_count,
        ) = _tensor_current(case, radius, height, width, vertical_extent, order)
        cost[name] = {
            "generation_seconds": perf_counter() - started,
            "current_density_evaluations": current_count,
            "normalized_flux_evaluations": flux_count,
        }

    started = perf_counter()
    (
        clipped,
        clipped_centroid,
        current_count,
        flux_count,
        clipped_counts,
    ) = _clipped_currents(case, radius, height, width, vertical_extent, classes)
    currents["clipped"] = clipped
    centroids["clipped"] = clipped_centroid
    cost["clipped"] = {
        "generation_seconds": perf_counter() - started,
        "current_density_evaluations": current_count,
        "normalized_flux_evaluations": flux_count,
        "current_density_evaluations_per_interior_cell": (
            clipped_counts["inside"] / np.count_nonzero(classes == "inside")
        ),
        "current_density_evaluations_per_boundary_cell": (
            clipped_counts["boundary"] / np.count_nonzero(classes == "boundary")
        ),
        "current_density_evaluations_per_outside_cell": (
            clipped_counts["outside"] / np.count_nonzero(classes == "outside")
        ),
    }
    return currents, cost, centroids


def _centroid_offset(
    clipped_centroid: np.ndarray,
    mesh: StencilMesh,
    boundary: np.ndarray,
    current: np.ndarray,
    width: float,
) -> dict[str, float | int]:
    """Report boundary-current centroid offsets from section centres."""

    selected = boundary & (current != 0.0)
    displacement = (clipped_centroid[selected] - mesh.coordinate[selected]) / width
    magnitude = np.linalg.norm(displacement, axis=1)
    return {
        "checked_boundary_cells": int(np.count_nonzero(selected)),
        "median_fraction_of_cell_width": float(np.median(magnitude)),
        "rms_fraction_of_cell_width": float(np.sqrt(np.mean(magnitude**2))),
        "sup_fraction_of_cell_width": float(np.max(magnitude)),
        "sup_radial_component_fraction_of_cell_width": float(
            np.max(np.abs(displacement[:, 0]))
        ),
        "sup_vertical_component_fraction_of_cell_width": float(
            np.max(np.abs(displacement[:, 1]))
        ),
    }


def _measure_resolution(
    radial_count: int, vertical_count: int, banked_sup: float, workers: int
) -> dict[str, Any]:
    """Measure all current representations on one grid."""

    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, width, vertical_extent = _hex_mesh(
        radial_count, vertical_count, case.major_radius, half_height
    )
    exact_current, classes = _exact_currents_and_classes(
        case, mesh, width, vertical_extent
    )
    variants, costs, current_centroids = _variant_currents(
        case, mesh, width, vertical_extent, classes
    )
    interior = classes == "inside"
    boundary = classes == "boundary"

    columns = [exact_current]
    for name in VARIANT_NAMES:
        columns.append(variants[name])
    for name in VARIANT_NAMES:
        columns.append((variants[name] - exact_current) * interior)
    for name in VARIANT_NAMES:
        columns.append((variants[name] - exact_current) * boundary)
    coupling_started = perf_counter()
    flux = _couple_vectors(
        mesh, np.column_stack(columns), width, vertical_extent, workers
    )
    coupling_seconds = perf_counter() - coupling_started

    results: dict[str, Any] = {}
    count = len(VARIANT_NAMES)
    for variant_index, name in enumerate(VARIANT_NAMES):
        total_field = flux[:, 1 + variant_index] - flux[:, 0]
        interior_field = flux[:, 1 + count + variant_index]
        boundary_field = flux[:, 1 + 2 * count + variant_index]
        zero_boundary = boundary & (exact_current != 0.0) & (variants[name] == 0.0)
        variant_cost = costs[name]
        results[name] = {
            "total": _flux_error(total_field),
            "interior_cell_contribution": _flux_error(interior_field),
            "boundary_cell_contribution": _flux_error(boundary_field),
            "zero_current_on_finite_exact_boundary_cells": int(
                np.count_nonzero(zero_boundary)
            ),
            "cost": {
                **variant_cost,
                "current_density_evaluations_per_cell": float(
                    variant_cost["current_density_evaluations"] / mesh.node_count
                ),
                "normalized_flux_evaluations_per_cell": float(
                    variant_cost["normalized_flux_evaluations"] / mesh.node_count
                ),
                "additional_green_evaluations_beyond_shared_matrix": 0,
            },
        }

    centroid_sup = results["centroid"]["total"]["sup_fraction_of_analytic_span"]
    relative_difference = abs(centroid_sup - banked_sup) / banked_sup
    if relative_difference > 5.0e-9:
        raise AssertionError(
            f"centroid baseline {centroid_sup:.12e} does not reproduce "
            f"{banked_sup:.12e}"
        )
    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "interior_cell_count": int(np.count_nonzero(interior)),
        "boundary_cell_count": int(np.count_nonzero(boundary)),
        "outside_cell_count": int(np.count_nonzero(classes == "outside")),
        "shared_exact_coupling_seconds": coupling_seconds,
        "centroid_baseline_relative_difference_from_banked": relative_difference,
        "variants": results,
        "current_centroid_offsets": {
            name: _centroid_offset(
                current_centroids[name], mesh, boundary, variants[name], width
            )
            for name in VARIANT_NAMES
        },
    }


def measure_representation_ladder(workers: int) -> dict[str, Any]:
    """Return the four-grid accuracy and cost ranking."""

    configure_dtypes()
    rows = [
        _measure_resolution(radial, vertical, banked, workers)
        for (radial, vertical), banked in zip(
            GRID_SEQUENCE, BANKED_CENTROID_SUP_FRACTIONS, strict=True
        )
    ]
    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in rows])
    fits: dict[str, Any] = {}
    for name in VARIANT_NAMES:
        sup = np.asarray(
            [
                row["variants"][name]["total"]["sup_fraction_of_analytic_span"]
                for row in rows
            ]
        )
        rms = np.asarray(
            [
                row["variants"][name]["total"]["rms_fraction_of_analytic_span"]
                for row in rows
            ]
        )
        fits[name] = {
            "sup": _fit_order(cell_size, sup),
            "rms": _fit_order(cell_size, rms),
        }

    finest = rows[-1]
    ranking = sorted(
        VARIANT_NAMES,
        key=lambda name: finest["variants"][name]["total"][
            "sup_fraction_of_analytic_span"
        ],
    )
    return {
        "analytic_source": {
            "name": ANALYTIC_CASE,
            "analytic_flux_span_wb": ANALYTIC_SPAN_WB,
            "stored_field_used": False,
            "interpolation_used": False,
        },
        "held_fixed": {
            "kernel": "cylinder_greens on every source-target pair",
            "filament_branch_used": False,
            "coupling_matrix_count_per_grid": 1,
            "additional_green_evaluations_for_any_representation": 0,
        },
        "cost_definition": {
            "centroid": (
                "one current-density and one normalized-flux evaluation per cell"
            ),
            "corners": (
                "four current-density and four normalized-flux evaluations per cell"
            ),
            "gauss_2": (
                "four current-density and four normalized-flux evaluations per cell"
            ),
            "gauss_4": (
                "sixteen current-density and sixteen normalized-flux evaluations "
                "per cell"
            ),
            "clipped": (
                "four normalized-flux edge values per cell; current density uses "
                "fixed fourth-order triangle quadrature over the linearly clipped "
                "polygon"
            ),
        },
        "resolutions": rows,
        "convergence": fits,
        "finest_grid_accuracy_ranking": ranking,
        "best_variant": {
            "name": ranking[0],
            "reason": (
                "lowest finest-grid sup flux difference from exact integrated assembly"
            ),
            "current_weighted_boundary_centroid_offsets_by_grid": [
                row["current_centroid_offsets"][ranking[0]] for row in rows
            ],
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
    """Print the representation ladder as stable JSON."""

    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    print(
        json.dumps(
            measure_representation_ladder(args.workers), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
