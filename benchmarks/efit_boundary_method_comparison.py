"""Compare separatrix-cell current representations against clipped integration.

The reference integrates the closed-form current density over the repaired
shared-edge polygon and couples its uniform part with the exact finite-section
kernel.  Only the smooth density remainder is numerically quadratured.  The
candidate arms keep full-cell sections fixed except for the explicitly moving
boundary filament.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from typing import Any

import numpy as np

from benchmarks.efit_clipped_area_conservation import (
    CONSERVATION_RELATIVE_TOLERANCE,
    _clip_polygon,
    _patch_receipt,
    _rectangle_vertices,
    _signed_polygon_area,
)
from benchmarks.efit_constant_current_attribution import (
    ANALYTIC_CASE,
    GRID_SEQUENCE,
    _fit_order,
    _hex_mesh,
)
from nova.biot.greens import cylinder_greens, greens_psi
from nova.biot.polygonanalytic import polygon_analytic_flux
from nova.jax.config import configure_dtypes
from tests.rotating_equilibrium_references import RotatingEquilibrium, reference_cases

ANALYTIC_SPAN_WB = 1.3812146953272917
BANKED_T_JUNCTION_CLIPPED_SUP = (
    2.103742358e-2,
    7.755259037e-3,
    3.980868721e-3,
    2.235201860e-3,
)
CURRENT_MOMENT_ORDER = 8
COARSE_COUPLING_ORDER = 4
REFERENCE_COUPLING_ORDER = 6
COARSE_BLOCK_SUBDIVISION = 4
FINE_BLOCK_SUBDIVISION = 6
POINT_BATCH_SIZE = 64
REFERENCE_DRIFT_LIMIT = 5.0e-5
ARM_NAMES = (
    "production_centroid",
    "clipped_geometric_centroid",
    "moving_current_centroid_filament",
    "fixed_linear_blocks",
)


def _clean_polygon(vertices: np.ndarray) -> np.ndarray:
    """Remove repeated and collinear vertices without changing polygon area."""

    if len(vertices) < 3:
        return np.empty((0, 2))
    cleaned: list[np.ndarray] = []
    for vertex in vertices:
        if not cleaned or not np.allclose(vertex, cleaned[-1], rtol=0.0, atol=1e-14):
            cleaned.append(vertex)
    if len(cleaned) > 1 and np.allclose(cleaned[0], cleaned[-1], rtol=0.0, atol=1e-14):
        cleaned.pop()
    changed = True
    while changed and len(cleaned) > 3:
        changed = False
        retained: list[np.ndarray] = []
        count = len(cleaned)
        for index, vertex in enumerate(cleaned):
            prior = cleaned[(index - 1) % count]
            following = cleaned[(index + 1) % count]
            incoming = vertex - prior
            outgoing = following - vertex
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            scale = np.linalg.norm(vertex - prior) * np.linalg.norm(following - vertex)
            if abs(float(cross)) <= 1e-13 * max(float(scale), 1.0):
                changed = True
            else:
                retained.append(vertex)
        cleaned = retained
    polygon = np.asarray(cleaned)
    if _signed_polygon_area(polygon) < 0.0:
        polygon = polygon[::-1]
    return polygon


def _cell_polygons(
    case: RotatingEquilibrium,
    coordinates: np.ndarray,
    width: float,
    vertical_extent: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return repaired clipped polygons and their geometric classes."""

    polygons: list[np.ndarray] = []
    classes = np.full(len(coordinates), "outside", dtype=object)
    for index, (radius, height) in enumerate(coordinates):
        split = _rectangle_vertices(
            float(radius),
            float(height),
            width,
            vertical_extent,
            split_horizontal_edges=True,
        )
        polygon, crossings = _clip_polygon(case, split)
        if len(crossings):
            classes[index] = "boundary"
            polygons.append(_clean_polygon(polygon))
        elif bool(case.contains(float(radius), float(height))):
            classes[index] = "inside"
            polygons.append(
                _rectangle_vertices(
                    float(radius),
                    float(height),
                    width,
                    vertical_extent,
                    split_horizontal_edges=False,
                )
            )
        else:
            polygons.append(np.empty((0, 2)))
    return polygons, classes


def _triangle_nodes(triangle: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return tensor-Duffy nodes and area weights for one triangle."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    first, second, third = triangle
    first_edge = second - first
    second_edge = third - first
    double_area = abs(
        float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
    )
    points: list[np.ndarray] = []
    area_weights: list[float] = []
    for radial_node, radial_weight in zip(unit_nodes, unit_weights, strict=True):
        for edge_node, edge_weight in zip(unit_nodes, unit_weights, strict=True):
            points.append(
                first
                + radial_node
                * ((1.0 - edge_node) * first_edge + edge_node * second_edge)
            )
            area_weights.append(double_area * radial_node * radial_weight * edge_weight)
    return np.asarray(points), np.asarray(area_weights)


def _polygon_nodes(polygon: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature nodes and area weights for a convex cell polygon."""

    if len(polygon) < 3:
        return np.empty((0, 2)), np.empty(0)
    point_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    for index in range(1, len(polygon) - 1):
        points, weights = _triangle_nodes(polygon[[0, index, index + 1]], order)
        point_rows.append(points)
        weight_rows.append(weights)
    return np.vstack(point_rows), np.concatenate(weight_rows)


def _current_moments(
    case: RotatingEquilibrium,
    coordinates: np.ndarray,
    polygons: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate clipped current and first moments about fixed cell centres."""

    current = np.zeros(len(coordinates))
    radial_moment = np.zeros(len(coordinates))
    vertical_moment = np.zeros(len(coordinates))
    current_centres = coordinates.copy()
    for index, (centre, polygon) in enumerate(zip(coordinates, polygons, strict=True)):
        points, weights = _polygon_nodes(polygon, CURRENT_MOMENT_ORDER)
        if not len(points):
            continue
        weighted_current = (
            np.asarray(case.toroidal_current_density(points[:, 0], points[:, 1]))
            * weights
        )
        current[index] = np.sum(weighted_current)
        displacement = points - centre
        radial_moment[index] = np.sum(weighted_current * displacement[:, 0])
        vertical_moment[index] = np.sum(weighted_current * displacement[:, 1])
        if current[index] != 0.0:
            current_centres[index] = (
                centre
                + np.asarray([radial_moment[index], vertical_moment[index]])
                / current[index]
            )
    return current, radial_moment, vertical_moment, current_centres


def _section_flux_chunk(
    source_indices: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
    coordinates: np.ndarray,
    polygons: list[np.ndarray],
    classes: np.ndarray,
    production_current: np.ndarray,
    clipped_current: np.ndarray,
    current_centres: np.ndarray,
    centre_density: np.ndarray,
    width: float,
    vertical_extent: float,
) -> np.ndarray:
    """Assemble fixed-section arms, moving filaments, and reference base."""

    fields = np.zeros((5, len(target_r)))
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in source_indices:
            radius, height = coordinates[source]
            fixed_response = cylinder_greens(
                target_r,
                target_z,
                float(radius),
                float(height),
                width,
                vertical_extent,
            )[0]
            fields[0] += production_current[source] * fixed_response
            fields[1] += clipped_current[source] * fixed_response
            fields[3] += clipped_current[source] * fixed_response
            if classes[source] == "boundary":
                moving_response = greens_psi(
                    target_r,
                    target_z,
                    float(current_centres[source, 0]),
                    float(current_centres[source, 1]),
                )
                reference_response = polygon_analytic_flux(
                    target_r, target_z, polygons[source]
                )
            else:
                moving_response = fixed_response
                reference_response = fixed_response
            fields[2] += clipped_current[source] * moving_response
            area = abs(_signed_polygon_area(polygons[source]))
            fields[4] += centre_density[source] * area * reference_response
    return fields


def _section_fluxes(
    target_r: np.ndarray,
    target_z: np.ndarray,
    coordinates: np.ndarray,
    polygons: list[np.ndarray],
    classes: np.ndarray,
    production_current: np.ndarray,
    clipped_current: np.ndarray,
    current_centres: np.ndarray,
    centre_density: np.ndarray,
    width: float,
    vertical_extent: float,
    workers: int,
) -> np.ndarray:
    """Assemble section responses by independent source chunks."""

    active = np.flatnonzero((production_current != 0.0) | (clipped_current != 0.0))
    chunks = [row for row in np.array_split(active, workers) if len(row)]
    arguments = (
        target_r,
        target_z,
        coordinates,
        polygons,
        classes,
        production_current,
        clipped_current,
        current_centres,
        centre_density,
        width,
        vertical_extent,
    )
    if len(chunks) == 1:
        rows = [_section_flux_chunk(chunks[0], *arguments)]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(_section_flux_chunk, chunk, *arguments)
                for chunk in chunks
            ]
            rows = [future.result() for future in futures]
    fields = np.sum(rows, axis=0)
    if not np.all(np.isfinite(fields)):
        raise FloatingPointError("section assembly produced non-finite flux")
    return fields


def _point_flux_chunk(
    points: np.ndarray,
    currents: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    """Couple point-current batches without materialising a full matrix."""

    field = np.zeros(len(target_r))
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for start in range(0, len(points), POINT_BATCH_SIZE):
            batch = slice(start, start + POINT_BATCH_SIZE)
            response = greens_psi(
                target_r[:, None],
                target_z[:, None],
                points[None, batch, 0],
                points[None, batch, 1],
            )
            field += response @ currents[batch]
    return field


def _point_flux(
    points: np.ndarray,
    currents: np.ndarray,
    target_r: np.ndarray,
    target_z: np.ndarray,
    workers: int,
) -> np.ndarray:
    """Couple a distributed point-current quadrature."""

    selected = currents != 0.0
    points = points[selected]
    currents = currents[selected]
    point_chunks = [row for row in np.array_split(points, workers) if len(row)]
    current_chunks = [row for row in np.array_split(currents, workers) if len(row)]
    if len(point_chunks) == 1:
        rows = [
            _point_flux_chunk(point_chunks[0], current_chunks[0], target_r, target_z)
        ]
    else:
        with ProcessPoolExecutor(max_workers=len(point_chunks)) as executor:
            futures = [
                executor.submit(
                    _point_flux_chunk,
                    point_chunk,
                    current_chunk,
                    target_r,
                    target_z,
                )
                for point_chunk, current_chunk in zip(
                    point_chunks, current_chunks, strict=True
                )
            ]
            rows = [future.result() for future in futures]
    field = np.sum(rows, axis=0)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("point quadrature produced non-finite flux")
    return field


def _reference_correction_nodes(
    case: RotatingEquilibrium,
    coordinates: np.ndarray,
    polygons: list[np.ndarray],
    centre_density: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped nodes for the smooth density remainder of the reference."""

    point_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    for centre, polygon, baseline in zip(
        coordinates, polygons, centre_density, strict=True
    ):
        del centre
        points, weights = _polygon_nodes(polygon, order)
        if not len(points):
            continue
        density = np.asarray(case.toroidal_current_density(points[:, 0], points[:, 1]))
        point_rows.append(points)
        current_rows.append((density - baseline) * weights)
    return np.vstack(point_rows), np.concatenate(current_rows)


def _linear_block_nodes(
    coordinates: np.ndarray,
    width: float,
    vertical_extent: float,
    radial_moment: np.ndarray,
    vertical_moment: np.ndarray,
    subdivision: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a converged full-cell subdivision of the two linear blocks."""

    nodes = (np.arange(subdivision) + 0.5) / subdivision - 0.5
    area_weight = width * vertical_extent / subdivision**2
    radial_second_moment = width**3 * vertical_extent / 12.0
    vertical_second_moment = width * vertical_extent**3 / 12.0
    point_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    active = np.flatnonzero((radial_moment != 0.0) | (vertical_moment != 0.0))
    for source in active:
        radius, height = coordinates[source]
        source_r = radius + width * nodes[:, None]
        source_z = height + vertical_extent * nodes[None, :]
        radial_grid, vertical_grid = np.broadcast_arrays(source_r, source_z)
        displacement_r = radial_grid - radius
        displacement_z = vertical_grid - height
        effective_current = area_weight * (
            radial_moment[source] / radial_second_moment * displacement_r
            + vertical_moment[source] / vertical_second_moment * displacement_z
        )
        point_rows.append(np.column_stack((radial_grid.ravel(), vertical_grid.ravel())))
        current_rows.append(effective_current.ravel())
    return np.vstack(point_rows), np.concatenate(current_rows)


def _flux_error(field: np.ndarray) -> dict[str, float]:
    """Return flux error as sup and RMS fractions of the analytic span."""

    return {
        "sup_fraction_of_analytic_span": float(
            np.max(np.abs(field)) / ANALYTIC_SPAN_WB
        ),
        "rms_fraction_of_analytic_span": float(
            np.sqrt(np.mean(field**2)) / ANALYTIC_SPAN_WB
        ),
    }


def _measure_resolution(
    radial_count: int,
    vertical_count: int,
    banked_clipped_sup: float,
    workers: int,
) -> dict[str, Any]:
    """Compare all boundary representations on one analytic grid."""

    case = reference_cases()[ANALYTIC_CASE]
    half_height = float(np.sqrt(case.axis_flux / case.field_coefficient))
    mesh, width, vertical_extent = _hex_mesh(
        radial_count, vertical_count, case.major_radius, half_height
    )
    conservation = _patch_receipt(
        case,
        mesh.coordinate,
        width,
        vertical_extent,
        split_horizontal_edges=True,
    )
    if conservation["relative_conservation_residual"] > CONSERVATION_RELATIVE_TOLERANCE:
        raise AssertionError("repaired clip no longer conserves area")
    polygons, classes = _cell_polygons(case, mesh.coordinate, width, vertical_extent)
    clipped_current, radial_moment, vertical_moment, current_centres = _current_moments(
        case, mesh.coordinate, polygons
    )
    radius = np.asarray(mesh.coordinate[:, 0])
    height = np.asarray(mesh.coordinate[:, 1])
    centre_density = np.asarray(case.toroidal_current_density(radius, height))
    production_current = np.where(
        case.contains(radius, height),
        centre_density * width * vertical_extent,
        0.0,
    )
    section_fields = _section_fluxes(
        radius,
        height,
        mesh.coordinate,
        polygons,
        classes,
        production_current,
        clipped_current,
        current_centres,
        centre_density,
        width,
        vertical_extent,
        workers,
    )

    reference_rows = {}
    fixed_rows = {}
    for order in (COARSE_COUPLING_ORDER, REFERENCE_COUPLING_ORDER):
        reference_points, reference_currents = _reference_correction_nodes(
            case, mesh.coordinate, polygons, centre_density, order
        )
        reference_rows[order] = section_fields[4] + _point_flux(
            reference_points, reference_currents, radius, height, workers
        )
    for subdivision in (COARSE_BLOCK_SUBDIVISION, FINE_BLOCK_SUBDIVISION):
        fixed_points, fixed_currents = _linear_block_nodes(
            mesh.coordinate,
            width,
            vertical_extent,
            radial_moment,
            vertical_moment,
            subdivision,
        )
        fixed_rows[subdivision] = section_fields[3] + _point_flux(
            fixed_points, fixed_currents, radius, height, workers
        )
    reference = reference_rows[REFERENCE_COUPLING_ORDER]
    reference_drift = _flux_error(reference - reference_rows[COARSE_COUPLING_ORDER])
    if reference_drift["sup_fraction_of_analytic_span"] > REFERENCE_DRIFT_LIMIT:
        raise AssertionError(
            "clipped exact-reference quadrature is not converged: "
            f"{reference_drift['sup_fraction_of_analytic_span']:.6e}"
        )
    fixed_drift = _flux_error(
        fixed_rows[FINE_BLOCK_SUBDIVISION] - fixed_rows[COARSE_BLOCK_SUBDIVISION]
    )
    arm_flux = {
        "production_centroid": section_fields[0],
        "clipped_geometric_centroid": section_fields[1],
        "moving_current_centroid_filament": section_fields[2],
        "fixed_linear_blocks": fixed_rows[FINE_BLOCK_SUBDIVISION],
    }
    errors = {name: _flux_error(field - reference) for name, field in arm_flux.items()}
    repaired_sup = errors["clipped_geometric_centroid"]["sup_fraction_of_analytic_span"]
    boundary = classes == "boundary"
    moving_green_count = int(np.count_nonzero(boundary & (clipped_current != 0.0)))
    return {
        "radial_count": radial_count,
        "vertical_count": vertical_count,
        "cell_count": mesh.node_count,
        "characteristic_cell_size_m": float(np.sqrt(mesh.cell_area[0])),
        "repaired_clip_precondition": {
            "relative_conservation_residual": conservation[
                "relative_conservation_residual"
            ],
            "required_maximum": CONSERVATION_RELATIVE_TOLERANCE,
            "passed": True,
            "boundary_cell_count": int(np.count_nonzero(boundary)),
        },
        "reference_convergence": {
            "coarse_order": COARSE_COUPLING_ORDER,
            "fine_order": REFERENCE_COUPLING_ORDER,
            **reference_drift,
        },
        "fixed_block_convergence": {
            "coarse_subdivision": COARSE_BLOCK_SUBDIVISION,
            "fine_subdivision": FINE_BLOCK_SUBDIVISION,
            **fixed_drift,
        },
        "arms": errors,
        "clipped_geometric_centroid_reaudit": {
            "banked_t_junction_sup_fraction": banked_clipped_sup,
            "repaired_shared_edge_sup_fraction": repaired_sup,
            "signed_shift": repaired_sup - banked_clipped_sup,
            "relative_shift": repaired_sup / banked_clipped_sup - 1.0,
        },
        "per_iteration_cost": {
            "production_centroid": {
                "green_evaluations": 0,
                "matrix_vector_products": 1,
            },
            "clipped_geometric_centroid": {
                "green_evaluations": 0,
                "matrix_vector_products": 1,
            },
            "moving_current_centroid_filament": {
                "green_evaluations": moving_green_count,
                "matrix_vector_products": 1,
            },
            "fixed_linear_blocks": {
                "green_evaluations": 0,
                "matrix_vector_products": 3,
            },
        },
    }


def _arm_contracts() -> dict[str, dict[str, Any]]:
    """Return traceability and scrape-off-layer extension contracts."""

    return {
        "production_centroid": {
            "fixed_array_shapes": True,
            "traceable": True,
            "cached_coupling_fixed": True,
            "sol_extension": (
                "carry nonzero exterior centroid currents in the existing vector; "
                "the boundary-cell centroid bias remains"
            ),
        },
        "clipped_geometric_centroid": {
            "fixed_array_shapes": True,
            "traceable": True,
            "cached_coupling_fixed": True,
            "sol_extension": (
                "integrate the nonzero exterior current over an outer support cut "
                "and keep the same fixed full-cell matrix"
            ),
        },
        "moving_current_centroid_filament": {
            "fixed_array_shapes": True,
            "traceable": True,
            "cached_coupling_fixed": False,
            "sol_extension": (
                "include the exterior support in each cell total and current centre; "
                "the per-step filament evaluations remain"
            ),
        },
        "fixed_linear_blocks": {
            "fixed_array_shapes": True,
            "traceable": True,
            "cached_coupling_fixed": True,
            "sol_extension": (
                "add exterior current to the same zeroth and first-moment vectors; "
                "no coupling block changes while the fixed cell band is retained"
            ),
        },
    }


def measure_boundary_methods(workers: int) -> dict[str, Any]:
    """Return four-grid accuracy, convergence, and solve-cost ranking."""

    configure_dtypes()
    rows = [
        _measure_resolution(radial, vertical, banked, workers)
        for (radial, vertical), banked in zip(
            GRID_SEQUENCE, BANKED_T_JUNCTION_CLIPPED_SUP, strict=True
        )
    ]
    cell_size = np.asarray([row["characteristic_cell_size_m"] for row in rows])
    convergence: dict[str, Any] = {}
    for name in ARM_NAMES:
        sup = np.asarray(
            [row["arms"][name]["sup_fraction_of_analytic_span"] for row in rows]
        )
        rms = np.asarray(
            [row["arms"][name]["rms_fraction_of_analytic_span"] for row in rows]
        )
        convergence[name] = {
            "sup": _fit_order(cell_size, sup),
            "rms": _fit_order(cell_size, rms),
        }
    finest = rows[-1]
    ranking = sorted(
        ARM_NAMES,
        key=lambda name: finest["arms"][name]["sup_fraction_of_analytic_span"],
    )
    production_sup = finest["arms"]["production_centroid"][
        "sup_fraction_of_analytic_span"
    ]
    clipped_sup = finest["arms"]["clipped_geometric_centroid"][
        "sup_fraction_of_analytic_span"
    ]
    moving_sup = finest["arms"]["moving_current_centroid_filament"][
        "sup_fraction_of_analytic_span"
    ]
    fixed_sup = finest["arms"]["fixed_linear_blocks"]["sup_fraction_of_analytic_span"]
    return {
        "analytic_source": ANALYTIC_CASE,
        "analytic_flux_span_wb": ANALYTIC_SPAN_WB,
        "grid_sequence": [list(row) for row in GRID_SEQUENCE],
        "held_fixed": {
            "reference": (
                "repaired shared-edge clipped polygons, exact uniform section "
                "kernel, and converged quadrature of the smooth density remainder"
            ),
            "production_and_geometric_kernel": "cylinder_greens on every cell",
            "fixed_linear_blocks": (
                "G0 exact; GR and GZ converged over the full fixed section"
            ),
            "matrix_build_cost_in_ranking": "irrelevant by the fixed-geometry contract",
        },
        "resolutions": rows,
        "convergence": convergence,
        "arm_contracts": _arm_contracts(),
        "finest_grid_accuracy_ranking": ranking,
        "conclusions": {
            "accuracy_at_equal_iteration_cost": {
                "winner": (
                    "clipped_geometric_centroid"
                    if clipped_sup < production_sup
                    else "production_centroid"
                ),
                "comparison": (
                    "one fixed matrix-vector product and zero Green evaluations"
                ),
                "production_sup_fraction": production_sup,
                "clipped_sup_fraction": clipped_sup,
                "accuracy_ratio_clipped_over_production": clipped_sup / production_sup,
            },
            "cost_at_equal_or_better_accuracy": {
                "winner": (
                    "fixed_linear_blocks"
                    if fixed_sup <= moving_sup
                    else "moving_current_centroid_filament"
                ),
                "moving_sup_fraction": moving_sup,
                "fixed_sup_fraction": fixed_sup,
                "moving_green_evaluations": finest["per_iteration_cost"][
                    "moving_current_centroid_filament"
                ]["green_evaluations"],
                "fixed_green_evaluations": 0,
                "moving_matvecs": 1,
                "fixed_matvecs": 3,
                "qualification": (
                    "fixed blocks dominate moving filaments in accuracy and remove "
                    "all per-step Green evaluations"
                    if fixed_sup <= moving_sup
                    else "moving filaments are more accurate, so no fixed arm reaches "
                    "equal accuracy; fixed blocks trade that gap for zero per-step "
                    "Green evaluations"
                ),
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="independent source chunks",
    )
    return parser


def main() -> None:
    """Print the comparison receipt as stable JSON."""

    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    print(json.dumps(measure_boundary_methods(args.workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
