#!/usr/bin/env python3
"""Audit the private-region oracle, wedge convention, and plotted cell geometry.

The production read is used only as a comparison target.  Its private mask is
checked against a host NumPy breadth-first search over the authored confined
cell adjacency.  The saddle-wedge shortcut is then evaluated under explicit
convention alternatives, and every baseline disagreement is located in the
saddle Hessian frame and against the analytic separatrix branches.

The figure diagnostic also recreates the retired renderer's synthetic regular
hexagons and compares them with ``OracleMachine.cell_polygons``.  This keeps the
visual defect reproducible without using those synthetic polygons in a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import private_region_parallel_kernel as prk
from benchmarks import solovev_certificate as certificate
from benchmarks import xpoint_cell_allocation_rca as allocation_rca
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

REQUESTED_CELLS = (500, 750, 1000, 2500)


@dataclass(frozen=True)
class SaddleGeometry:
    """Local saddle basis and both kinds of candidate straight directions."""

    hessian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    eigenvector_rays: np.ndarray
    separatrix_tangent_rays: np.ndarray


def case_scene(requested_cells: int) -> tuple[Any, Any]:
    """Return the cached oracle machine and analytic single-null state."""
    carrier, _source, exact = certificate._case(prk.DIVERTED)
    machine = limiter_audit._machine(
        prk.DIVERTED, carrier, exact, -requested_cells, prk.WALL_NODE_COUNT
    )
    return machine, exact


def aligned_cell_polygons(bundle: prk._RungBundle, machine: Any) -> list[np.ndarray]:
    """Return each oracle polygon in the connectivity-coordinate ordering."""
    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    node = np.asarray(machine.node, dtype=np.float64)
    polygons = [np.asarray(item, dtype=np.float64) for item in machine.cell_polygons]
    if len(coordinate) != len(node) or len(node) != len(polygons):
        raise RuntimeError(
            "cell polygon positive control failed: coordinate, node, and polygon "
            f"counts are {len(coordinate)}, {len(node)}, and {len(polygons)}"
        )
    direct_error = np.linalg.norm(coordinate - node, axis=1)
    if float(np.max(direct_error)) <= 1.0e-12:
        return polygons

    distance = np.linalg.norm(coordinate[:, None, :] - node[None, :, :], axis=-1)
    assignment = np.argmin(distance, axis=1)
    errors = distance[np.arange(len(coordinate)), assignment]
    if len(np.unique(assignment)) != len(assignment) or float(np.max(errors)) > 1.0e-10:
        raise RuntimeError(
            "cell polygon positive control failed: machine.node cannot be aligned "
            f"to connectivity_coordinate (maximum error {float(np.max(errors)):.3e} m)"
        )
    return [polygons[int(index)] for index in assignment]


def independent_private_mask(
    bundle: prk._RungBundle,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flood the confined authored adjacency with a host NumPy queue."""
    psi = np.asarray(bundle.psi_grid, dtype=np.float64)
    polarity = int(bundle.operator.polarity)
    if polarity > 0:
        closed = psi >= float(bundle.boundary_flux)
    else:
        closed = psi < float(bundle.boundary_flux)
    confined = closed & np.asarray(bundle.inside, dtype=bool)
    rings = np.asarray(bundle.rings, dtype=np.int64)
    admissible = np.asarray(bundle.link_admissible, dtype=bool)
    seed_cells = np.flatnonzero(np.asarray(bundle.seed, dtype=bool) & confined)
    if len(seed_cells) != 1:
        raise RuntimeError(
            "breadth-first-search positive control failed: expected one confined "
            f"axis seed, found {len(seed_cells)}"
        )
    adjacency: list[set[int]] = [set() for _ in range(len(confined))]
    authored_edges = 0
    for row in range(len(rings)):
        centre = int(rings[row, 0])
        for slot in range(1, rings.shape[1]):
            if not admissible[row, slot]:
                continue
            neighbour = int(rings[row, slot])
            adjacency[centre].add(neighbour)
            adjacency[neighbour].add(centre)
            authored_edges += 1

    reached = np.zeros_like(confined, dtype=bool)
    queue: deque[int] = deque([int(seed_cells[0])])
    reached[seed_cells[0]] = True
    traversed_edges = 0
    while queue:
        cell = queue.popleft()
        for neighbour in adjacency[cell]:
            if not confined[neighbour] or reached[neighbour]:
                continue
            reached[neighbour] = True
            queue.append(neighbour)
            traversed_edges += 1
    private = confined & ~reached
    checks = {
        "axis_seed_cell": int(seed_cells[0]),
        "confined_cells": int(np.count_nonzero(confined)),
        "reached_cells": int(np.count_nonzero(reached)),
        "private_cells": int(np.count_nonzero(private)),
        "authored_ring_rows": int(len(rings)),
        "authored_directed_edges": authored_edges,
        "queue_edges_accepting_new_cell": traversed_edges,
        "axis_seed_reached": bool(reached[seed_cells[0]]),
    }
    if not checks["axis_seed_reached"] or checks["reached_cells"] == 0:
        raise RuntimeError(
            "breadth-first search did not see its known-present axis seed"
        )
    return private, checks


def saddle_geometry(exact: Any, x_point: np.ndarray) -> SaddleGeometry:
    """Return the Hessian basis and zero-level tangents at the saddle."""
    h = 1.0e-4

    def flux_at(offset_r: float, offset_z: float) -> float:
        point = np.asarray(x_point, dtype=np.float64) + (offset_r, offset_z)
        return float(limiter_audit._exact_flux(prk.DIVERTED, exact, point[None, :])[0])

    centre = flux_at(0.0, 0.0)
    hrr = (flux_at(h, 0.0) - 2.0 * centre + flux_at(-h, 0.0)) / h**2
    hzz = (flux_at(0.0, h) - 2.0 * centre + flux_at(0.0, -h)) / h**2
    hrz = (flux_at(h, h) - flux_at(h, -h) - flux_at(-h, h) + flux_at(-h, -h)) / (
        4.0 * h**2
    )
    hessian = np.asarray([[hrr, hrz], [hrz, hzz]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    if not (eigenvalues[0] < 0.0 < eigenvalues[1]):
        raise RuntimeError(f"analytic X-point Hessian is not indefinite: {eigenvalues}")

    eigenvector_rays = _orient_downward(eigenvectors.T)
    ratio = float(np.sqrt(-eigenvalues[0] / eigenvalues[1]))
    tangent_first = eigenvectors[:, 0] + ratio * eigenvectors[:, 1]
    tangent_second = eigenvectors[:, 0] - ratio * eigenvectors[:, 1]
    tangents = np.stack((tangent_first, tangent_second))
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = _orient_downward(tangents)
    null_residual = np.einsum("ni,ij,nj->n", tangents, hessian, tangents)
    if float(np.max(np.abs(null_residual))) > 1.0e-8 * float(
        np.max(np.abs(eigenvalues))
    ):
        raise RuntimeError(
            "constructed separatrix tangents do not null the saddle quadratic"
        )
    return SaddleGeometry(
        hessian=hessian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvector_rays=eigenvector_rays,
        separatrix_tangent_rays=tangents,
    )


def _orient_downward(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64).copy()
    return np.stack(
        [direction if direction[1] < 0.0 else -direction for direction in directions]
    )


def _cross2(directions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return (
        offsets[:, 1][:, None] * directions[:, 0][None, :]
        - offsets[:, 0][:, None] * directions[:, 1][None, :]
    )


def wedge_mask(
    bundle: prk._RungBundle,
    directions: np.ndarray,
    *,
    flip_flux_side: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one straight-ray wedge convention and return its side readings."""
    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    x_point = np.asarray(bundle.x_point, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    probe = np.asarray([x_point[0], x_point[1] - 1.0e-3], dtype=np.float64)
    probe_cross = _cross2(directions, (probe - x_point)[None, :])[0]
    side_sign = np.where(probe_cross > 0.0, 1.0, -1.0)
    cross = _cross2(directions, coordinate - x_point)
    private_side = side_sign[None, :] * cross >= 0.0
    wedge = np.all(private_side, axis=1)

    polarity = int(bundle.operator.polarity)
    if polarity >= 0:
        open_side = np.asarray(bundle.psi_grid) >= float(bundle.x_point_flux)
    else:
        open_side = np.asarray(bundle.psi_grid) <= float(bundle.x_point_flux)
    if flip_flux_side:
        open_side = ~open_side
    closed = np.asarray(
        bundle.topo.psi_mask(
            polarity,
            jnp.asarray(bundle.psi_grid),
            jnp.asarray(bundle.boundary_flux),
        ),
        dtype=bool,
    )
    return closed & np.asarray(bundle.inside) & open_side & wedge, cross, private_side


def _vertical_intersections(polyline: np.ndarray, radius: float) -> list[float]:
    intersections: list[float] = []
    polyline = np.asarray(polyline, dtype=np.float64)
    for first, second in zip(polyline[:-1], polyline[1:], strict=True):
        low, high = sorted((float(first[0]), float(second[0])))
        if radius < low or radius > high:
            continue
        span = float(second[0] - first[0])
        if abs(span) < 1.0e-14:
            if abs(radius - float(first[0])) < 1.0e-12:
                intersections.extend((float(first[1]), float(second[1])))
            continue
        fraction = (radius - float(first[0])) / span
        if 0.0 <= fraction <= 1.0:
            intersections.append(float(first[1] + fraction * (second[1] - first[1])))
    unique: list[float] = []
    for value in sorted(intersections):
        if not unique or abs(value - unique[-1]) > 1.0e-9:
            unique.append(value)
    return unique


def _line_height(direction: np.ndarray, x_point: np.ndarray, radius: float) -> float:
    if abs(float(direction[0])) < 1.0e-12:
        return float("nan")
    return float(x_point[1] + direction[1] / direction[0] * (radius - x_point[0]))


def _side_label(is_private: bool, cross: float) -> str:
    if abs(cross) <= 1.0e-14:
        return "on-ray"
    return "private" if is_private else "outside"


def disagreement_rows(
    bundle: prk._RungBundle,
    oracle_private: np.ndarray,
    baseline_mask: np.ndarray,
    geometry: SaddleGeometry,
    exact: Any,
) -> list[dict[str, Any]]:
    """Locate each baseline wedge disagreement in local and branch coordinates."""
    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    pitch = float(bundle.raster["pitch"])
    x_point = np.asarray(bundle.x_point, dtype=np.float64)
    _mask, cross, private_side = wedge_mask(bundle, geometry.eigenvector_rays)
    branches = allocation_rca._analytic_separatrix_branches(prk.DIVERTED, exact)
    analytic_lines = [np.asarray(branches["core_lobe"], dtype=np.float64)] + [
        np.asarray(item, dtype=np.float64) for item in branches["legs"]
    ]
    span = abs(float(bundle.axis_flux) - float(bundle.x_point_flux))
    rows: list[dict[str, Any]] = []
    for cell in np.flatnonzero(baseline_mask != oracle_private):
        point = coordinate[cell]
        local = geometry.eigenvectors.T @ (point - x_point) / pitch
        analytic_heights = sorted(
            value
            for polyline in analytic_lines
            for value in _vertical_intersections(polyline, float(point[0]))
        )
        eigen_heights = [
            _line_height(direction, x_point, float(point[0]))
            for direction in geometry.eigenvector_rays
        ]
        tangent_heights = [
            _line_height(direction, x_point, float(point[0]))
            for direction in geometry.separatrix_tangent_rays
        ]

        def separation(lines: list[float]) -> float | None:
            finite = [value for value in lines if np.isfinite(value)]
            if not finite or not analytic_heights:
                return None
            return float(
                min(
                    abs(first - second)
                    for first in finite
                    for second in analytic_heights
                )
                / pitch
            )

        rows.append(
            {
                "requested_cells": bundle.requested,
                "realised_cells": bundle.realised,
                "cell": int(cell),
                "radius_m": float(point[0]),
                "height_m": float(point[1]),
                "negative_curvature_coordinate_pitch": float(local[0]),
                "positive_curvature_coordinate_pitch": float(local[1]),
                "flux_minus_xpoint_wb": float(
                    bundle.psi_grid[cell] - bundle.x_point_flux
                ),
                "flux_minus_xpoint_fraction_of_span": float(
                    (bundle.psi_grid[cell] - bundle.x_point_flux) / span
                ),
                "first_eigenvector_side": _side_label(
                    bool(private_side[cell, 0]), float(cross[cell, 0])
                ),
                "second_eigenvector_side": _side_label(
                    bool(private_side[cell, 1]), float(cross[cell, 1])
                ),
                "oracle_private": bool(oracle_private[cell]),
                "baseline_wedge_private": bool(baseline_mask[cell]),
                "analytic_branch_heights_m": analytic_heights,
                "eigenvector_line_heights_m": eigen_heights,
                "separatrix_tangent_line_heights_m": tangent_heights,
                "nearest_eigenvector_line_to_analytic_branch_pitch": separation(
                    eigen_heights
                ),
                "nearest_tangent_line_to_analytic_branch_pitch": separation(
                    tangent_heights
                ),
            }
        )
    return rows


def _legacy_hexagon(
    centre: np.ndarray, neighbours: np.ndarray, pitch: float
) -> np.ndarray:
    """Reproduce the synthetic regular hexagon used by the retired renderer."""
    offset = neighbours - centre
    mean_angle = float(np.arctan2(np.mean(offset[:, 1]), np.mean(offset[:, 0])))
    lattice_step = int(np.rint(mean_angle / (np.pi / 6.0)))
    vertex_angle = (lattice_step + 0.5) * np.pi / 6.0
    angle = vertex_angle + np.arange(6) * (np.pi / 3.0)
    radius = pitch / np.sqrt(3.0)
    return centre + radius * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def legacy_renderer_diagnostic(
    bundle: prk._RungBundle,
    polygons: list[np.ndarray],
    selected: np.ndarray,
) -> dict[str, Any]:
    """Quantify the synthetic renderer against authored oracle polygons."""
    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    pitch = float(bundle.raster["pitch"])
    selected_cells = np.flatnonzero(selected)
    legacy: list[Polygon] = []
    authored: list[Polygon] = []
    intersection_over_union: list[float] = []
    centroid_error_pitch: list[float] = []
    area_ratio: list[float] = []
    for cell in selected_cells:
        distance2 = np.sum((coordinate - coordinate[cell]) ** 2, axis=1)
        neighbour = np.flatnonzero(
            (distance2 > 1.0e-9) & (distance2 <= (1.05 * pitch) ** 2)
        )
        if len(neighbour) == 0:
            raise RuntimeError(
                f"legacy renderer found no neighbour for cell {int(cell)}"
            )
        old = Polygon(_legacy_hexagon(coordinate[cell], coordinate[neighbour], pitch))
        own = Polygon(polygons[int(cell)])
        legacy.append(old)
        authored.append(own)
        union = old.union(own).area
        intersection_over_union.append(float(old.intersection(own).area / union))
        centroid_error_pitch.append(float(old.centroid.distance(own.centroid) / pitch))
        area_ratio.append(float(old.area / own.area))

    def overlap_pairs(items: list[Polygon]) -> tuple[int, float]:
        count = 0
        total = 0.0
        for index, first in enumerate(items):
            for second in items[index + 1 :]:
                overlap = float(first.intersection(second).area)
                if overlap > 1.0e-12 * pitch**2:
                    count += 1
                    total += overlap
        return count, total / pitch**2

    legacy_pairs, legacy_overlap = overlap_pairs(legacy)
    authored_pairs, authored_overlap = overlap_pairs(authored)
    return {
        "selected_cells": int(len(selected_cells)),
        "legacy_overlap_pairs": legacy_pairs,
        "legacy_overlap_area_pitch2": legacy_overlap,
        "authored_overlap_pairs": authored_pairs,
        "authored_overlap_area_pitch2": authored_overlap,
        "intersection_over_union_min": float(np.min(intersection_over_union)),
        "intersection_over_union_median": float(np.median(intersection_over_union)),
        "centroid_error_pitch_max": float(np.max(centroid_error_pitch)),
        "centroid_error_pitch_median": float(np.median(centroid_error_pitch)),
        "area_ratio_min": float(np.min(area_ratio)),
        "area_ratio_max": float(np.max(area_ratio)),
    }


def _difference(mask: np.ndarray, reference: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool) != reference))


def _mismatch_reason(
    bundle: prk._RungBundle,
    oracle_private: np.ndarray,
    production_rebuilt: np.ndarray,
) -> list[dict[str, Any]]:
    """Classify any production mismatch against the three authored mechanisms."""
    mismatches = np.flatnonzero(bundle.production_private != oracle_private)
    reasons: list[dict[str, Any]] = []
    for cell in mismatches:
        near_saddle = bool(
            np.linalg.norm(
                np.asarray(bundle.topo.connectivity_coordinate[cell]) - bundle.x_point
            )
            <= 3.0 * float(bundle.raster["pitch"])
        )
        if bool(production_rebuilt[cell]) == bool(oracle_private[cell]):
            candidate_reason = "representative-candidate loop"
        elif near_saddle:
            candidate_reason = "saddle-cell bridging rule"
        else:
            candidate_reason = "hysteresis band"
        reasons.append(
            {
                "cell": int(cell),
                "production_private": bool(bundle.production_private[cell]),
                "oracle_private": bool(oracle_private[cell]),
                "candidate_reason": candidate_reason,
            }
        )
    return reasons


def audit_rung(requested_cells: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run every independent and convention check for one cached mesh."""
    bundle = prk._build_rung(requested_cells)
    machine, exact = case_scene(requested_cells)
    polygons = aligned_cell_polygons(bundle, machine)
    oracle_private, bfs_checks = independent_private_mask(bundle)
    geometry = saddle_geometry(exact, np.asarray(bundle.x_point, dtype=np.float64))

    baseline, _cross, _side = wedge_mask(bundle, geometry.eigenvector_rays)
    sign_reversed, _cross, _side = wedge_mask(bundle, -geometry.eigenvector_rays)
    order_reversed, _cross, _side = wedge_mask(bundle, geometry.eigenvector_rays[::-1])
    flux_reversed, _cross, _side = wedge_mask(
        bundle, geometry.eigenvector_rays, flip_flux_side=True
    )
    tangent, _cross, _side = wedge_mask(bundle, geometry.separatrix_tangent_rays)
    pointer = np.asarray(
        jax.jit(lambda state: prk._mask_c(bundle, state))(jnp.asarray(bundle.psi_grid)),
        dtype=bool,
    )
    production_rebuilt = np.asarray(
        jax.jit(lambda state: prk._mask_a(bundle, state))(jnp.asarray(bundle.psi_grid)),
        dtype=bool,
    )
    prk_wedge = np.asarray(
        jax.jit(lambda state: prk._mask_d(bundle, state))(jnp.asarray(bundle.psi_grid)),
        dtype=bool,
    )
    if not np.array_equal(baseline, prk_wedge):
        raise RuntimeError(
            "audit reproduction does not match the banked wedge implementation"
        )

    details = disagreement_rows(bundle, oracle_private, baseline, geometry, exact)
    curvature = [
        row["nearest_tangent_line_to_analytic_branch_pitch"]
        for row in details
        if row["nearest_tangent_line_to_analytic_branch_pitch"] is not None
    ]
    renderer_selection = oracle_private | (baseline != oracle_private)
    row = {
        "requested": requested_cells,
        "realised": bundle.realised,
        "pitch_m": float(bundle.raster["pitch"]),
        "oracle_bfs": bfs_checks,
        "oracle_comparison": {
            "production_private_cells": int(
                np.count_nonzero(bundle.production_private)
            ),
            "numpy_bfs_private_cells": int(np.count_nonzero(oracle_private)),
            "production_flood_differing_cells": _difference(
                production_rebuilt, oracle_private
            ),
            "production_label_differing_cells": _difference(
                bundle.production_private, oracle_private
            ),
            "pointer_jumping_differing_cells": _difference(pointer, oracle_private),
            "mismatches": _mismatch_reason(bundle, oracle_private, production_rebuilt),
        },
        "saddle": {
            "x_point_rz_m": np.asarray(bundle.x_point).tolist(),
            "eigenvalues": geometry.eigenvalues.tolist(),
            "eigenvectors_columns": geometry.eigenvectors.tolist(),
            "eigenvector_quadratic_values": np.einsum(
                "ni,ij,nj->n",
                geometry.eigenvector_rays,
                geometry.hessian,
                geometry.eigenvector_rays,
            ).tolist(),
            "separatrix_tangent_quadratic_values": np.einsum(
                "ni,ij,nj->n",
                geometry.separatrix_tangent_rays,
                geometry.hessian,
                geometry.separatrix_tangent_rays,
            ).tolist(),
        },
        "wedge_candidates_differing_cells": {
            "banked_eigenvectors": _difference(baseline, oracle_private),
            "eigenvector_sign_reversed": _difference(sign_reversed, oracle_private),
            "eigenvector_order_reversed": _difference(order_reversed, oracle_private),
            "flux_side_inequality_reversed": _difference(flux_reversed, oracle_private),
            "zero_level_tangent_pair": _difference(tangent, oracle_private),
        },
        "convention_invariance": {
            "sign_reversal_changes_mask": _difference(sign_reversed, baseline),
            "order_reversal_changes_mask": _difference(order_reversed, baseline),
            "banked_matches_original_driver": _difference(baseline, prk_wedge),
        },
        "curvature_at_banked_disagreements_pitch": {
            "count": len(curvature),
            "median": float(np.median(curvature)) if curvature else None,
            "maximum": float(np.max(curvature)) if curvature else None,
        },
        "legacy_renderer_reproduction": legacy_renderer_diagnostic(
            bundle, polygons, renderer_selection
        ),
    }
    return row, details


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    if not scalar_rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# Private-region oracle and wedge audit",
        "",
        "The cell polygons in the evidence panels now come directly from "
        "`OracleMachine.cell_polygons`, aligned against `machine.node`. The retired "
        "renderer inferred a regular hexagon from neighbour centroids; its measured "
        "overlap and shape errors are reproduced below.",
        "",
        "## Oracle and convention counts",
        "",
        "| requested | realised | BFS != production | BFS != pointer | "
        "banked wedge != BFS | sign/order flip | flux flip | zero-level tangents |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in receipt["rungs"]:
        oracle = row["oracle_comparison"]
        candidates = row["wedge_candidates_differing_cells"]
        lines.append(
            f"| {row['requested']} | {row['realised']} | "
            f"{oracle['production_label_differing_cells']} | "
            f"{oracle['pointer_jumping_differing_cells']} | "
            f"{candidates['banked_eigenvectors']} | "
            f"{candidates['eigenvector_sign_reversed']} / "
            f"{candidates['eigenvector_order_reversed']} | "
            f"{candidates['flux_side_inequality_reversed']} | "
            f"{candidates['zero_level_tangent_pair']} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            receipt["verdict"],
            "",
            "The per-cell CSV gives coordinates in the negative- and "
            "positive-curvature "
            "eigenvector frame, flux relative to the X-point, both ray-side verdicts, "
            "and the straight-line-to-analytic-branch separation at the same radius.",
            "",
            "## Corrected evidence panels",
            "",
            "![private mask, 1074 cells]"
            "(/nova/figures/cut-cell-current-attribution/private-region/"
            "private-mask-poloidal-1074.svg)",
            "",
            "![private mask, 2616 cells]"
            "(/nova/figures/cut-cell-current-attribution/private-region/"
            "private-mask-poloidal-2616.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=list(REQUESTED_CELLS))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rungs: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for requested in args.cells:
        row, rung_details = audit_rung(requested)
        rungs.append(row)
        details.extend(rung_details)
        oracle = row["oracle_comparison"]
        print(
            f"requested {requested}, realised {row['realised']}: "
            f"BFS/production {oracle['production_label_differing_cells']}, "
            f"BFS/pointer {oracle['pointer_jumping_differing_cells']}, "
            f"wedge {row['wedge_candidates_differing_cells']}"
        )

    exact_oracle = all(
        row["oracle_comparison"][key] == 0
        for row in rungs
        for key in (
            "production_flood_differing_cells",
            "production_label_differing_cells",
            "pointer_jumping_differing_cells",
        )
    )
    tangent_counts = [
        row["wedge_candidates_differing_cells"]["zero_level_tangent_pair"]
        for row in rungs
    ]
    baseline_counts = [
        row["wedge_candidates_differing_cells"]["banked_eigenvectors"] for row in rungs
    ]
    if all(count == 0 for count in tangent_counts):
        wedge_verdict = (
            "Replacing the Hessian eigenvectors with the saddle quadratic's zero-level "
            "tangents makes the wedge exact at every measured rung; the prior "
            "curvature claim was an instrument convention error."
        )
    else:
        wedge_verdict = (
            "The zero-level tangent pair reduces the banked disagreement counts from "
            f"{baseline_counts} to {tangent_counts}; the remaining cells are the "
            "finite-radius curvature residual quantified in pitch units in the "
            "receipt and CSV."
        )
    verdict = (
        (
            "The independent NumPy breadth-first search confirms both the production "
            "flood and pointer jumping cell by cell at all four rungs. "
            if exact_oracle
            else "The independent NumPy breadth-first search found an oracle "
            "disagreement; the per-cell reason census in the receipt is authoritative. "
        )
        + wedge_verdict
        + " The banked costs remain 0.007 to 0.016 ms for the wedge and 0.020 to "
        "0.047 ms for pointer jumping."
    )
    receipt = {
        "rungs": rungs,
        "disagreement_cells": details,
        "oracle_confirmed": exact_oracle,
        "verdict": verdict,
        "timing_context_ms": {
            "banked_wedge_range": [0.007, 0.016],
            "banked_pointer_jumping_range": [0.020, 0.047],
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    _write_csv(args.csv, details)
    _write_report(args.report, receipt)
    print(verdict)
    print("wrote", args.output)
    print("wrote", args.csv)
    print("wrote", args.report)


if __name__ == "__main__":
    main()
