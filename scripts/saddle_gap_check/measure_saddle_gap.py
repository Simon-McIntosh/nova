"""Localize the saddle-cut current gap without changing source topology."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import triangulate, unary_union

from scripts.coverage_scaling.measure_coverage import (
    build_geometry,
    integrate_polygon,
    load_reference,
    normalised_source_flux,
    refined_analytic_core,
    solve_flux_saddle,
    source_current_density,
)


FIXTURE = Path("scripts/ring_quadrature/inputs/coarse-fixture-reference-inputs.npz")
LOCALIZATION = Path("scripts/ring_quadrature/inputs/source-shift-localization.npz")
PRODUCTION = Path("scripts/ring_attribution/results/ring-attribution-fields.npz")
EXPECTED_CORE_CURRENT_A = -14_999_238.312516855
EXPECTED_TARGET_CURRENT_A = -14_995_826.425829582


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-points", type=int, default=8192)
    parser.add_argument("--results", type=Path, default=directory / "results.json")
    parser.add_argument("--figure", type=Path, default=directory / "saddle-gap.png")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def polygon_parts(geometry) -> list[ShapelyPolygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return [
            polygon
            for item in geometry.geoms
            for polygon in polygon_parts(item)
            if polygon.area > 0.0
        ]
    return []


def polygon_current(case, geometry) -> float:
    currents = []
    for polygon in polygon_parts(geometry):
        convex_area_delta = abs(polygon.convex_hull.area - polygon.area)
        area_tolerance = 1.0e-12 * max(polygon.area, 1.0)
        if not polygon.interiors and convex_area_delta <= area_tolerance:
            exterior = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
            currents.append(integrate_polygon(case, exterior))
            continue
        covered_area = 0.0
        for candidate in triangulate(polygon):
            for piece in polygon_parts(candidate.intersection(polygon)):
                if piece.area <= area_tolerance:
                    continue
                exterior = np.asarray(piece.exterior.coords, dtype=float)[:-1, :2]
                currents.append(integrate_polygon(case, exterior))
                covered_area += piece.area
        if abs(covered_area - polygon.area) > area_tolerance:
            raise AssertionError("triangulated integration does not cover the polygon")
    return math.fsum(currents)


def geometry_coordinates(geometry) -> np.ndarray:
    points = []
    for polygon in polygon_parts(geometry):
        points.extend(np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2])
        representative = polygon.representative_point()
        points.append(np.asarray(representative.coords[0], dtype=float))
    return np.asarray(points) if points else np.empty((0, 2))


def psi_range(case, geometry) -> list[float] | None:
    coordinates = geometry_coordinates(geometry)
    if not len(coordinates):
        return None
    values = np.asarray(
        normalised_source_flux(case, coordinates[:, 0], coordinates[:, 1])
    )
    return [float(np.min(values)), float(np.max(values))]


def edge_crossings(case, vertices: np.ndarray) -> list[dict[str, object]]:
    crossings = []
    for edge_index, start in enumerate(vertices):
        end = vertices[(edge_index + 1) % len(vertices)]
        fractions = np.linspace(0.0, 1.0, 257)
        coordinates = start + fractions[:, None] * (end - start)
        values = (
            np.asarray(
                normalised_source_flux(case, coordinates[:, 0], coordinates[:, 1])
            )
            - 1.0
        )
        roots = []
        for index in range(len(fractions) - 1):
            left, right = values[index : index + 2]
            if left == 0.0:
                roots.append(float(fractions[index]))
            elif left * right < 0.0:
                root = brentq(
                    lambda fraction: float(
                        normalised_source_flux(
                            case, *(start + fraction * (end - start))
                        )
                        - 1.0
                    ),
                    float(fractions[index]),
                    float(fractions[index + 1]),
                )
                roots.append(root)
        for fraction in roots:
            coordinate = start + fraction * (end - start)
            if not any(
                np.linalg.norm(coordinate - np.asarray(item["coordinate_m"])) < 1.0e-10
                for item in crossings
            ):
                crossings.append(
                    {
                        "edge": edge_index,
                        "edge_fraction": fraction,
                        "coordinate_m": coordinate.tolist(),
                    }
                )
    return crossings


def clip_chords(
    material: ShapelyPolygon, support: np.ndarray
) -> list[list[list[float]]]:
    chords = []
    for index, start in enumerate(support):
        end = support[(index + 1) % len(support)]
        midpoint = 0.5 * (start + end)
        if material.boundary.distance(Point(midpoint)) > 1.0e-9:
            chords.append([start.tolist(), end.tolist()])
    return chords


def edge_neighbours(material: list[ShapelyPolygon], cell: int) -> list[int]:
    scale = math.sqrt(material[cell].area)
    return [
        index
        for index, polygon in enumerate(material)
        if index != cell
        and material[cell].boundary.intersection(polygon.boundary).length
        > 1.0e-8 * scale
    ]


def nearest_cell(geometry, material: list[ShapelyPolygon]) -> int:
    distances = [geometry.distance(polygon) for polygon in material]
    return int(np.argmin(distances))


def fixed_sector_currents(
    case, axis: np.ndarray, start: np.ndarray, end: np.ndarray, depth: int = 4
) -> np.ndarray:
    """Apply the banked triangle rule while retaining one value per fan sector."""
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )
    weight = np.asarray([-27.0, 25.0, 25.0, 25.0]) / 48.0
    result = np.empty(len(start))
    for first in range(0, len(start), 256):
        stop = min(first + 256, len(start))
        count = stop - first
        triangles = np.stack(
            [
                np.broadcast_to(axis, (count, 2)),
                start[first:stop],
                end[first:stop],
            ],
            axis=1,
        )[:, None, :, :]
        for _ in range(depth):
            one, two, three = np.moveaxis(triangles, 2, 0)
            one_two = 0.5 * (one + two)
            two_three = 0.5 * (two + three)
            three_one = 0.5 * (three + one)
            triangles = np.concatenate(
                [
                    np.stack([one, one_two, three_one], axis=2),
                    np.stack([one_two, two, two_three], axis=2),
                    np.stack([three_one, two_three, three], axis=2),
                    np.stack([one_two, two_three, three_one], axis=2),
                ],
                axis=1,
            )
        one = triangles[:, :, 1] - triangles[:, :, 0]
        two = triangles[:, :, 2] - triangles[:, :, 0]
        area = 0.5 * np.abs(one[..., 0] * two[..., 1] - one[..., 1] * two[..., 0])
        points = np.einsum("qa,bpad->bpqd", barycentric, triangles)
        density = source_current_density(case, points)
        result[first:stop] = np.sum(
            area[:, :, None] * weight[None, None, :] * density, axis=(1, 2)
        )
    return result


def adaptive_sector_currents(
    case,
    axis: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    angular_order: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Double radial panels until sector currents converge collectively."""
    angular_node, angular_weight = leggauss(angular_order)
    angular_fraction = 0.5 * (angular_node + 1.0)
    angular_weight = 0.5 * angular_weight
    radial_node, radial_weight = leggauss(8)
    previous = None
    history = []
    for panels in (2, 4, 8, 16, 32, 64, 128, 256):
        edges = np.linspace(0.0, 1.0, panels + 1)
        centre = 0.5 * (edges[:-1] + edges[1:])
        half_width = 0.5 * np.diff(edges)
        radius = (centre[:, None] + half_width[:, None] * radial_node[None, :]).ravel()
        radial_weights = (half_width[:, None] * radial_weight[None, :]).ravel()
        current = np.empty(len(start))
        for first_slot in range(0, len(start), 128):
            stop = min(first_slot + 128, len(start))
            batch_start = start[first_slot:stop]
            batch_end = end[first_slot:stop]
            ray = (
                batch_start[:, None, :]
                + angular_fraction[None, :, None]
                * (batch_end - batch_start)[:, None, :]
                - axis
            )
            first = batch_start - axis
            second = batch_end - axis
            jacobian = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
            points = axis + radius[None, :, None, None] * ray[:, None, :, :]
            density = source_current_density(case, points)
            angular = np.sum(density * angular_weight[None, None, :], axis=2)
            current[first_slot:stop] = jacobian * np.sum(
                angular * radius[None, :] * radial_weights[None, :], axis=1
            )
        total = math.fsum(current)
        signed_delta = None if previous is None else total - math.fsum(previous)
        l1_delta = (
            None if previous is None else float(np.sum(np.abs(current - previous)))
        )
        history.append(
            {
                "radial_panels": panels,
                "radial_nodes": panels * len(radial_node),
                "current_a": total,
                "signed_change_a": signed_delta,
                "sector_l1_change_a": l1_delta,
            }
        )
        if previous is not None and abs(signed_delta) <= 0.1 and l1_delta <= 5.0:
            return current, l1_delta, history
        previous = current
    return current, l1_delta, history


def sector_owner_indices(
    start: np.ndarray, end: np.ndarray, centres: np.ndarray
) -> np.ndarray:
    midpoint = 0.5 * (start + end)
    distance_squared = np.sum((midpoint[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    return np.argmin(distance_squared, axis=1)


def make_figure(
    path: Path,
    material_vertices: list[np.ndarray],
    core,
    cell_delta: np.ndarray,
    contributing: np.ndarray,
    sector_bias: np.ndarray,
    saddle: np.ndarray,
    saddle_cell: int,
    neighbours: list[int],
    support: np.ndarray,
    crossings: list[dict[str, object]],
) -> None:
    figure, (fan_axis, cell_axis, zoom) = plt.subplots(1, 3, figsize=(16.0, 5.4))
    core_vertices = np.asarray(core.exterior.coords, dtype=float)[:-1, :2]
    sector_end = np.roll(core_vertices, -1, axis=0)
    segments = np.stack([core_vertices, sector_end], axis=1)
    fan_limit = max(float(np.max(np.abs(sector_bias))), 1.0e-6)
    fan_collection = LineCollection(
        segments,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-fan_limit, vcenter=0.0, vmax=fan_limit),
        linewidth=2.2,
    )
    fan_collection.set_array(sector_bias)
    fan_axis.add_collection(fan_collection)
    fan_axis.scatter(*saddle, marker="x", color="black", s=45, zorder=5)
    fan_axis.set_aspect("equal")
    fan_axis.autoscale()
    fan_axis.set_xlabel("R [m]")
    fan_axis.set_ylabel("Z [m]")
    fan_axis.set_title("Fixed minus adaptive fan-sector current")
    figure.colorbar(
        fan_collection,
        ax=fan_axis,
        shrink=0.78,
        label="sector quadrature bias [A]",
    )

    patches = [
        PolygonPatch(material_vertices[cell], closed=True) for cell in contributing
    ]
    values = cell_delta[contributing]
    limit = max(float(np.max(np.abs(values))), 1.0)
    collection = PatchCollection(
        patches,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        edgecolor="0.75",
        linewidth=0.35,
    )
    collection.set_array(values)
    cell_axis.add_collection(collection)
    closed_core = np.vstack([core_vertices, core_vertices[0]])
    cell_axis.plot(closed_core[:, 0], closed_core[:, 1], color="black", linewidth=1.0)
    cell_axis.scatter(*saddle, marker="x", color="black", s=45, zorder=6)
    cell_axis.set_aspect("equal")
    cell_axis.autoscale()
    cell_axis.set_xlabel("R [m]")
    cell_axis.set_ylabel("Z [m]")
    cell_axis.set_title("Corrected core minus per-cell target")
    figure.colorbar(
        collection,
        ax=cell_axis,
        shrink=0.78,
        label="cell residual [A]",
    )

    cluster = [saddle_cell, *neighbours]
    for cell in cluster:
        points = np.vstack([material_vertices[cell], material_vertices[cell][0]])
        zoom.plot(
            points[:, 0],
            points[:, 1],
            color="#2f6f9f" if cell != saddle_cell else "#b22222",
            linewidth=1.2 if cell != saddle_cell else 2.2,
        )
    support_closed = np.vstack([support, support[0]])
    zoom.plot(
        support_closed[:, 0],
        support_closed[:, 1],
        color="#1b9e77",
        linewidth=2.0,
        label="production support",
    )
    zoom.plot(closed_core[:, 0], closed_core[:, 1], color="black", linewidth=1.4)
    crossing_points = np.asarray([item["coordinate_m"] for item in crossings])
    if len(crossing_points):
        zoom.scatter(
            crossing_points[:, 0],
            crossing_points[:, 1],
            s=26,
            facecolor="white",
            edgecolor="black",
            zorder=5,
            label="analytic edge crossing",
        )
    zoom.scatter(
        *saddle, marker="x", color="black", s=55, zorder=6, label="Newton saddle"
    )
    saddle_vertices = material_vertices[saddle_cell]
    margin = 0.12
    zoom.set_xlim(
        np.min(saddle_vertices[:, 0]) - margin, np.max(saddle_vertices[:, 0]) + margin
    )
    zoom.set_ylim(
        np.min(saddle_vertices[:, 1]) - margin, np.max(saddle_vertices[:, 1]) + margin
    )
    zoom.set_aspect("equal")
    zoom.set_xlabel("R [m]")
    zoom.set_ylabel("Z [m]")
    zoom.set_title(f"Saddle cell {saddle_cell}: analytic contour and clip chord")
    zoom.legend(loc="best", fontsize=8, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    case = load_reference()
    fixture = load_npz(FIXTURE)
    localization = load_npz(LOCALIZATION)
    production = load_npz(PRODUCTION)
    saddle, saddle_diagnostics = solve_flux_saddle(case)
    _centres, material_vertices, _geometry, _radius = build_geometry(case, -500)
    material = [ShapelyPolygon(vertices) for vertices in material_vertices]
    core, fixed_core_current, root_diagnostics, refinement = refined_analytic_core(
        case, args.arc_points, saddle
    )
    count = fixture["support_vertex_count"]
    support_vertices = fixture["support_vertices"]
    centres = fixture["consistent_centres"]
    lower_leg = np.asarray(production["lower_leg_mask"], dtype=bool)
    target_current = np.asarray(localization["analytic_m0"], dtype=float).copy()
    target_current[lower_leg] = 0.0

    core_vertices = np.asarray(core.exterior.coords, dtype=float)[:-1, :2]
    sector_start = core_vertices
    sector_end = np.roll(core_vertices, -1, axis=0)
    fixed_sector = fixed_sector_currents(
        case, np.asarray(case.axis), sector_start, sector_end
    )
    throat_slots = np.asarray(
        [
            item["angular_slot"]
            for item in root_diagnostics["saddle_throat_cut_receipts"]
        ],
        dtype=int,
    )
    throat_adaptive_coarse, throat_coarse_error, throat_coarse_history = (
        adaptive_sector_currents(
            case,
            np.asarray(case.axis),
            sector_start[throat_slots],
            sector_end[throat_slots],
            angular_order=8,
        )
    )
    throat_adaptive, throat_error, throat_history = adaptive_sector_currents(
        case,
        np.asarray(case.axis),
        sector_start[throat_slots],
        sector_end[throat_slots],
        angular_order=16,
    )
    print("throat-sector adaptive check complete", flush=True)
    adaptive_sector, adaptive_radial_error, adaptive_history = adaptive_sector_currents(
        case,
        np.asarray(case.axis),
        sector_start,
        sector_end,
        angular_order=8,
    )
    print("all-sector adaptive radial check complete", flush=True)
    fixed_sector_total = math.fsum(fixed_sector)
    corrected_core_current = math.fsum(adaptive_sector)
    sector_bias = fixed_sector - adaptive_sector
    fan_bias = math.fsum(sector_bias)
    throat_fixed_total = math.fsum(fixed_sector[throat_slots])
    throat_adaptive_coarse_total = math.fsum(throat_adaptive_coarse)
    throat_adaptive_total = math.fsum(throat_adaptive)
    throat_bias = throat_fixed_total - throat_adaptive_total

    ray_cell_current = np.zeros(len(material))
    core_piece = []
    for cell, polygon in enumerate(material):
        piece = polygon.intersection(core)
        core_piece.append(piece)
        ray_cell_current[cell] = polygon_current(case, piece)
    ray_cells_total = math.fsum(ray_cell_current)
    target_total = math.fsum(target_current)
    original_gap = fixed_core_current - target_total
    corrected_gap = corrected_core_current - target_total
    cell_delta = ray_cell_current - target_current
    contributing = np.flatnonzero(np.abs(cell_delta) > 1.0e-6)
    saddle_matches = [
        cell for cell, polygon in enumerate(material) if polygon.covers(Point(saddle))
    ]
    if len(saddle_matches) != 1:
        raise AssertionError(f"expected one saddle cell, found {saddle_matches}")
    saddle_cell = saddle_matches[0]
    neighbours = edge_neighbours(material, saddle_cell)
    cluster = {saddle_cell, *neighbours}
    cell_records = []
    for cell in contributing:
        piece = core_piece[cell]
        location = piece.centroid if not piece.is_empty else material[cell].centroid
        support = (
            ShapelyPolygon(support_vertices[cell, : count[cell]])
            if count[cell] >= 3
            else None
        )
        sampled_geometry = piece if support is None else unary_union([piece, support])
        cell_records.append(
            {
                "kind": "support_cell",
                "cell": int(cell),
                "location_m": [float(location.x), float(location.y)],
                "psi_norm_range": psi_range(case, sampled_geometry),
                "minimum_distance_to_saddle_m": float(
                    material[cell].distance(Point(saddle))
                ),
                "centre_distance_to_saddle_m": float(
                    np.linalg.norm(centres[cell] - saddle)
                ),
                "qualification": (
                    "topology_zero_lower_leg" if lower_leg[cell] else "core_qualified"
                ),
                "topology_target_current_a": float(target_current[cell]),
                "ray_core_cell_current_a": float(ray_cell_current[cell]),
                "current_gap_contribution_a": float(cell_delta[cell]),
                "production_attributed_current_a": float(
                    production["attributed_m0"][cell]
                ),
                "support_vertex_count": int(count[cell]),
                "in_saddle_edge_cluster": bool(cell in cluster),
            }
        )
    sector_owner = sector_owner_indices(sector_start, sector_end, centres)
    sector_midpoint = 0.5 * (sector_start + sector_end)
    fan_records = []
    for cell in np.unique(sector_owner):
        selection = sector_owner == cell
        bias = math.fsum(sector_bias[selection])
        if abs(bias) <= 1.0e-9:
            continue
        weights = np.abs(sector_bias[selection])
        location = np.average(
            sector_midpoint[selection],
            axis=0,
            weights=weights if np.sum(weights) > 0.0 else None,
        )
        boundary_points = np.concatenate(
            [sector_start[selection], sector_end[selection]], axis=0
        )
        boundary_psi = np.asarray(
            normalised_source_flux(case, boundary_points[:, 0], boundary_points[:, 1])
        )
        fan_records.append(
            {
                "kind": "boundary_fan_sector_group",
                "nearest_cell": int(cell),
                "sector_count": int(np.count_nonzero(selection)),
                "location_m": location.tolist(),
                "psi_norm_range": [0.0, float(np.max(boundary_psi))],
                "minimum_distance_to_saddle_m": float(
                    np.min(np.linalg.norm(sector_midpoint[selection] - saddle, axis=1))
                ),
                "fixed_rule_current_a": math.fsum(fixed_sector[selection]),
                "adaptive_current_a": math.fsum(adaptive_sector[selection]),
                "current_gap_contribution_a": bias,
                "in_saddle_edge_cluster": bool(cell in cluster),
            }
        )
    cluster_fan_bias = math.fsum(
        record["current_gap_contribution_a"]
        for record in fan_records
        if record["in_saddle_edge_cluster"]
    )
    cluster_cell_gap = math.fsum(cell_delta[list(cluster)])
    cluster_gap = cluster_fan_bias + cluster_cell_gap
    ring_gap = original_gap - cluster_gap
    all_contributions = [*fan_records, *cell_records]
    cluster_l1 = math.fsum(
        abs(record["current_gap_contribution_a"])
        for record in all_contributions
        if record["in_saddle_edge_cluster"]
    )
    total_l1 = math.fsum(
        abs(record["current_gap_contribution_a"]) for record in all_contributions
    )
    saddle_material = material[saddle_cell]
    saddle_support_array = support_vertices[saddle_cell, : count[saddle_cell]]
    saddle_support = ShapelyPolygon(saddle_support_array)
    crossings = edge_crossings(case, material_vertices[saddle_cell])
    chords = clip_chords(saddle_material, saddle_support_array)
    chord_endpoints = np.asarray([point for chord in chords for point in chord])
    crossing_points = np.asarray([item["coordinate_m"] for item in crossings])
    chord_crossing_distance = (
        float(
            max(
                np.min(np.linalg.norm(crossing_points - endpoint, axis=1))
                for endpoint in chord_endpoints
            )
        )
        if len(chord_endpoints) and len(crossing_points)
        else None
    )
    centre_psi = np.asarray(normalised_source_flux(case, centres[:, 0], centres[:, 1]))
    private_flux = lower_leg & (centre_psi <= 1.0)
    open_leg = lower_leg & (centre_psi > 1.0)
    exterior = (
        np.asarray([polygon.intersection(core).area == 0.0 for polygon in material])
        & (centre_psi > 1.0)
        & (count < 3)
    )
    signed_zero_checks = {}
    for name, mask in {
        "psi_norm_greater_than_one_exterior_cells": exterior,
        "private_flux_cells": private_flux,
        "open_leg_cells": open_leg,
        "all_topology_zero_lower_leg_cells": lower_leg,
    }.items():
        signed_zero_checks[name] = {
            "cells": int(np.count_nonzero(mask)),
            "production_support_current_a": float(
                np.sum(production["attributed_m0"][mask])
            ),
            "per_cell_oracle_current_a": float(np.sum(target_current[mask])),
            "ray_traced_core_current_a": float(np.sum(ray_cell_current[mask])),
        }
    fixed_reproduction_residual = fixed_sector_total - fixed_core_current
    adaptive_cell_closure = corrected_core_current - ray_cells_total
    additive_closure = original_gap - (fan_bias + corrected_gap)
    cluster_closure = original_gap - (cluster_gap + ring_gap)
    angular_order_delta = throat_adaptive_total - throat_adaptive_coarse_total
    if abs(fixed_core_current - EXPECTED_CORE_CURRENT_A) > 1.0e-6:
        raise AssertionError("ray-traced core current changed")
    if abs(target_total - EXPECTED_TARGET_CURRENT_A) > 1.0e-6:
        raise AssertionError("topology-qualified target current changed")
    if abs(fixed_reproduction_residual) > 1.0e-5:
        raise AssertionError("per-sector fixed rule does not reproduce the bank")
    if abs(angular_order_delta) > 0.5:
        raise AssertionError("adaptive angular-order check does not converge")
    if abs(adaptive_cell_closure) > 5.0:
        raise AssertionError(
            f"adaptive fan and cellwise core differ by {adaptive_cell_closure:.6f} A"
        )
    if abs(additive_closure) > 1.0e-8 or abs(cluster_closure) > 1.0e-8:
        raise AssertionError("fan-bias decomposition does not close")
    for name, values in signed_zero_checks.items():
        if any(
            abs(values[key]) > 1.0e-8 for key in values if key.endswith("_current_a")
        ):
            raise AssertionError(f"nonzero forbidden-territory current in {name}")
    make_figure(
        args.figure,
        material_vertices,
        core,
        cell_delta,
        contributing,
        sector_bias,
        saddle,
        saddle_cell,
        neighbours,
        saddle_support_array,
        crossings,
    )
    result = {
        "totals": {
            "superseded_fixed_fan_core_current_a": fixed_core_current,
            "corrected_adaptive_fan_core_current_a": corrected_core_current,
            "topology_qualified_per_cell_target_current_a": target_total,
            "original_signed_fixed_fan_minus_target_gap_a": original_gap,
            "fixed_minus_adaptive_fan_bias_a": fan_bias,
            "corrected_core_minus_target_gap_a": corrected_gap,
            "cellwise_core_minus_target_a": math.fsum(cell_delta),
            "adaptive_fan_minus_cellwise_core_a": adaptive_cell_closure,
            "additive_closure_residual_a": additive_closure,
        },
        "fan_quadrature_diagnosis": {
            "verdict": (
                "The banked ray-core total is superseded: angular refinement "
                "held the radial rule fixed, while adaptive radial integration "
                "removes the 3.2 kA bias and agrees with cellwise integration."
            ),
            "fixed_rule": "four uniform triangle subdivisions and a four-point rule",
            "adaptive_rule": (
                "Eight-point Gauss-Legendre radial panels doubled through 256 "
                "panels (2048 radial nodes), with the terminal total checked "
                "against independent cellwise integration; the 14 throat sectors "
                "are cross-checked at angular order 16"
            ),
            "fixed_sector_total_a": fixed_sector_total,
            "fixed_total_reproduction_residual_a": fixed_reproduction_residual,
            "adaptive_full_fan_angular_order_eight_total_a": corrected_core_current,
            "throat_adaptive_angular_order_eight_total_a": (
                throat_adaptive_coarse_total
            ),
            "throat_adaptive_angular_order_sixteen_total_a": (throat_adaptive_total),
            "throat_angular_order_delta_a": angular_order_delta,
            "terminal_sector_l1_refinement_change_a": adaptive_radial_error,
            "adaptive_radial_convergence": adaptive_history,
            "throat_sector_count": len(throat_slots),
            "throat_slots": throat_slots.tolist(),
            "throat_fixed_current_a": throat_fixed_total,
            "throat_adaptive_current_a": throat_adaptive_total,
            "throat_fixed_minus_adaptive_bias_a": throat_bias,
            "throat_fraction_of_full_fan_bias": throat_bias / fan_bias,
            "throat_order_eight_terminal_l1_change_a": throat_coarse_error,
            "throat_order_sixteen_terminal_l1_change_a": throat_error,
            "throat_angular_order_eight_radial_convergence": (throat_coarse_history),
            "throat_angular_order_sixteen_radial_convergence": throat_history,
            "coverage_gate_recommendation_affected": False,
            "coverage_gate_note": (
                "The coverage gate compares production supports with the "
                "topology-qualified per-cell target and never uses the fan total."
            ),
            "boundary_sector_groups": fan_records,
        },
        "saddle": {
            "coordinates_m": saddle.tolist(),
            "diagnostics": saddle_diagnostics,
            "source_psi_norm": float(normalised_source_flux(case, *saddle)),
            "cell": saddle_cell,
            "edge_sharing_neighbours": neighbours,
            "separatrix_edge_crossing_count": len(crossings),
            "separatrix_edge_crossings": crossings,
            "production_clip_chords_m": chords,
            "clip_endpoint_max_distance_to_analytic_crossing_m": (
                chord_crossing_distance
            ),
            "qualification": (
                "topology_zero_lower_leg"
                if lower_leg[saddle_cell]
                else "core_qualified_boundary_cell"
            ),
            "support_vertex_count": int(count[saddle_cell]),
            "support_area_fraction_of_hex": float(
                saddle_support.area / saddle_material.area
            ),
            "support_covers_saddle": bool(saddle_support.covers(Point(saddle))),
            "production_attributed_current_a": float(
                production["attributed_m0"][saddle_cell]
            ),
            "per_cell_oracle_current_a": float(target_current[saddle_cell]),
            "analytic_core_side_wedge_current_a": float(ray_cell_current[saddle_cell]),
        },
        "gap_localization": {
            "contributing_support_cells": len(cell_records),
            "saddle_cell_plus_edge_neighbours_signed_gap_a": cluster_gap,
            "remaining_boundary_ring_signed_gap_a": ring_gap,
            "saddle_cluster_fan_bias_a": cluster_fan_bias,
            "saddle_cluster_corrected_cell_gap_a": cluster_cell_gap,
            "saddle_cluster_fraction_of_original_signed_gap": (
                cluster_gap / original_gap
            ),
            "remaining_ring_fraction_of_original_signed_gap": ring_gap / original_gap,
            "saddle_cluster_fraction_of_contribution_l1": cluster_l1 / total_l1,
            "contribution_l1_a": total_l1,
            "support_cells": cell_records,
        },
        "forbidden_territory_signed_integrals": {
            "interpretation": (
                "Each construction is integrated only on its carried core domain. "
                "Exterior cells have no ray-core intersection; production and the "
                "per-cell oracle carry exact zeros on private-flux and lower-leg "
                "labels."
            ),
            "outside_each_construction_domain_a": {
                "production_support": 0.0,
                "per_cell_oracle": 0.0,
                "corrected_ray_core": 0.0,
            },
            **signed_zero_checks,
        },
        "oracle": {
            "arc_points": root_diagnostics["requested_angular_points"],
            "root_diagnostics": root_diagnostics,
            "refinement_pairs": refinement,
        },
        "artifacts": {
            "figure": str(args.figure.resolve()),
            "results": str(args.results.resolve()),
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(f"original fixed-fan gap: {original_gap:.9f} A")
    print(f"fan quadrature bias: {fan_bias:.9f} A")
    print(f"corrected core-target gap: {corrected_gap:.9f} A")
    print(
        f"throat-sector bias: {throat_bias:.9f} A "
        f"({throat_bias / fan_bias:.6%} of fan bias)"
    )
    print(
        "saddle cluster: {:.9f} A ({:.6%}); remaining ring: {:.9f} A ({:.6%})".format(
            cluster_gap,
            cluster_gap / original_gap,
            ring_gap,
            ring_gap / original_gap,
        )
    )
    print(
        f"saddle cell: {saddle_cell}; crossings: {len(crossings)}; "
        f"chords: {len(chords)}"
    )
    print(f"adaptive fan-cell closure: {adaptive_cell_closure:.9e} A")
    print(f"results: {args.results}")
    print(f"figure: {args.figure}")


if __name__ == "__main__":
    main()
