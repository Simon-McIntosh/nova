"""Render containment, clipping, and exact-solution evidence from retained cases."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np
from shapely import LineString, Point
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from nova.media.ink import DEFAULT_INK
from nova.media.layout import poloidal_view
from nova.media.poloidal import (
    contour_levels,
    draw_boundary,
    draw_flux_contours,
    draw_nulls,
    draw_plasma_cells,
    draw_wall,
)
from nova.media.sources.frame import inside_wall_units
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
CERTIFICATE_SOURCE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/triage/"
    "rebank-certificate-and-bank/artifacts/job-1267132/measurement-summary.json"
)
CLIP_SOURCE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/null-identification/"
    "clipped-moment-repair-logs/two-rung-physical-moment-acceptance.json"
)
CASES = (
    ("weak-rotation-reactor-static", -110),
    ("weak-rotation-reactor-static", -300),
    ("weak-rotation-reactor-static", -500),
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -110),
    ("moderate-rotation-conventional-static", -300),
)
COLD_START_TARGET_CURRENT_A = 10301.073531668495


def _slug(case_name: str, requested_cells: int) -> str:
    return (
        f"{case_name}-{'reduced' if requested_cells == -110 else abs(requested_cells)}"
    )


def _write_receipt(payload: dict[str, Any]) -> None:
    path = OUTPUT / "evidence-receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _existing_receipt() -> dict[str, Any]:
    path = OUTPUT / "evidence-receipt.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _extent(
    *groups: np.ndarray, padding: float = 0.08
) -> tuple[float, float, float, float]:
    points = np.vstack(
        [np.asarray(group, dtype=float).reshape(-1, 2) for group in groups]
    )
    points = points[np.all(np.isfinite(points), axis=1)]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 1.0)
    return (
        float(lower[0] - padding * span[0]),
        float(upper[0] + padding * span[0]),
        float(lower[1] - padding * span[1]),
        float(upper[1] + padding * span[1]),
    )


def _finite_points(values: Any) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points[:, :2]
    return points[np.all(np.isfinite(points), axis=1)]


def _structured_flux(
    operator: Any, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(operator.grid.coordinate, dtype=np.float64)
    values = np.asarray(state, dtype=np.float64)[: len(points)]
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(values)
    points = points[finite]
    values = values[finite]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    radial = np.linspace(lower[0], upper[0], 181)
    vertical = np.linspace(lower[1], upper[1], 181)
    mesh_radius, mesh_height = np.meshgrid(radial, vertical, indexing="xy")
    query = np.column_stack((mesh_radius.ravel(), mesh_height.ravel()))
    flux = LinearNDInterpolator(points, values, fill_value=np.nan)(query)
    if not np.any(np.isfinite(flux)):
        flux = NearestNDInterpolator(points, values)(query)
    flux = np.asarray(flux, dtype=np.float64).reshape(mesh_radius.shape)
    return radial, vertical, flux


def _analytic_grid(
    case_name: str, exact: Any, radius: np.ndarray, height: np.ndarray
) -> np.ndarray:
    mesh_radius, mesh_height = np.meshgrid(radius, height, indexing="xy")
    coordinates = np.column_stack((mesh_radius.ravel(), mesh_height.ravel()))
    return certificate._exact_state(case_name, exact, coordinates).reshape(
        mesh_height.shape
    )


def _shared_levels(
    solved: np.ndarray,
    analytic: np.ndarray,
    *boundary_values: float | None,
) -> np.ndarray:
    finite = np.concatenate(
        (
            np.asarray(solved, dtype=float).ravel(),
            np.asarray(analytic, dtype=float).ravel(),
        )
    )
    finite = finite[np.isfinite(finite)]
    values = contour_levels(finite[None, :], count=15)
    extras = np.asarray(
        [value for value in boundary_values if value is not None], dtype=float
    )
    extras = extras[np.isfinite(extras)]
    return np.unique(np.concatenate((values, extras)))


def _build_certificate_case(case_name: str, requested_cells: int) -> dict[str, Any]:
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, analytic_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, analytic_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, current_centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        current_centroid,
        current_receipt,
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"evidence:{case_name}:{requested_cells}",
    )
    started = perf_counter()
    solve_receipt = profile.solve(request)
    state = np.asarray(solve_receipt.equilibrium.flux, dtype=np.float64)
    jax.block_until_ready(state)
    elapsed = perf_counter() - started
    radius, height, solved_grid = _structured_flux(operator, state)
    analytic_grid = _analytic_grid(case_name, exact, radius, height)
    topology = certificate._topology(operator, state)
    analytic_topology = certificate._topology(operator, analytic_state)
    boundary = certificate._boundary(case_name, exact)
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "machine": machine,
        "operator": operator,
        "exact": exact,
        "state": state,
        "analytic_state": analytic_state,
        "radius": radius,
        "height": height,
        "solved_grid": solved_grid,
        "analytic_grid": analytic_grid,
        "topology": topology,
        "analytic_topology": analytic_topology,
        "boundary": boundary,
        "terminal_residual": float(solve_receipt.equilibrium.fixed_point.residual),
        "converged": bool(solve_receipt.equilibrium.fixed_point.converged),
        "solve_seconds": elapsed,
        "wall": np.asarray(machine.wall_node, dtype=np.float64),
    }


def _topology_point(record: dict[str, Any], key: str) -> np.ndarray | None:
    value = record.get(key)
    if value is None:
        return None
    point = np.asarray(value, dtype=float)
    return point if point.shape == (2,) and np.all(np.isfinite(point)) else None


def _draw_certificate(record: dict[str, Any]) -> dict[str, Any]:
    radius = record["radius"]
    height = record["height"]
    solved = record["solved_grid"]
    analytic = record["analytic_grid"]
    topology = record["topology"]
    analytic_topology = record["analytic_topology"]
    boundary = record["boundary"]
    wall = record["wall"]
    levels = _shared_levels(
        solved,
        analytic,
        topology.get("boundary_flux_wb"),
        analytic_topology.get("boundary_flux_wb"),
    )
    view = poloidal_view(_extent(wall, boundary), height=5.8)
    axes = view.poloidal
    draw_flux_contours(axes, radius, height, analytic, levels, color="#3366cc")
    draw_flux_contours(axes, radius, height, solved, levels, color="#cc7722")
    root_boundary = topology.get("boundary_flux_wb")
    if root_boundary is not None and np.isfinite(root_boundary):
        draw_flux_contours(
            axes,
            radius,
            height,
            solved,
            [float(root_boundary)],
            color="#cc7722",
            linewidth=1.4,
        )
    draw_boundary(axes, boundary[:, 0], boundary[:, 1], color="#cc0000")
    draw_wall(axes, wall[:, 0], wall[:, 1])
    marker_style = DEFAULT_INK.variant(axis_marker="^", axis_color="#cc7722")
    reference_style = replace(marker_style, axis_color="#666666")
    root_axis = _topology_point(topology, "axis_rz_m")
    analytic_axis = _topology_point(analytic_topology, "axis_rz_m")
    draw_nulls(
        axes,
        magnetic_axis=root_axis,
        x_points=_topology_point(topology, "x_point_rz_m"),
        style=marker_style,
        contain=wall,
    )
    draw_nulls(
        axes,
        magnetic_axis=analytic_axis,
        x_points=_topology_point(analytic_topology, "x_point_rz_m"),
        style=reference_style,
        contain=wall,
    )
    axes.plot([], [], color="#3366cc", lw=1.0, label="analytic flux")
    axes.plot([], [], color="#cc7722", lw=1.0, label="solved terminal flux")
    axes.plot([], [], color="#cc0000", lw=1.4, label="analytic boundary")
    axes.legend(loc="upper left", fontsize="xx-small", frameon=True)
    title = f"{record['case']} · {record['requested_cells']} cells"
    axes.set_title(title, fontsize=8)
    path = (
        OUTPUT / f"certificate-{_slug(record['case'], record['requested_cells'])}.png"
    )
    view.figure.savefig(path, dpi=180)
    return {
        "path": str(path.relative_to(ROOT)),
        "project_src": "/nova/figures/null-identification-authority/mechanism-evidence/"
        + path.name,
        "levels_wb": levels.tolist(),
        "axis_error_m": (
            None
            if root_axis is None or analytic_axis is None
            else float(np.linalg.norm(root_axis - analytic_axis))
        ),
        "terminal_residual": record["terminal_residual"],
        "converged": record["converged"],
        "solve_seconds": record["solve_seconds"],
        "root_boundary_flux_wb": topology.get("boundary_flux_wb"),
        "analytic_boundary_flux_wb": analytic_topology.get("boundary_flux_wb"),
    }


def _error_locality(record: dict[str, Any]) -> dict[str, Any]:
    machine = record["machine"]
    root = np.asarray(record["state"], dtype=np.float64)[: len(machine.node)]
    analytic = np.asarray(record["analytic_state"], dtype=np.float64)[
        : len(machine.node)
    ]
    points = np.asarray(machine.node, dtype=np.float64)
    error = root - analytic
    finite = np.isfinite(error) & np.all(np.isfinite(points), axis=1)
    magnitude = np.abs(error[finite])
    coordinates = points[finite]
    squared = magnitude**2
    total = float(np.sum(squared))
    ranked = np.sort(squared)[::-1]
    energy_count = int(np.searchsorted(np.cumsum(ranked), 0.9 * total) + 1)
    high_cutoff = float(np.quantile(magnitude, 0.95))
    high = magnitude >= high_cutoff
    high_coordinates = coordinates[high]
    weighted_centre = np.average(coordinates, axis=0, weights=squared)
    boundary = LineString(record["boundary"])
    wall = LineString(record["wall"])
    analytic_axis = _topology_point(record["analytic_topology"], "axis_rz_m")
    outboard_window = (
        (coordinates[:, 0] >= 7.25)
        & (coordinates[:, 0] <= 7.75)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= 0.5)
    )
    window_energy = (
        float(np.sum(squared[outboard_window]) / total) if total > 0.0 else float("nan")
    )
    pitch = float(np.sqrt(np.median(np.asarray(machine.area, dtype=float))))
    boundary_distance = float(boundary.distance(Point(weighted_centre)))
    wall_distance = float(wall.distance(Point(weighted_centre)))
    axis_distance = (
        None
        if analytic_axis is None
        else float(np.linalg.norm(weighted_centre - analytic_axis))
    )
    return {
        "node_count": int(len(coordinates)),
        "top_five_percent_node_count": int(np.count_nonzero(high)),
        "top_five_percent_domain_fraction": float(np.mean(high)),
        "top_five_percent_energy_fraction": (
            float(np.sum(squared[high]) / total) if total > 0.0 else None
        ),
        "minimum_node_fraction_for_ninety_percent_squared_error": (
            float(energy_count / len(coordinates))
        ),
        "weighted_error_centre_rz_m": weighted_centre.tolist(),
        "top_five_percent_bounds_rz_m": [
            float(np.min(high_coordinates[:, 0])),
            float(np.max(high_coordinates[:, 0])),
            float(np.min(high_coordinates[:, 1])),
            float(np.max(high_coordinates[:, 1])),
        ],
        "outboard_midplane_window_domain_fraction": float(np.mean(outboard_window)),
        "outboard_midplane_window_squared_error_fraction": window_energy,
        "characteristic_pitch_m": pitch,
        "weighted_centre_boundary_distance_m": boundary_distance,
        "weighted_centre_wall_distance_m": wall_distance,
        "weighted_centre_axis_distance_m": axis_distance,
        "boundary_contact_within_one_pitch": boundary_distance <= pitch,
        "limiter_contact_within_one_pitch": wall_distance <= pitch,
        "axis_anchor_within_one_pitch": (
            None if axis_distance is None else axis_distance <= pitch
        ),
    }


def _draw_error_locality(record: dict[str, Any], locality: dict[str, Any]) -> str:
    radius = record["radius"]
    height = record["height"]
    solved = record["solved_grid"]
    analytic = record["analytic_grid"]
    wall = record["wall"]
    boundary = record["boundary"]
    levels = _shared_levels(solved, analytic)
    view = poloidal_view(_extent(wall, boundary), height=5.8)
    axes = view.poloidal
    draw_flux_contours(axes, radius, height, analytic, levels, color="#3366cc")
    draw_flux_contours(axes, radius, height, solved, levels, color="#cc7722")
    draw_boundary(axes, boundary[:, 0], boundary[:, 1])
    draw_wall(axes, wall[:, 0], wall[:, 1])
    machine = record["machine"]
    root = np.asarray(record["state"], dtype=np.float64)[: len(machine.node)]
    analytic_node = np.asarray(record["analytic_state"], dtype=np.float64)[
        : len(machine.node)
    ]
    error = np.abs(root - analytic_node)
    cutoff = float(np.quantile(error[np.isfinite(error)], 0.95))
    high = np.isfinite(error) & (error >= cutoff)
    axes.plot(
        machine.node[high, 0],
        machine.node[high, 1],
        marker="o",
        markersize=2.5,
        color="#7a3e00",
        linestyle="none",
        label="top 5% |psi error| nodes",
    )
    centre = np.asarray(locality["weighted_error_centre_rz_m"], dtype=float)
    axes.plot(
        centre[0],
        centre[1],
        marker="*",
        markersize=7,
        color="#7a3e00",
        linestyle="none",
        label="squared-error centre",
    )
    axes.legend(loc="upper left", fontsize="xx-small", frameon=True)
    axes.set_title("weak rotation · 500 cells · error support", fontsize=8)
    path = OUTPUT / "weak-rotation-500-error-locality.png"
    view.figure.savefig(path, dpi=180)
    return "/nova/figures/null-identification-authority/mechanism-evidence/" + path.name


def _build_diverted_terminal(requested_cells: int) -> dict[str, Any]:
    case_name = "diverted-jump-bearing"
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, analytic_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, analytic_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        target_current,
        centroid,
        current_receipt,
    )
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target_current,
        carrier_identity=f"evidence-clip:{requested_cells}",
    )
    started = perf_counter()
    receipt = profile.solve(request)
    state = np.asarray(receipt.equilibrium.flux, dtype=np.float64)
    jax.block_until_ready(state)
    return {
        "machine": machine,
        "operator": operator,
        "state": state,
        "target_current": float(target_current),
        "residual": float(receipt.equilibrium.fixed_point.residual),
        "refusals": int(receipt.equilibrium.fixed_point.topology_trial_refusals),
        "boundary": certificate._boundary(case_name, exact),
        "wall": np.asarray(machine.wall_node, dtype=np.float64),
        "solve_seconds": perf_counter() - started,
    }


def _draw_clipped_cells(requested_cells: int) -> dict[str, Any]:
    record = _build_diverted_terminal(requested_cells)
    operator = record["operator"]
    state = record["state"]
    partition = operator._support_partition(state)
    masks, topology, sample_psi_norm, _profile_support = partition
    physical_unscaled = operator._physical_partitioned_current_moments(partition)
    amplitude = operator.current_normalisation_amplitude(
        record["target_current"], jnp.sum(physical_unscaled.cell_current)
    )
    physical = operator.scaled_current_moments(physical_unscaled, amplitude)
    shared_psi_norm = (
        operator.shared_node_flux(state) - topology.axis_flux
    ) / topology.flux_span
    atomic_mesh = operator.moment_geometry.atomic_mesh
    direct_support = jax.lax.cond(
        jnp.all(jnp.isfinite(topology.x_point)),
        lambda x_point: atomic_mesh.traced_clip(
            1.0 - shared_psi_norm, saddle_vertex=x_point
        ),
        lambda _x_point: atomic_mesh.traced_clip(1.0 - shared_psi_norm),
        topology.x_point,
    )
    direct_unscaled = operator.support_current_moments(
        operator.source.core,
        masks.psi_norm,
        sample_psi_norm,
        direct_support,
    )
    direct = operator.scaled_current_moments(direct_unscaled, amplitude)
    area_fraction = np.asarray(
        direct_support.area / direct_support.full_area, dtype=np.float64
    )
    cut = (area_fraction > 1.0e-12) & (area_fraction < 1.0 - 1.0e-12)
    indices = np.flatnonzero(cut)
    raw_polygons = [
        np.asarray(operator.moment_geometry.polygons[index]) for index in indices
    ]
    vertices = np.asarray(direct_support.support_vertices, dtype=np.float64)
    counts = np.asarray(direct_support.vertex_count, dtype=int)
    clipped_polygons = [vertices[index, : counts[index]] for index in indices]
    centres = np.asarray(
        operator.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    physical_current = np.asarray(physical.cell_current, dtype=np.float64)
    direct_current = np.asarray(direct.cell_current, dtype=np.float64)
    physical_first = np.column_stack(
        (
            np.asarray(physical.radial_moment, dtype=np.float64),
            np.asarray(physical.vertical_moment, dtype=np.float64),
        )
    )
    direct_first = np.column_stack(
        (
            np.asarray(direct.radial_moment, dtype=np.float64),
            np.asarray(direct.vertical_moment, dtype=np.float64),
        )
    )
    physical_centroid = (
        centres[indices] + physical_first[indices] / physical_current[indices, None]
    )
    direct_centroid = (
        centres[indices] + direct_first[indices] / direct_current[indices, None]
    )
    view = poloidal_view(
        _extent(*raw_polygons, record["boundary"], padding=0.25), height=5.8
    )
    axes = view.poloidal
    draw_plasma_cells(
        axes,
        raw_polygons,
        facecolor="none",
        edgecolor="#666666",
        alpha=1.0,
    )
    draw_plasma_cells(
        axes,
        clipped_polygons,
        facecolor="#d7c3f0",
        edgecolor="#a98fd0",
        alpha=0.85,
    )
    draw_boundary(axes, record["boundary"][:, 0], record["boundary"][:, 1])
    draw_wall(axes, record["wall"][:, 0], record["wall"][:, 1])
    draw_nulls(
        axes,
        magnetic_axis=np.asarray(topology.axis, dtype=float),
        x_points=np.asarray(topology.x_point, dtype=float),
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#cc7722", xpoint_marker="X"
        ),
        contain=record["wall"],
    )
    axes.plot(
        physical_centroid[:, 0],
        physical_centroid[:, 1],
        marker="o",
        markersize=4,
        color="#3366cc",
        linestyle="none",
        label="physical first-moment centroid",
    )
    axes.plot(
        direct_centroid[:, 0],
        direct_centroid[:, 1],
        marker="x",
        markersize=6,
        markeredgewidth=1.2,
        color="#000000",
        linestyle="none",
        label="direct clipped centroid",
    )
    axes.legend(loc="upper left", fontsize="xx-small", frameon=True)
    axes.set_title(f"diverted exact field · {requested_cells} cells", fontsize=8)
    path = OUTPUT / f"clipped-cells-{abs(requested_cells)}.png"
    view.figure.savefig(path, dpi=180)
    differences = np.linalg.norm(physical_centroid - direct_centroid, axis=1)
    current_error = np.divide(
        np.abs(physical_current[indices] - direct_current[indices]),
        np.maximum(np.abs(physical_current[indices]), np.abs(direct_current[indices])),
        out=np.zeros(len(indices), dtype=float),
        where=np.maximum(
            np.abs(physical_current[indices]), np.abs(direct_current[indices])
        )
        > 0.0,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "project_src": "/nova/figures/null-identification-authority/mechanism-evidence/"
        + path.name,
        "boundary_cut_cells": int(len(indices)),
        "max_centroid_separation_m": float(np.max(differences)),
        "max_current_relative_error": float(np.max(current_error)),
        "terminal_residual": record["residual"],
        "topology_trial_refusals": record["refusals"],
        "solve_seconds": record["solve_seconds"],
    }


def _cold_start_data() -> dict[str, Any]:
    case_name = "diverted-jump-bearing"
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        -110,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, analytic_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, analytic_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    _target_current, centroid, current_receipt = (
        certificate._closed_form_current_target(
            case_name, source_case, operator, exact_physical
        )
    )
    seed, _requested_class, _seed_receipt = certificate._production_seed(
        profile,
        case_name,
        COLD_START_TARGET_CURRENT_A,
        centroid,
        current_receipt,
    )
    state = np.asarray(seed, dtype=np.float64)
    radius, height, flux = _structured_flux(operator, state)
    physical = state[: operator.physical_node_number]
    grid_flux, _wall_flux = operator._fixed_design_topology.split_flux_map(
        jnp.asarray(physical)
    )
    raw_o, raw_x = operator._fixed_design_topology.grid(grid_flux)
    masks, topology, connected, admitted = operator._fixed_design_read(
        jnp.asarray(physical)
    )
    jax.block_until_ready((raw_o, raw_x, admitted))
    return {
        "machine": machine,
        "wall": np.asarray(machine.wall_node, dtype=float),
        "radius": radius,
        "height": height,
        "flux": flux,
        "raw_o": _finite_points(raw_o),
        "raw_x": _finite_points(raw_x),
        "axis_admitted": bool(np.asarray(admitted)),
        "axis": np.asarray(topology.axis, dtype=float),
        "target_current_a": COLD_START_TARGET_CURRENT_A,
        "connected_cell_count": int(np.count_nonzero(np.asarray(connected))),
        "profile_participation_cell_count": int(
            np.count_nonzero(np.asarray(masks.profile_participation))
        ),
    }


def _draw_cold_start() -> dict[str, Any]:
    record = _cold_start_data()
    view = poloidal_view(_extent(record["wall"]), height=5.8)
    axes = view.poloidal
    levels = contour_levels(record["flux"], count=15)
    draw_flux_contours(axes, record["radius"], record["height"], record["flux"], levels)
    draw_wall(axes, record["wall"][:, 0], record["wall"][:, 1])
    draw_nulls(
        axes,
        magnetic_axis=record["axis"],
        x_points=None,
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color="#cc7722", xpoint_marker="X"
        ),
        contain=record["wall"],
    )
    o_points = record["raw_o"]
    x_points = record["raw_x"]
    if len(o_points):
        axes.plot(
            o_points[:, 0],
            o_points[:, 1],
            marker="+",
            markersize=6,
            color="#7a3e00",
            linestyle="none",
            label="raw O-candidate",
        )
    if len(x_points):
        inside = np.asarray(inside_wall_units(x_points, record["wall"]), dtype=bool)
        if np.any(inside):
            axes.plot(
                x_points[inside, 0],
                x_points[inside, 1],
                marker="+",
                markersize=6,
                color="#7a3e00",
                linestyle="none",
                label="raw in-wall X-candidate",
            )
        if np.any(~inside):
            axes.plot(
                x_points[~inside, 0],
                x_points[~inside, 1],
                marker="s",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="#7a3e00",
                linestyle="none",
                label="raw out-of-wall X-candidate",
            )
    axes.legend(loc="upper left", fontsize="xx-small", frameon=True)
    axes.set_title("diverted cold start · first iterate · 110 cells", fontsize=8)
    path = OUTPUT / "cold-start-stationary-candidates.png"
    view.figure.savefig(path, dpi=180)
    return {
        "path": str(path.relative_to(ROOT)),
        "project_src": "/nova/figures/null-identification-authority/mechanism-evidence/"
        + path.name,
        "axis_admitted": record["axis_admitted"],
        "raw_o_candidate_count": int(len(o_points)),
        "raw_x_candidate_count": int(len(x_points)),
        "connected_cell_count": record["connected_cell_count"],
        "profile_participation_cell_count": record["profile_participation_cell_count"],
        "contour_levels_wb": levels.tolist(),
    }


def _draw_legend() -> str:
    extent = (0.0, 5.0, 0.0, 2.0)
    view = poloidal_view(extent, height=2.2)
    axes = view.poloidal
    draw_wall(axes, [0.2, 4.8, 4.8, 0.2], [0.2, 0.2, 1.8, 1.8])
    style = DEFAULT_INK.variant(
        axis_marker="^", axis_color="#cc7722", xpoint_marker="X"
    )
    draw_nulls(
        axes,
        magnetic_axis=[0.8, 1.1],
        x_points=np.asarray([[1.9, 1.1]]),
        style=style,
        contain=np.asarray([[0.2, 0.2], [4.8, 0.2], [4.8, 1.8], [0.2, 1.8]]),
    )
    axes.plot(
        2.9,
        1.1,
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="#cc7722",
        linestyle="none",
    )
    axes.plot(
        3.9,
        1.1,
        marker="s",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="#cc7722",
        linestyle="none",
    )
    axes.plot(0.8, 0.62, marker="^", markersize=6, color="#cc7722", linestyle="none")
    axes.plot(1.9, 0.62, marker="X", markersize=6, color="#cc7722", linestyle="none")
    axes.plot(2.9, 0.62, marker="o", markersize=5, color="#3366cc", linestyle="none")
    axes.plot(3.9, 0.62, marker="x", markersize=6, color="#000000", linestyle="none")
    labels = (
        (0.8, "magnetic axis"),
        (1.9, "admitted saddle"),
        (2.9, "other in-wall X"),
        (3.9, "outside-wall X"),
    )
    for position, label in labels:
        axes.text(position, 1.42, label, ha="center", va="bottom", fontsize=6)
    axes.text(0.8, 0.27, "physical centroid", ha="center", va="bottom", fontsize=6)
    axes.text(1.9, 0.27, "direct centroid", ha="center", va="bottom", fontsize=6)
    axes.text(2.9, 0.27, "analytic contour", ha="center", va="bottom", fontsize=6)
    axes.text(3.9, 0.27, "solved contour", ha="center", va="bottom", fontsize=6)
    path = OUTPUT / "marker-legend.png"
    view.figure.savefig(path, dpi=180)
    return "/nova/figures/null-identification-authority/mechanism-evidence/" + path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "certificate", "clip", "cold"),
        default="all",
    )
    arguments = parser.parse_args()
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("jax_enable_x64 is false")
    if np.dtype(jnp.asarray(1.0).dtype) != np.dtype(np.float64):
        raise RuntimeError("JAX default dtype is not float64")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = _existing_receipt()
    payload.update(
        {
            "renderer": str(Path(__file__).relative_to(ROOT)),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_platform": jax.default_backend(),
            "certificate_source": str(CERTIFICATE_SOURCE),
            "clip_source": str(CLIP_SOURCE),
        }
    )
    payload.setdefault("certificate", [])
    payload.setdefault("clipped_cells", [])
    payload.setdefault("cold_start", None)
    payload.setdefault("error_locality", None)
    payload.setdefault("legend", None)
    if arguments.mode in ("all", "certificate"):
        for case_name, requested_cells in CASES:
            record = _build_certificate_case(case_name, requested_cells)
            rendered = _draw_certificate(record)
            payload["certificate"].append(
                {
                    "case": case_name,
                    "requested_cells": requested_cells,
                    **rendered,
                }
            )
            if case_name == "weak-rotation-reactor-static" and requested_cells == -500:
                locality = _error_locality(record)
                payload["error_locality"] = locality | {
                    "project_src": _draw_error_locality(record, locality)
                }
            _write_receipt(payload)
            print(f"PERSISTED certificate {case_name} {requested_cells}", flush=True)
            jax.clear_caches()
    if arguments.mode in ("all", "clip"):
        for requested_cells in (-110, -342):
            payload["clipped_cells"].append(_draw_clipped_cells(requested_cells))
            _write_receipt(payload)
            print(f"PERSISTED clipped cells {requested_cells}", flush=True)
            jax.clear_caches()
    if arguments.mode in ("all", "cold"):
        payload["cold_start"] = _draw_cold_start()
        payload["legend"] = _draw_legend()
        _write_receipt(payload)
        print("PERSISTED cold start and marker legend", flush=True)


if __name__ == "__main__":
    main()
