#!/usr/bin/env python3
"""Measure four fixed-capacity wedges in the analytic single-null X-point cell."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib.path import Path as PlotPath
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from benchmarks import topology_read_resolution_ladder as topology_ladder
from nova.equilibrium.clip_quadrature import saddle_wedge_current_moments
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/cut-cell-current-attribution/xpoint-cell"
REQUESTED_CELLS = (110, 300, 500)
REFERENCE_CELLS = 110
MU_0 = 4.0e-7 * math.pi
CORE_CURRENT_RELATIVE_LIMIT = 1.0e-13
COLOURS = {
    "analytic": "#2563eb",
    "read": "#d97706",
    "cell": "#111827",
    "core": "#0f766e",
    "private": "#7c3aed",
    "sol": "#dc2626",
}


def _part_path(output: Path, requested_cells: int) -> Path:
    return output / "parts" / f"single-null-wedges-cells-{requested_cells}.json"


class _FluxPlaceholder:
    @staticmethod
    def sample(points, _cell_index):
        zero = jnp.zeros(points.shape[:-1], dtype=points.dtype)
        return zero, zero, zero


@dataclass(frozen=True)
class _AnalyticCurrentProfile:
    source_parameter: float
    flux_scale: float
    major_radius: float

    def current_density(self, radius, _normalised_flux):
        x = radius / self.major_radius
        source = (
            self.flux_scale
            / self.major_radius**2
            * (self.source_parameter + (1.0 - self.source_parameter) * x**2)
        )
        return -source / (MU_0 * radius)


class _ZeroCurrentProfile:
    @staticmethod
    def current_density(radius, _normalised_flux):
        return jnp.zeros_like(radius)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if not job_id or partition != "all_debug":
        raise RuntimeError("the wedge oracle must run in one all_debug allocation")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp in the allocation")
    if os.environ.get("JAX_PLATFORMS") != "cpu" or jax.default_backend() != "cpu":
        raise RuntimeError("the wedge oracle must select the JAX CPU backend")
    return {
        "job_id": int(job_id),
        "partition": partition,
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platforms": ["cpu"],
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[float, float]:
    edge = end - start
    fraction = float(np.dot(point - start, edge) / np.dot(edge, edge))
    clipped = float(np.clip(fraction, 0.0, 1.0))
    nearest = start + clipped * edge
    return float(np.linalg.norm(point - nearest)), fraction


def _candidate_cells(machine: Any, x_point: np.ndarray) -> list[dict[str, Any]]:
    """Return point-in-polygon and nearest-edge facts for saddle candidates."""
    rows = []
    for index, item in enumerate(machine.cell_polygons):
        polygon = np.asarray(item, dtype=np.float64)
        distances = [
            _point_segment_distance(x_point, first, second)[0]
            for first, second in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
        ]
        path = PlotPath(polygon)
        strict = bool(path.contains_point(x_point))
        expanded = bool(path.contains_point(x_point, radius=1.0e-12))
        contracted = bool(path.contains_point(x_point, radius=-1.0e-12))
        if expanded or min(distances) <= 1.0e-12:
            rows.append(
                {
                    "cell": index,
                    "contains_strict": strict,
                    "contains_expanded": expanded,
                    "contains_contracted": contracted,
                    "nearest_edge_distance_m": min(distances),
                }
            )
    return rows


def _xpoint_cell(machine: Any, x_point: np.ndarray) -> tuple[int, list[dict[str, Any]]]:
    candidates = _candidate_cells(machine, x_point)
    containing = [row["cell"] for row in candidates if row["contains_expanded"]]
    if len(containing) != 1:
        raise RuntimeError(
            f"expected one polygon containing the analytic saddle, found {containing}"
        )
    return int(containing[0]), candidates


def _edge_root_diagnostics(
    exact: Any,
    polygon: np.ndarray,
    x_point: np.ndarray,
    boundary_flux: float,
) -> list[dict[str, Any]]:
    """Resolve exact roots and saddle coincidences on every candidate edge."""
    rows = []
    flux_scale = max(
        abs(float(exact.flux(exact.magnetic_axis[None, :])[0]) - boundary_flux),
        1.0,
    )
    flux_tolerance = 256.0 * np.finfo(np.float64).eps * flux_scale
    geometry_scale = max(float(np.max(np.abs(polygon))), 1.0)
    edge_tolerance = 2048.0 * np.finfo(np.float64).eps * geometry_scale
    for edge_index, (start, end) in enumerate(
        zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    ):
        parameters = np.linspace(0.0, 1.0, 257)
        points = start[None, :] + parameters[:, None] * (end - start)[None, :]
        values = np.asarray(exact.flux(points), dtype=np.float64) - boundary_flux
        roots: list[dict[str, Any]] = []

        def append_root(fraction: float, kind: str) -> None:
            point = start + fraction * (end - start)
            duplicate = next(
                (
                    row
                    for row in roots
                    if np.linalg.norm(point - np.asarray(row["coordinate_rz_m"]))
                    <= 1.0e-12
                ),
                None,
            )
            if duplicate is not None:
                if kind == "saddle-coincident":
                    duplicate["kind"] = kind
                    duplicate["multiplicity"] = 2
                return
            roots.append(
                {
                    "fraction": fraction,
                    "coordinate_rz_m": point,
                    "kind": kind,
                    "multiplicity": 1,
                    "flux_residual_wb": float(
                        exact.flux(point[None, :])[0] - boundary_flux
                    ),
                }
            )

        for slot in np.flatnonzero(np.abs(values) <= flux_tolerance):
            append_root(float(parameters[slot]), "sample-zero")
        for slot in np.flatnonzero(values[:-1] * values[1:] < 0.0):
            lower = float(parameters[slot])
            upper = float(parameters[slot + 1])
            fraction = brentq(
                lambda value: float(
                    exact.flux((start + value * (end - start))[None, :])[0]
                    - boundary_flux
                ),
                lower,
                upper,
                xtol=4.0 * np.finfo(np.float64).eps,
                rtol=4.0 * np.finfo(np.float64).eps,
            )
            append_root(float(fraction), "sign-change")
        saddle_distance, saddle_fraction = _point_segment_distance(x_point, start, end)
        saddle_on_edge = bool(
            saddle_distance <= edge_tolerance
            and -edge_tolerance <= saddle_fraction <= 1.0 + edge_tolerance
        )
        if saddle_on_edge:
            append_root(float(np.clip(saddle_fraction, 0.0, 1.0)), "saddle-coincident")
        rows.append(
            {
                "edge": edge_index,
                "start_rz_m": start,
                "end_rz_m": end,
                "endpoint_signed_flux_wb": [float(values[0]), float(values[-1])],
                "endpoint_sign_change": bool((values[0] > 0.0) != (values[-1] > 0.0)),
                "saddle_distance_m": saddle_distance,
                "saddle_fraction": saddle_fraction,
                "saddle_on_edge": saddle_on_edge,
                "root_count": sum(int(root["multiplicity"]) for root in roots),
                "roots": roots,
            }
        )
    return rows


def _edge_root_arrays(
    edge_rows: list[dict[str, Any]], polarity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack exact edge roots and their following signs into fixed arrays."""
    width = len(edge_rows)
    events = []
    for row in edge_rows:
        edge = int(row["edge"])
        for root in row["roots"]:
            parameter = (edge + float(root["fraction"])) % width
            coordinate = np.asarray(root["coordinate_rz_m"], dtype=np.float64)
            duplicate = next(
                (
                    event
                    for event in events
                    if abs(float(event["parameter"]) - parameter) <= 1.0e-12
                    and np.linalg.norm(np.asarray(event["coordinate"]) - coordinate)
                    <= 1.0e-12
                ),
                None,
            )
            multiplicity = int(root.get("multiplicity", 1))
            if duplicate is None:
                events.append(
                    {
                        "parameter": parameter,
                        "coordinate": coordinate,
                        "multiplicity": multiplicity,
                    }
                )
            else:
                duplicate["multiplicity"] = max(
                    int(duplicate["multiplicity"]), multiplicity
                )
    events.sort(key=lambda event: float(event["parameter"]))
    expanded = []
    positive = polarity * float(edge_rows[0]["endpoint_signed_flux_wb"][0]) > 0.0
    for event in events:
        for _copy in range(int(event["multiplicity"])):
            positive = not positive
            expanded.append((float(event["parameter"]), positive))
    if len(expanded) != 4:
        raise RuntimeError(f"expected four separatrix roots, found {len(expanded)}")

    fractions = np.zeros((1, width, 2), dtype=np.float64)
    counts = np.zeros((1, width), dtype=np.int32)
    positive_after = np.zeros((1, width, 2), dtype=bool)
    for parameter, following_positive in expanded:
        edge = min(int(math.floor(parameter)), width - 1)
        slot = int(counts[0, edge])
        if slot >= 2:
            raise RuntimeError(f"edge {edge} carries more than two separatrix roots")
        fractions[0, edge, slot] = parameter - edge
        positive_after[0, edge, slot] = following_positive
        counts[0, edge] += 1
    return fractions, counts, positive_after


def _wedge_shape_diagnostics(wedges: Any, x_point: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(wedges.vertex_count)[0]
    vertices = np.asarray(wedges.support_vertices)[0]
    rows = []
    for slot, count in enumerate(counts):
        padding = vertices[slot, count:]
        nonzero = np.flatnonzero(np.any(padding != 0.0, axis=1)) + count
        rows.append(
            {
                "wedge": slot,
                "vertex_count": int(count),
                "first_vertex_rz_m": vertices[slot, 0],
                "first_vertex_delta_from_saddle_m": vertices[slot, 0] - x_point,
                "first_vertex_exact_saddle": bool(
                    np.array_equal(vertices[slot, 0], x_point)
                ),
                "nonzero_padding_slots": nonzero,
                "nonzero_padding_coordinates_rz_m": vertices[slot, nonzero],
            }
        )
    return {
        "saddle_row": bool(np.asarray(wedges.saddle)[0]),
        "vertex_capacity": int(vertices.shape[1]),
        "wedges": rows,
    }


def _vertical_intervals(
    vertices: np.ndarray, radius: float
) -> list[tuple[float, float]]:
    heights = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        span = float(second[0] - first[0])
        if span == 0.0:
            continue
        fraction = (radius - float(first[0])) / span
        if 0.0 <= fraction <= 1.0:
            heights.append(float(first[1] + fraction * (second[1] - first[1])))
    unique = []
    for height in sorted(heights):
        if not unique or abs(height - unique[-1]) > 1.0e-12:
            unique.append(height)
    return [
        (lower, upper)
        for lower, upper in zip(unique[0::2], unique[1::2])
        if upper > lower
    ]


def _analytic_polygon_moments(
    vertices: np.ndarray,
    centre: np.ndarray,
    profile: _AnalyticCurrentProfile | _ZeroCurrentProfile,
) -> np.ndarray:
    if isinstance(profile, _ZeroCurrentProfile):
        return np.zeros(3)
    breaks = sorted(set(float(value) for value in vertices[:, 0]))

    def density(radius: float) -> float:
        x = radius / profile.major_radius
        source = (
            profile.flux_scale
            / profile.major_radius**2
            * (profile.source_parameter + (1.0 - profile.source_parameter) * x**2)
        )
        return -source / (MU_0 * radius)

    values = np.zeros(3)
    for lower, upper in zip(breaks, breaks[1:]):
        if upper <= lower:
            continue

        def integrand(radius: float, moment: int) -> float:
            current = density(radius)
            if moment == 0:
                return current * sum(
                    top - bottom
                    for bottom, top in _vertical_intervals(vertices, radius)
                )
            if moment == 1:
                return (
                    current
                    * (radius - centre[0])
                    * sum(
                        top - bottom
                        for bottom, top in _vertical_intervals(vertices, radius)
                    )
                )
            return (
                0.5
                * current
                * sum(
                    (top - centre[1]) ** 2 - (bottom - centre[1]) ** 2
                    for bottom, top in _vertical_intervals(vertices, radius)
                )
            )

        for moment in range(3):
            value, _error = quad(
                lambda radius, slot=moment: integrand(radius, slot),
                lower,
                upper,
                epsabs=1.0e-10,
                epsrel=2.0e-13,
                limit=100,
            )
            values[moment] += value
    return values


def _support_vertices(wedges: Any, slot: int) -> np.ndarray:
    count = int(np.asarray(wedges.vertex_count)[0, slot])
    return np.asarray(wedges.support_vertices)[0, slot, :count]


def _exact_edge_roots(
    polygon: np.ndarray, exact: Any, boundary_flux: float
) -> list[tuple[float, np.ndarray]]:
    """Resolve the separatrix roots on the cell boundary from the exact flux."""
    roots: list[tuple[float, np.ndarray]] = []
    parameters = np.linspace(0.0, 1.0, 257)
    for edge, (start, end) in enumerate(
        zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    ):
        points = start[None, :] + parameters[:, None] * (end - start)[None, :]
        values = np.asarray(exact.flux(points), dtype=np.float64) - boundary_flux
        for slot in np.flatnonzero(values[:-1] * values[1:] < 0.0):
            fraction = brentq(
                lambda value, s=start, e=end: float(
                    exact.flux((s + value * (e - s))[None, :])[0] - boundary_flux
                ),
                float(parameters[slot]),
                float(parameters[slot + 1]),
                xtol=8.9e-16,
                rtol=8.9e-16,
            )
            roots.append((edge + float(fraction), start + fraction * (end - start)))
    roots.sort(key=lambda item: item[0])
    return roots


def _polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    radial = vertices[:, 0]
    vertical = vertices[:, 1]
    following_radial = np.roll(radial, -1)
    following_vertical = np.roll(vertical, -1)
    cross = radial * following_vertical - following_radial * vertical
    area = 0.5 * float(np.sum(cross))
    return np.asarray(
        [
            float(np.sum((radial + following_radial) * cross)) / (6.0 * area),
            float(np.sum((vertical + following_vertical) * cross)) / (6.0 * area),
        ]
    )


def _independent_branch_sectors(
    polygon: np.ndarray,
    exact: Any,
    saddle: np.ndarray,
    boundary_flux: float,
    polarity: float,
) -> list[dict[str, Any]]:
    """Build the four sectors from the exact roots and the cell boundary alone.

    Every sector is bounded by one boundary arc of the cell (the straight
    edges between two consecutive separatrix roots) and the two branch chords
    that meet at the saddle.  The confined side of each sector is read from the
    exact flux at the midpoint of its boundary arc, so neither the sector
    polygons nor their signs come from the carrier's emitted vertices.
    """
    roots = _exact_edge_roots(polygon, exact, boundary_flux)
    if len(roots) != 4:
        raise RuntimeError(f"expected four exact separatrix roots, found {len(roots)}")
    width = len(polygon)
    perimeter = float(width)
    sectors = []
    for slot in range(4):
        parameter, point = roots[slot]
        next_parameter, next_point = roots[(slot + 1) % 4]
        span = next_parameter - parameter
        if span <= 0.0:
            span += perimeter
        between = []
        for step in range(1, width + 1):
            vertex_index = (int(math.floor(parameter)) + step) % width
            absolute = float(vertex_index)
            if absolute <= parameter:
                absolute += perimeter
            if parameter < absolute < parameter + span:
                between.append(polygon[vertex_index])
        vertices = np.asarray([saddle, point, *between, next_point], dtype=np.float64)
        probe_parameter = parameter + 0.5 * span
        probe_edge = int(math.floor(probe_parameter)) % width
        probe_fraction = probe_parameter - math.floor(probe_parameter)
        probe = polygon[probe_edge] + probe_fraction * (
            polygon[(probe_edge + 1) % width] - polygon[probe_edge]
        )
        confined = bool(
            polarity * (float(exact.flux(probe[None, :])[0]) - boundary_flux) > 0.0
        )
        sectors.append(
            {
                "vertices": vertices,
                "confined": confined,
                "arc_midpoint_rz_m": probe,
            }
        )
    return sectors


def _closing_vertex_sequence(vertices: np.ndarray) -> list[list[float]]:
    """Return a rotation- and direction-free canonical vertex sequence."""
    points = [tuple(float(value) for value in row) for row in vertices]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    forward = min(points[index:] + points[:index] for index in range(len(points)))
    reversed_points = list(reversed(points))
    backward = min(
        reversed_points[index:] + reversed_points[:index]
        for index in range(len(reversed_points))
    )
    return [list(point) for point in min(forward, backward)]


def _observed_nulls(operator: Any, analytic: np.ndarray) -> dict[str, Any]:
    physical = jnp.asarray(analytic[: operator.physical_node_number], dtype=jnp.float64)
    _masks, topology, _connected, axis_admitted = jax.block_until_ready(
        operator._fixed_design_read(physical)
    )
    axis = np.asarray(topology.axis, dtype=np.float64)
    saddle = np.asarray(topology.x_point, dtype=np.float64)
    admitted = bool(topology.diverted and np.all(np.isfinite(saddle)))
    return {
        "axis_admitted": bool(axis_admitted),
        "axis_rz_m": axis,
        "saddle_admitted": admitted,
        "saddle_rz_m": saddle if admitted else None,
    }


def _render_panel(
    output: Path,
    requested_cells: int,
    machine: Any,
    exact: Any,
    polygon: np.ndarray,
    wedges: Any,
    observed: dict[str, Any],
) -> str:
    wall = np.asarray(machine.wall_node, dtype=np.float64)
    radial = np.linspace(float(np.min(wall[:, 0])), float(np.max(wall[:, 0])), 241)
    vertical = np.linspace(float(np.min(wall[:, 1])), float(np.max(wall[:, 1])), 241)
    radial_grid, vertical_grid = np.meshgrid(radial, vertical)
    points = np.column_stack((radial_grid.ravel(), vertical_grid.ravel()))
    flux = np.asarray(exact.flux(points), dtype=np.float64).reshape(radial_grid.shape)
    levels = poloidal.contour_levels(flux, count=16)
    figure, axis = plt.subplots(figsize=(6.2, 6.6), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, vertical, flux, levels, color=COLOURS["analytic"]
    )
    poloidal.draw_wall(axis, units=(wall,), linewidth=0.75)
    analytic_style = DEFAULT_INK.variant(
        axis_marker="^",
        axis_color=COLOURS["analytic"],
        xpoint_color=COLOURS["analytic"],
    )
    poloidal.draw_nulls(
        axis,
        magnetic_axis=np.asarray(exact.magnetic_axis),
        x_points=np.asarray(exact.x_point)[None, :],
        style=analytic_style,
        contain=(wall,),
    )
    observed_style = DEFAULT_INK.variant(
        axis_marker="v",
        axis_color=COLOURS["read"],
        xpoint_color=COLOURS["read"],
    )
    observed_x = (
        np.asarray(observed["saddle_rz_m"])[None, :]
        if observed["saddle_admitted"]
        else np.empty((0, 2))
    )
    poloidal.draw_nulls(
        axis,
        magnetic_axis=np.asarray(observed["axis_rz_m"]),
        x_points=observed_x,
        style=observed_style,
        contain=(wall,),
    )
    cell_loop = np.vstack((polygon, polygon[0]))
    axis.plot(cell_loop[:, 0], cell_loop[:, 1], color=COLOURS["cell"], linewidth=2.0)
    for slot, colour in enumerate(
        (COLOURS["core"], COLOURS["private"], COLOURS["sol"], COLOURS["sol"])
    ):
        wedge = _support_vertices(wedges, slot)
        wedge_loop = np.vstack((wedge, wedge[0]))
        axis.plot(wedge_loop[:, 0], wedge_loop[:, 1], color=colour, linewidth=1.1)
    poloidal_axes(axis)
    axis.set_title(
        f"{requested_cells} requested / {len(machine.node)} realised cells\n"
        "analytic nulls blue; production read ochre; X-point cell outlined",
        fontsize=9,
    )
    name = f"single-null-wedges-cells-{requested_cells}.svg"
    destination = output / name
    figure.savefig(destination)
    plt.close(figure)
    return f"/nova/figures/cut-cell-current-attribution/xpoint-cell/{name}"


def _measure_row(
    requested_cells: int,
    output: Path,
    allocation: dict[str, Any],
    *,
    diagnose_only: bool = False,
) -> dict[str, Any]:
    started = perf_counter()
    part_path = _part_path(output, requested_cells)
    progress = {
        "schema": "nova.xpoint-cell-wedge-oracle-part",
        "version": 1,
        "source_revision": _source_revision(),
        "allocation": allocation,
        "requested_cells": requested_cells,
        "reference": requested_cells == REFERENCE_CELLS,
        "completed": False,
        "stage": "machine-load",
    }
    _write_json(part_path, progress)
    machine, operator, analytic = topology_ladder._machine_and_field(requested_cells)
    exact = topology_ladder.ANALYTIC
    x_point = np.asarray(exact.x_point, dtype=np.float64)
    axis = np.asarray(exact.magnetic_axis, dtype=np.float64)
    boundary_flux = float(exact.flux(x_point[None, :])[0])
    axis_flux = float(exact.flux(axis[None, :])[0])
    polarity = math.copysign(1.0, axis_flux - boundary_flux)
    cell, candidates = _xpoint_cell(machine, x_point)
    candidate_diagnostics = []
    selected_geometry = None
    for candidate in candidates:
        candidate_cell = int(candidate["cell"])
        candidate_polygon = np.asarray(
            machine.cell_polygons[candidate_cell], dtype=np.float64
        )
        candidate_centre = np.asarray(machine.node[candidate_cell], dtype=np.float64)
        candidate_mesh = AtomicCellMesh.from_cells(
            [candidate_polygon], centroids=candidate_centre[None, :]
        )
        node_flux = np.asarray(
            exact.flux(candidate_mesh.node_coordinates), dtype=np.float64
        )
        signed_flux = jnp.asarray(polarity * (node_flux - boundary_flux))
        edge_rows = _edge_root_diagnostics(
            exact, candidate_polygon, x_point, boundary_flux
        )
        root_fraction, root_count, root_positive_after = _edge_root_arrays(
            edge_rows, polarity
        )
        candidate_wedges = jax.jit(
            lambda values: candidate_mesh.traced_saddle_wedges(
                values,
                saddle_vertex=jnp.asarray(x_point),
                core_reference=jnp.asarray(axis),
                edge_root_fraction=jnp.asarray(root_fraction),
                edge_root_count=jnp.asarray(root_count),
                edge_root_positive_after=jnp.asarray(root_positive_after),
            )
        )(signed_flux)
        candidate_row = candidate | {
            "selected": candidate_cell == cell,
            "centroid_rz_m": candidate_centre,
            "characteristic_pitch_m": math.sqrt(
                float(np.asarray(machine.area)[candidate_cell])
            ),
            "edge_root_count": int(sum(row["root_count"] for row in edge_rows)),
            "edge_roots": edge_rows,
            "wedge_shape": _wedge_shape_diagnostics(candidate_wedges, x_point),
        }
        candidate_diagnostics.append(candidate_row)
        if candidate_cell == cell:
            selected_geometry = (
                candidate_polygon,
                candidate_centre,
                candidate_wedges,
                candidate_row,
            )
    if selected_geometry is None:
        raise RuntimeError("the selected X-point cell has no diagnostic row")
    polygon, centre, wedges, selected_diagnostic = selected_geometry
    saddle_case = (
        "edge-coincident-degenerate-wedge"
        if any(row["saddle_on_edge"] for row in selected_diagnostic["edge_roots"])
        else "interior-four-crossing"
    )
    progress = progress | {
        "realised_cells": len(machine.node),
        "machine_cache": machine.cache,
        "xpoint_cell": cell,
        "saddle_case": saddle_case,
        "candidate_cells": candidate_diagnostics,
        "stage": "geometry-diagnosed",
        "wall_seconds": perf_counter() - started,
    }
    _write_json(part_path, progress)
    if diagnose_only:
        diagnostic = progress | {
            "diagnostic_completed": True,
            "oracle_completed": False,
            "completed": True,
        }
        _write_json(part_path, diagnostic)
        return diagnostic

    profile = _AnalyticCurrentProfile(
        source_parameter=float(exact.source_parameter),
        flux_scale=float(exact.flux_scale_per_radian_wb),
        major_radius=float(exact.major_radius),
    )
    zero = _ZeroCurrentProfile()
    profiles = (profile, zero, zero, zero)
    measured = saddle_wedge_current_moments(wedges, _FluxPlaceholder(), profiles)
    actual = np.stack(
        (
            np.asarray(measured.cell_current)[0],
            np.asarray(measured.radial_moment)[0],
            np.asarray(measured.vertical_moment)[0],
        ),
        axis=1,
    )
    sectors = _independent_branch_sectors(
        polygon, exact, x_point, boundary_flux, polarity
    )
    core_direction = axis - x_point
    confined_slots = [slot for slot, item in enumerate(sectors) if item["confined"]]
    if len(confined_slots) != 2:
        raise RuntimeError(
            f"expected two confined sectors, found {len(confined_slots)}"
        )
    core_slot = max(
        confined_slots,
        key=lambda slot: float(
            np.dot(
                _polygon_centroid(sectors[slot]["vertices"]) - x_point,
                core_direction,
            )
        ),
    )
    expected = np.zeros((4, 3))
    expected[0] = _analytic_polygon_moments(
        sectors[core_slot]["vertices"], centre, profiles[0]
    )
    geometry_matches_carrier = _closing_vertex_sequence(
        _support_vertices(wedges, 0)
    ) == _closing_vertex_sequence(sectors[core_slot]["vertices"])
    scale = np.maximum(np.abs(expected), 1.0e-12)
    relative_error = np.abs(actual - expected) / scale
    core_current_relative_error = float(relative_error[0, 0])
    progress = progress | {
        "stage": "moments-integrated",
        "measured_moments": actual,
        "analytic_moments": expected,
        "relative_moment_error": relative_error,
        "core_current_relative_error": core_current_relative_error,
        "private_flux_current_a": float(actual[1, 0]),
        "common_sol_current_a": [float(actual[2, 0]), float(actual[3, 0])],
        "wall_seconds": perf_counter() - started,
    }
    _write_json(part_path, progress)
    if core_current_relative_error > CORE_CURRENT_RELATIVE_LIMIT:
        raise AssertionError(
            f"core current relative error {core_current_relative_error:.3e} exceeds "
            f"{CORE_CURRENT_RELATIVE_LIMIT:.1e}"
        )
    if not np.array_equal(actual[1:], np.zeros((3, 3))):
        raise AssertionError("non-core wedge profiles produced non-zero moments")
    counts = np.asarray(wedges.vertex_count)[0]
    vertices = np.asarray(wedges.support_vertices)[0]
    exact_zero_padding = all(
        bool(np.all(vertices[slot, count:] == 0.0)) for slot, count in enumerate(counts)
    )
    saddle_inserted = all(
        np.array_equal(vertices[slot, 0], x_point) for slot in range(4)
    )
    if not exact_zero_padding or not saddle_inserted:
        raise AssertionError("wedge padding or saddle insertion is not exact")
    area_closure = float(
        np.sum(np.asarray(wedges.area)) - np.asarray(wedges.full_area)[0]
    )
    if abs(area_closure) > 2.0e-12:
        raise AssertionError(f"wedge area closure is {area_closure:.3e} m2")
    observed = _observed_nulls(operator, analytic)
    figure_src = _render_panel(
        output, requested_cells, machine, exact, polygon, wedges, observed
    )
    row = {
        "schema": "nova.xpoint-cell-wedge-oracle-part",
        "version": 1,
        "source_revision": _source_revision(),
        "allocation": allocation,
        "requested_cells": requested_cells,
        "reference": requested_cells == REFERENCE_CELLS,
        "realised_cells": len(machine.node),
        "machine_cache": machine.cache,
        "xpoint_cell": cell,
        "saddle_case": saddle_case,
        "candidate_cells": candidate_diagnostics,
        "wedge_vertex_capacity": int(wedges.support_vertices.shape[2]),
        "wedge_vertex_count": counts,
        "wedge_area_m2": np.asarray(wedges.area)[0],
        "wedge_area_closure_m2": area_closure,
        "saddle_inserted_as_first_vertex": saddle_inserted,
        "exact_zero_padding": exact_zero_padding,
        "profile_order": ["confined-core", "zero-private", "zero-sol", "zero-sol"],
        "independent_core_sector": int(core_slot),
        "independent_confined_sectors": [int(slot) for slot in confined_slots],
        "independent_sector_area_m2": [
            abs(
                float(
                    np.sum(
                        np.asarray(item["vertices"])[:, 0]
                        * np.roll(np.asarray(item["vertices"])[:, 1], -1)
                    )
                )
                - float(
                    np.sum(
                        np.asarray(item["vertices"])[:, 1]
                        * np.roll(np.asarray(item["vertices"])[:, 0], -1)
                    )
                )
            )
            / 2.0
            for item in sectors
        ],
        "carrier_core_polygon_matches_independent": geometry_matches_carrier,
        "measured_moments": actual,
        "analytic_moments": expected,
        "relative_moment_error": relative_error,
        "core_current_relative_error": core_current_relative_error,
        "private_flux_current_a": float(actual[1, 0]),
        "common_sol_current_a": [float(actual[2, 0]), float(actual[3, 0])],
        "observed_nulls": observed,
        "figure_src": figure_src,
        "wall_seconds": perf_counter() - started,
        "oracle_completed": True,
        "completed": True,
        "stage": "complete",
    }
    _write_json(part_path, row)
    return row


def run(
    output: Path,
    requested_cells: tuple[int, ...] = REQUESTED_CELLS,
    *,
    diagnose_only: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    allocation = _allocation()
    receipt = {
        "schema": "nova.xpoint-cell-wedge-oracle",
        "version": 1,
        "source_revision": _source_revision(),
        "allocation": allocation,
        "reference_requested_cells": REFERENCE_CELLS,
        "requested_cells": list(requested_cells),
        "diagnose_only": diagnose_only,
        "core_current_relative_limit": CORE_CURRENT_RELATIVE_LIMIT,
        "rows": [],
        "completed": False,
    }
    receipt_path = output / "receipt.json"
    _write_json(receipt_path, receipt)
    rows = []
    for cells in requested_cells:
        try:
            row = _measure_row(
                cells,
                output,
                allocation,
                diagnose_only=diagnose_only,
            )
        except Exception as error:
            part_path = _part_path(output, cells)
            failed = (
                json.loads(part_path.read_text(encoding="utf-8"))
                if part_path.exists()
                else {"requested_cells": cells}
            )
            failed.update(
                {
                    "completed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_json(part_path, failed)
            receipt.update(
                {
                    "rows": rows,
                    "failed_requested_cells": cells,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_json(receipt_path, receipt)
            raise
        rows.append(row)
        receipt["rows"] = rows
        _write_json(receipt_path, receipt)

    tested = [
        row
        for row in rows
        if not row["reference"] and bool(row.get("oracle_completed"))
    ]
    receipt.update(
        {
            "tested_requested_cells": [row["requested_cells"] for row in tested],
            "max_core_current_relative_error": (
                max(row["core_current_relative_error"] for row in tested)
                if tested
                else None
            ),
            "private_flux_current_exact_zero": (
                all(row["private_flux_current_a"] == 0.0 for row in tested)
                if tested
                else None
            ),
            "common_sol_current_exact_zero": (
                all(row["common_sol_current_a"] == [0.0, 0.0] for row in tested)
                if tested
                else None
            ),
            "all_saddles_inserted": (
                all(row["saddle_inserted_as_first_vertex"] for row in rows)
                if not diagnose_only
                else None
            ),
            "all_padding_exact_zero": (
                all(row["exact_zero_padding"] for row in rows)
                if not diagnose_only
                else None
            ),
            "all_carrier_core_polygons_match_independent": (
                all(
                    row["carrier_core_polygon_matches_independent"]
                    for row in rows
                    if row.get("oracle_completed")
                )
                if not diagnose_only
                else None
            ),
            "diagnostic_completed": diagnose_only,
            "completed": True,
        }
    )
    _write_json(receipt_path, receipt)
    if diagnose_only:
        print(
            f"XPOINT_WEDGE_DIAGNOSTIC cells={list(requested_cells)} parts={len(rows)}",
            flush=True,
        )
        return receipt
    print(
        "XPOINT_WEDGE_ORACLE "
        f"tested={receipt['tested_requested_cells']} "
        f"max_core_relative={receipt['max_core_current_relative_error']:.3e} "
        f"private_zero={receipt['private_flux_current_exact_zero']} "
        f"sol_zero={receipt['common_sol_current_exact_zero']}",
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cells", type=int, nargs="+", default=REQUESTED_CELLS)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    run(args.output, tuple(args.cells), diagnose_only=args.diagnose_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
