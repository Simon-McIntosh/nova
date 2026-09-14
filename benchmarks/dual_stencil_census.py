#!/usr/bin/env python3
"""Compare a cell-centred and a vertex-centred stationary-point census.

The cell-centred census fits one quadratic on each cell centroid and its six
authored sampling vertices (seven samples, six coefficients) and reports the
closed-form stationary point with its Hessian class, admitted when the point
lies inside the cell polygon or within a quarter characteristic pitch of it.
The vertex-centred dual fits the same quadratic about every interior mesh
vertex on the vertex value, the three surrounding cell centroids and the three
adjacent vertices (seven samples, six coefficients), and applies the same
stationary-point and Hessian rule about the vertex's own triangle of
centroids.  Both read the analytic single-null flux directly on the cached
oracle carriers, so the comparison isolates the census geometry from any
production reconstruction.

Per rung and row the receipt records each census's admission of the analytic
X-point and axis, the position error in metres and pitch, the distance of the
analytic null to the nearest cell edge and nearest sampling vertex in pitch,
and the separation of the two polished positions where both admit.  False
candidates are counted before and after the Hessian and containment filters.
The report bins the position error by the null's distance to the nearest cell
boundary and states, per bin, the mean error of each census and the fraction
of nulls at which the dual census is the more accurate of the two.

The labels are data identifiers only; the implementation names mechanisms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any
import uuid

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from nova.equilibrium.stencil_mesh import RING_CONDITION_LIMIT
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/dual-stencil"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/dual-stencil"
)
REQUESTED_CELL_COUNTS = (110, 200, 300, 342, 400, 500, 750, 1000, 2500)
STATIC_CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
CPU_ROWS = tuple(
    (certificate.DIVERTED_CASE_NAME, cells) for cells in REQUESTED_CELL_COUNTS
) + tuple((case_name, cells) for case_name in STATIC_CASES for cells in (300, 1000))
WALL_NODE_COUNT = 121
EDGE_TOLERANCE_IN_PITCH = 0.25
COLOURS = {
    "primary": "#3366cc",
    "dual": "#8a2be2",
    "axis": "#1b7837",
    "saddle": "#d1495b",
    "analytic": "#222222",
}


def _strict(value: Any) -> Any:
    """Return JSON-native data with non-finite values represented explicitly."""

    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_revision() -> str:
    """Return the repository revision supplying this benchmark."""

    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _slug(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-cells-{requested_cells}"


def _part_path(report_directory: Path, case_name: str, requested_cells: int) -> Path:
    return report_directory / "parts" / f"{_slug(case_name, requested_cells)}.json"


def _load_part(path: Path) -> dict[str, Any]:
    """Load one complete receipt part, rejecting pathologically large files."""

    if path.stat().st_size > 64 * 1024 * 1024:
        raise RuntimeError(f"part receipt exceeds the sixty-four MiB bound: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("completed"):
        raise RuntimeError(f"part receipt is incomplete: {path}")
    return payload


def _allocation(kind: str) -> dict[str, Any]:
    """Require the declared scheduler resources and numerical backend."""

    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    memory = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    platforms = os.environ.get("JAX_PLATFORMS")
    if not job_id:
        raise RuntimeError("the measurement must run inside a scheduler allocation")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if memory < 64 * 1024:
        raise RuntimeError(f"expected at least 64 GiB, received {memory} MiB")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp")
    if kind == "cpu":
        if partition != "all_debug" or platforms != "cpu":
            raise RuntimeError(
                f"expected all_debug and cpu, received {partition!r} and {platforms!r}"
            )
        if jax.default_backend() != "cpu":
            raise RuntimeError("the CPU measurement selected a non-CPU backend")
    else:
        raise ValueError(f"unknown allocation kind {kind!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": partition,
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": cpus,
        "memory_mb": memory,
        "jax_platforms": platforms.split(",") if platforms else [],
        "jax_default_backend": jax.default_backend(),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _machine_and_field(case_name: str, requested_cells: int):
    """Load one cached carrier, operator, exact state, and reference nulls."""

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = limiter_audit._machine(
        case_name, carrier_case, exact, -requested_cells, WALL_NODE_COUNT
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = limiter_audit._exact_flux(case_name, exact, coordinates)
    if float(np.ptp(state[: len(machine.node)])) <= 1.0e-10:
        raise RuntimeError("analytic-flux positive control saw a uniform grid field")
    operator = limiter_audit.oracle_fixture.forward_operator(source_case, machine)
    return machine, operator, state, exact


def _support_stencil(operator: Any) -> Any:
    """Return the one six-vertex own-node stencil covering every cell."""

    matches = [
        stencil
        for stencil in operator._support_moment_stencils
        if stencil.ring_gather_index is not None
        and stencil.ring_gather_index.shape[1] == 7
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one six-vertex stencil, received {len(matches)}")
    stencil = matches[0]
    expected = np.arange(operator.grid.node_number, dtype=np.intp)
    if not np.array_equal(stencil.ring_centre, expected):
        raise RuntimeError("the six-vertex stencil does not cover every cell")
    return stencil


def _cell_edges(polygons: tuple[np.ndarray, ...]):
    """Return fixed-capacity directed edges for every cell polygon."""

    width = max(len(polygon) for polygon in polygons)
    start = np.zeros((len(polygons), width, 2), dtype=np.float64)
    end = np.zeros_like(start)
    valid = np.zeros((len(polygons), width), dtype=bool)
    for cell, polygon in enumerate(polygons):
        count = len(polygon)
        start[cell, :count] = polygon
        end[cell, :count] = np.roll(polygon, -1, axis=0)
        valid[cell, :count] = True
    return start, end, valid


def _inside_or_near_cell(
    position: jax.Array,
    edge_start: jax.Array,
    edge_end: jax.Array,
    edge_valid: jax.Array,
    distance_limit: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Test each row's point against its corresponding polygon and edges."""

    point = position[:, None, :]
    first_x, first_y = edge_start[..., 0], edge_start[..., 1]
    second_x, second_y = edge_end[..., 0], edge_end[..., 1]
    point_x, point_y = point[..., 0], point[..., 1]
    vertical_span = second_y - first_y
    safe_span = jnp.where(jnp.abs(vertical_span) > 0.0, vertical_span, 1.0)
    crossing_x = first_x + (point_y - first_y) * (second_x - first_x) / safe_span
    crossing = (
        edge_valid
        & ((first_y > point_y) != (second_y > point_y))
        & (point_x < crossing_x)
    )
    inside = (jnp.sum(crossing, axis=1) % 2) == 1

    edge = edge_end - edge_start
    relative = point - edge_start
    length_squared = jnp.sum(edge * edge, axis=-1)
    safe_length = jnp.where(length_squared > 0.0, length_squared, 1.0)
    fraction = jnp.clip(jnp.sum(relative * edge, axis=-1) / safe_length, 0.0, 1.0)
    closest = edge_start + fraction[..., None] * edge
    separation = jnp.linalg.norm(point - closest, axis=-1)
    edge_distance = jnp.min(jnp.where(edge_valid, separation, jnp.inf), axis=1)
    return inside | (edge_distance <= distance_limit), inside, edge_distance


def _vertex_structure(machine: Any) -> dict[str, Any]:
    """Build the fixed vertex-centred dual stencil from the sampling graph.

    Every authored sampling vertex of the hex carrier belongs to a fixed set of
    cells (the cells whose authored sampling ring carries the vertex) and a
    fixed set of adjacent vertices (the other ring corners reached by the edges
    of those cells).  Interior vertices of the hex tiling own exactly three
    incident cells and three adjacent vertices, which with the vertex value
    itself are the seven samples of the dual quadratic.  Boundary vertices own
    fewer and are excluded from the fit.
    """

    cell_sample = np.asarray(machine.moment_geometry.cell_sample_nodes, dtype=np.intp)
    n_cells, ring_size = cell_sample.shape
    if ring_size != 6:
        raise RuntimeError(
            f"the oracle carrier must author six-vertex rings, got {ring_size}"
        )
    cells = np.broadcast_to(
        np.arange(n_cells, dtype=np.intp)[:, None], cell_sample.shape
    )
    corners = np.broadcast_to(
        np.arange(ring_size, dtype=np.intp)[None, :], cell_sample.shape
    )
    flat = np.column_stack((cells.ravel(), corners.ravel(), cell_sample.ravel()))
    order = np.argsort(flat[:, 2], kind="stable")
    flat = flat[order]
    unique_vertices, starts = np.unique(flat[:, 2], return_index=True)
    boundaries = np.append(starts, len(flat))
    vertex_count = len(unique_vertices)
    incident_cells: list[np.ndarray] = []
    incident_neighbours: list[np.ndarray] = []
    excluded_reason: list[str] = []
    for index in range(vertex_count):
        rows = flat[boundaries[index] : boundaries[index + 1]]
        this_vertex = unique_vertices[index]
        cell_ids = rows[:, 0].tolist()
        neighbour_set: set[int] = set()
        for cell_id, corner in rows[:, :2].tolist():
            neighbour_set.add(int(cell_sample[cell_id, (corner - 1) % ring_size]))
            neighbour_set.add(int(cell_sample[cell_id, (corner + 1) % ring_size]))
        neighbour_set.discard(int(this_vertex))
        if len(cell_ids) != 3:
            excluded_reason.append(f"incident_cells={len(cell_ids)}")
        elif len(neighbour_set) != 3:
            excluded_reason.append(f"adjacent_vertices={len(neighbour_set)}")
        else:
            excluded_reason.append("")
        incident_cells.append(np.asarray(cell_ids, dtype=np.intp))
        incident_neighbours.append(np.asarray(sorted(neighbour_set), dtype=np.intp))

    counts = np.array([len(cells) for cells in incident_cells], dtype=np.intp)
    interior = counts == 3
    interior &= np.array([not reason for reason in excluded_reason], dtype=bool)
    vertex_ids = unique_vertices
    sample = np.asarray(machine.sample_coordinates, dtype=np.float64)
    node = np.asarray(machine.node, dtype=np.float64)
    selected = np.flatnonzero(interior)
    if len(selected) == 0:
        raise RuntimeError("the carrier exposes no three-cell dual vertices")
    centre = sample[vertex_ids[selected]]
    cell_ring = np.asarray([incident_cells[index] for index in selected], dtype=np.intp)
    neighbour_ring = np.asarray(
        [incident_neighbours[index] for index in selected], dtype=np.intp
    )
    centroid_point = node[cell_ring]
    neighbour_point = sample[vertex_ids[neighbour_ring]]
    ring_point = np.concatenate(
        (centre[:, None, :], centroid_point, neighbour_point), axis=1
    )
    offset = ring_point - centre[:, None, :]
    scale = np.max(np.abs(offset), axis=(1, 2))
    if np.any(scale <= 0.0):
        raise RuntimeError("every dual stencil must span both coordinate axes")
    local = offset / scale[:, None, None]
    quadratic = np.stack(
        [
            np.ones_like(local[..., 0]),
            local[..., 0],
            local[..., 1],
            local[..., 0] ** 2,
            local[..., 0] * local[..., 1],
            local[..., 1] ** 2,
        ],
        axis=-1,
    )
    condition = np.linalg.cond(quadratic)
    pathological = condition > RING_CONDITION_LIMIT
    if np.any(pathological):
        raise RuntimeError(
            f"{np.count_nonzero(pathological)} dual stencils exceed the "
            f"{RING_CONDITION_LIMIT:.0f} conditioning limit"
        )
    return {
        "vertex_id": vertex_ids[selected],
        "centre": centre,
        "cell_ring": cell_ring,
        "neighbour_ring": neighbour_ring,
        "centre_index_in_pool": len(node) + vertex_ids[selected],
        "gather": np.column_stack(
            (
                len(node) + vertex_ids[selected],
                cell_ring,
                len(node) + vertex_ids[neighbour_ring],
            )
        ),
        "ring_point": ring_point,
        "ring_coordinate_scale": scale,
        "ring_condition": condition,
        "triangle_point": centroid_point,
        "vertex_incident_count_histogram": np.bincount(counts, minlength=4).tolist(),
        "vertex_count": vertex_count,
        "interior_vertex_count": len(selected),
        "boundary_vertex_count": int(vertex_count - len(selected)),
    }


def _primary_read_function(operator: Any, machine: Any, pitch: float):
    """Build the jitted cell-centred census over the own-node stencil."""

    stencil = _support_stencil(operator)
    gather = jnp.asarray(stencil.ring_gather_index, dtype=jnp.int32)
    weight = jnp.asarray(stencil.ring_flux_weight, dtype=jnp.float64)
    centre = jnp.asarray(stencil.ring_sampling_centre, dtype=jnp.float64)
    scale = jnp.asarray(stencil.ring_coordinate_scale, dtype=jnp.float64)
    edge_start, edge_end, edge_valid = _cell_edges(machine.cell_polygons)
    edge_start = jnp.asarray(edge_start)
    edge_end = jnp.asarray(edge_end)
    edge_valid = jnp.asarray(edge_valid)
    axis_kind = operator.polarity
    if axis_kind is None:
        raise RuntimeError("the production read has no declared axis polarity")

    def read(state: jax.Array) -> dict[str, jax.Array]:
        physical = jnp.asarray(
            state[: operator.physical_node_number], dtype=jnp.float64
        )
        centroid, _wall = operator._fixed_design_topology.split_flux_map(physical)
        sample = state[operator.physical_node_number :]
        pool = jnp.concatenate((centroid, sample))
        values = pool[gather]
        delta = values[:, 1:] - values[:, :1]
        above = delta > 0.0
        crossing = jnp.sum(above != jnp.roll(above, -1, axis=1), axis=1)
        common_sign = jnp.all(delta > 0.0, axis=1) | jnp.all(delta < 0.0, axis=1)
        raw_extremum = (crossing == 0) & common_sign
        raw_saddle = crossing == 4

        coefficient = jnp.einsum("nps,ns->np", weight, values)
        h00 = 2.0 * coefficient[:, 3]
        h01 = coefficient[:, 4]
        h11 = 2.0 * coefficient[:, 5]
        determinant = h00 * h11 - h01 * h01
        nonsingular = jnp.abs(determinant) > 1.0e-12
        safe_determinant = jnp.where(nonsingular, determinant, 1.0)
        local_radial = (
            h01 * coefficient[:, 2] - h11 * coefficient[:, 1]
        ) / safe_determinant
        local_vertical = (
            h01 * coefficient[:, 1] - h00 * coefficient[:, 2]
        ) / safe_determinant
        step = jnp.stack((local_radial, local_vertical), axis=1) * scale
        position = centre + step
        local = step / scale
        value = (
            coefficient[:, 0]
            + coefficient[:, 1] * local[:, 0]
            + coefficient[:, 2] * local[:, 1]
            + coefficient[:, 3] * local[:, 0] ** 2
            + coefficient[:, 4] * local[:, 0] * local[:, 1]
            + coefficient[:, 5] * local[:, 1] ** 2
        )
        finite = nonsingular & jnp.all(jnp.isfinite(position), axis=1)
        near_cell, inside_cell, cell_distance = _inside_or_near_cell(
            position, edge_start, edge_end, edge_valid, EDGE_TOLERANCE_IN_PITCH * pitch
        )
        saddle_type = determinant < -1.0e-12
        extremum_type = (determinant > 1.0e-12) & (
            jnp.where(h00 + h11 < 0.0, 1, -1) == axis_kind
        )
        extremal_centroid = jnp.argmax(operator.polarity * centroid)
        axis_seed = jnp.arange(centroid.shape[0]) == extremal_centroid
        stationary_saddle = finite & saddle_type
        stationary_extremum = finite & axis_seed & extremum_type
        typed_saddle = stationary_saddle & near_cell
        typed_extremum = stationary_extremum & near_cell
        return {
            "raw_extremum": raw_extremum,
            "raw_saddle": raw_saddle,
            "stationary_saddle": stationary_saddle,
            "stationary_extremum": stationary_extremum,
            "typed_saddle": typed_saddle,
            "typed_extremum": typed_extremum,
            "retained_position": position,
            "retained_value": value,
            "stationary_determinant": determinant,
            "inside_region": inside_cell,
            "distance_to_region_m": cell_distance,
            "axis_seed": axis_seed,
        }

    return jax.jit(read)


def _dual_read_function(
    operator: Any, structure: dict[str, Any], pitch: float, cell_vertices: np.ndarray
):
    """Build the jitted vertex-centred census over the dual stencil."""

    gather = jnp.asarray(structure["gather"], dtype=jnp.int32)
    cell_vertices = jnp.asarray(cell_vertices, dtype=jnp.int32)
    local_point = np.asarray(structure["ring_point"], dtype=np.float64)
    scale = np.asarray(structure["ring_coordinate_scale"], dtype=np.float64)
    centre = np.asarray(structure["centre"], dtype=np.float64)
    normalised = (local_point - centre[:, None, :]) / scale[:, None, None]
    design = np.stack(
        [
            np.ones_like(normalised[..., 0]),
            normalised[..., 0],
            normalised[..., 1],
            normalised[..., 0] ** 2,
            normalised[..., 0] * normalised[..., 1],
            normalised[..., 1] ** 2,
        ],
        axis=-1,
    )
    weight = np.linalg.pinv(design)
    weight = jnp.asarray(weight, dtype=jnp.float64)
    triangle = np.asarray(structure["triangle_point"], dtype=np.float64)
    tri_start = jnp.asarray(triangle)
    tri_end = jnp.roll(tri_start, -1, axis=1)
    tri_valid = jnp.ones((len(triangle), 3), dtype=bool)
    axis_kind = operator.polarity
    if axis_kind is None:
        raise RuntimeError("the production read has no declared axis polarity")

    def read(state: jax.Array) -> dict[str, jax.Array]:
        physical = jnp.asarray(
            state[: operator.physical_node_number], dtype=jnp.float64
        )
        centroid, _wall = operator._fixed_design_topology.split_flux_map(physical)
        sample = state[operator.physical_node_number :]
        pool = jnp.concatenate((centroid, sample))
        values = pool[gather]
        delta = values[:, 1:] - values[:, :1]
        above = delta > 0.0
        crossing = jnp.sum(above != jnp.roll(above, -1, axis=1), axis=1)
        common_sign = jnp.all(delta > 0.0, axis=1) | jnp.all(delta < 0.0, axis=1)
        raw_extremum = (crossing == 0) & common_sign
        raw_saddle = crossing == 4

        coefficient = jnp.einsum("vps,vs->vp", weight, values)
        h00 = 2.0 * coefficient[:, 3]
        h01 = coefficient[:, 4]
        h11 = 2.0 * coefficient[:, 5]
        determinant = h00 * h11 - h01 * h01
        nonsingular = jnp.abs(determinant) > 1.0e-12
        safe_determinant = jnp.where(nonsingular, determinant, 1.0)
        local_radial = (
            h01 * coefficient[:, 2] - h11 * coefficient[:, 1]
        ) / safe_determinant
        local_vertical = (
            h01 * coefficient[:, 1] - h00 * coefficient[:, 2]
        ) / safe_determinant
        step = jnp.stack((local_radial, local_vertical), axis=1) * scale[:, None]
        position = centre + step
        local = step / scale[:, None]
        value = (
            coefficient[:, 0]
            + coefficient[:, 1] * local[:, 0]
            + coefficient[:, 2] * local[:, 1]
            + coefficient[:, 3] * local[:, 0] ** 2
            + coefficient[:, 4] * local[:, 0] * local[:, 1]
            + coefficient[:, 5] * local[:, 1] ** 2
        )
        finite = nonsingular & jnp.all(jnp.isfinite(position), axis=1)
        near_triangle, inside_triangle, triangle_distance = _inside_or_near_cell(
            position, tri_start, tri_end, tri_valid, EDGE_TOLERANCE_IN_PITCH * pitch
        )
        saddle_type = determinant < -1.0e-12
        extremum_type = (determinant > 1.0e-12) & (
            jnp.where(h00 + h11 < 0.0, 1, -1) == axis_kind
        )
        extremal_centroid = jnp.argmax(operator.polarity * centroid)
        extremal_vertices = cell_vertices[extremal_centroid]
        axis_seed = jnp.isin(
            gather[:, 0] - operator.grid.node_number, extremal_vertices
        )
        stationary_saddle = finite & saddle_type
        stationary_extremum = finite & axis_seed & extremum_type
        typed_saddle = stationary_saddle & near_triangle
        typed_extremum = stationary_extremum & near_triangle
        return {
            "raw_extremum": raw_extremum,
            "raw_saddle": raw_saddle,
            "stationary_saddle": stationary_saddle,
            "stationary_extremum": stationary_extremum,
            "typed_saddle": typed_saddle,
            "typed_extremum": typed_extremum,
            "retained_position": position,
            "retained_value": value,
            "stationary_determinant": determinant,
            "inside_region": inside_triangle,
            "distance_to_region_m": triangle_distance,
            "axis_seed": axis_seed,
        }

    return jax.jit(read)


def _selected_error(
    rows: np.ndarray,
    reference_position: np.ndarray,
    reference_flux: float,
    pitch: float,
    span: float,
) -> dict[str, Any]:
    """Measure the candidate nearest one analytic stationary point."""

    if not len(rows):
        return {
            "admitted": False,
            "position_rz_m": None,
            "flux_wb": None,
            "position_error_m": None,
            "position_error_in_pitch": None,
            "level_error_wb": None,
            "level_error_in_span": None,
        }
    distances = np.linalg.norm(rows[:, :2] - reference_position, axis=1)
    selected = rows[int(np.argmin(distances))]
    position_error = float(np.min(distances))
    level_error = abs(float(selected[2]) - reference_flux)
    return {
        "admitted": bool(position_error <= pitch),
        "position_rz_m": selected[:2].tolist(),
        "flux_wb": float(selected[2]),
        "position_error_m": position_error,
        "position_error_in_pitch": position_error / pitch,
        "level_error_wb": level_error,
        "level_error_in_span": level_error / span,
    }


def _true_cell_mask(machine: Any, point: np.ndarray, pitch: float) -> np.ndarray:
    """Mark cells geometrically containing a reference point."""

    contained = np.asarray(
        [
            PolygonPath(np.asarray(polygon)).contains_point(
                point, radius=64.0 * np.finfo(np.float64).eps
            )
            for polygon in machine.cell_polygons
        ],
        dtype=bool,
    )
    if not np.any(contained):
        nearest = int(np.argmin(np.linalg.norm(machine.node - point, axis=1)))
        if np.linalg.norm(machine.node[nearest] - point) <= pitch:
            contained[nearest] = True
    return contained


def _point_edge_distance_numpy(
    point: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    edge_valid: np.ndarray,
) -> float:
    """Return the distance from one point to the closest directed edge."""

    edge = edge_end - edge_start
    relative = point - edge_start
    length_squared = np.sum(edge * edge, axis=-1)
    safe = np.where(length_squared > 0.0, length_squared, 1.0)
    fraction = np.clip(np.sum(relative * edge, axis=-1) / safe, 0.0, 1.0)
    closest = edge_start + fraction[..., None] * edge
    separation = np.linalg.norm(point - closest, axis=-1)
    separation = np.where(edge_valid, separation, np.inf)
    return float(np.min(separation))


def _vertice_regions_containing(
    structure: dict[str, Any], point: np.ndarray, pitch: float
) -> np.ndarray:
    """Mark each dual vertex whose centroid triangle contains a reference point."""

    triangle = structure["triangle_point"]
    start = triangle
    end = np.roll(triangle, -1, axis=1)
    relative = point[None, None, :] - start
    edge = end - start
    length_squared = np.sum(edge * edge, axis=-1)
    safe = np.where(length_squared > 0.0, length_squared, 1.0)
    fraction = np.clip(np.sum(relative * edge, axis=-1) / safe, 0.0, 1.0)
    closest = start + fraction[..., None] * edge
    separation = np.linalg.norm(point[None, None, :] - closest, axis=-1)
    edge_distance = np.min(separation, axis=1)

    vertex_0 = triangle[:, 1] - triangle[:, 0]
    vertex_1 = triangle[:, 2] - triangle[:, 0]
    point_2 = point[None, :] - triangle[:, 0]
    d00 = np.sum(vertex_0 * vertex_0, axis=1)
    d01 = np.sum(vertex_0 * vertex_1, axis=1)
    d11 = np.sum(vertex_1 * vertex_1, axis=1)
    d20 = np.sum(point_2 * vertex_0, axis=1)
    d21 = np.sum(point_2 * vertex_1, axis=1)
    denominator = d00 * d11 - d01 * d01
    safe_denominator = np.where(np.abs(denominator) > 0.0, denominator, 1.0)
    vertical = (d11 * d20 - d01 * d21) / safe_denominator
    horizontal = (d00 * d21 - d01 * d20) / safe_denominator
    remaining = 1.0 - vertical - horizontal
    inside = (remaining >= 0.0) & (vertical >= 0.0) & (horizontal >= 0.0)
    return inside | (edge_distance <= EDGE_TOLERANCE_IN_PITCH * pitch)


def _null_boundary_proximity(
    machine: Any,
    structure: dict[str, Any] | None,
    point: np.ndarray,
    pitch: float,
) -> dict[str, Any]:
    """Measure how close an analytic null sits to cells and vertices."""

    start, end, valid = _cell_edges(machine.cell_polygons)
    edge_distance = _point_edge_distance_numpy(point, start, end, valid)
    vertex_distance = float(
        np.min(np.linalg.norm(np.asarray(machine.sample_coordinates) - point, axis=1))
    )
    return {
        "distance_to_nearest_cell_edge_m": edge_distance,
        "distance_to_nearest_cell_edge_in_pitch": edge_distance / pitch,
        "distance_to_nearest_vertex_m": vertex_distance,
        "distance_to_nearest_vertex_in_pitch": vertex_distance / pitch,
    }


def _manufactured_controls(
    machine: Any,
    operator: Any,
    structure: dict[str, Any],
    primary_read: Any,
    dual_read: Any,
    pitch: float,
) -> dict[str, Any]:
    """Show both censuses polish exact quadratic stationary points to 1e-10."""

    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    domain_centre = np.mean(machine.node, axis=0)
    primary_cell = int(np.argmin(np.linalg.norm(machine.node - domain_centre, axis=1)))
    interior_ids = set(int(vertex) for vertex in structure["vertex_id"])
    cell_vertices = np.asarray(machine.moment_geometry.cell_sample_nodes)
    target_vertex = next(
        (
            int(vertex)
            for vertex in cell_vertices[primary_cell].tolist()
            if int(vertex) in interior_ids
        ),
        None,
    )
    if target_vertex is None:
        raise RuntimeError("the control cell has no interior dual vertex")
    axis_kind = operator.polarity

    def manufactured(target: np.ndarray, saddle_state: bool) -> np.ndarray:
        local = (coordinates - target) / pitch
        if saddle_state:
            return local[:, 0] ** 2 - local[:, 1] ** 2
        return -float(axis_kind) * (local[:, 0] ** 2 + local[:, 1] ** 2)

    def probe(
        target: np.ndarray,
        output: dict[str, Any],
        index: int,
        detection_key: str,
    ) -> dict[str, Any]:
        position = np.asarray(output["retained_position"])[index]
        return {
            "target_rz_m": target.tolist(),
            "quadratic_detected": bool(np.asarray(output[detection_key])[index]),
            "polished_position_error_m": float(np.linalg.norm(position - target)),
        }

    primary_saddle = jax.block_until_ready(
        primary_read(jnp.asarray(manufactured(machine.node[primary_cell], True)))
    )
    primary_extremum = jax.block_until_ready(
        primary_read(jnp.asarray(manufactured(machine.node[primary_cell], False)))
    )
    target_vertex_position = np.asarray(machine.sample_coordinates[target_vertex])
    dual_saddle = jax.block_until_ready(
        dual_read(jnp.asarray(manufactured(target_vertex_position, True)))
    )
    dual_extremum = jax.block_until_ready(
        dual_read(jnp.asarray(manufactured(target_vertex_position, False)))
    )
    vertex_index = int(np.flatnonzero(structure["vertex_id"] == target_vertex)[0])
    return {
        "primary_cell": primary_cell,
        "dual_vertex_id": int(target_vertex),
        "dual_vertex_index_in_stencil": vertex_index,
        "primary_saddle": probe(
            machine.node[primary_cell],
            primary_saddle,
            primary_cell,
            "stationary_saddle",
        ),
        "primary_extremum": probe(
            machine.node[primary_cell],
            primary_extremum,
            primary_cell,
            "stationary_extremum",
        ),
        "dual_saddle": probe(
            target_vertex_position, dual_saddle, vertex_index, "stationary_saddle"
        ),
        "dual_extremum": probe(
            target_vertex_position,
            dual_extremum,
            vertex_index,
            "stationary_extremum",
        ),
    }


def _smooth_perturbation(coordinates: np.ndarray, span: float) -> np.ndarray:
    """Return deterministic smooth noise with peak amplitude one ten-thousandth span."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    signal = np.sin(1.7 * coordinates[:, 0] + 0.3 * coordinates[:, 1]) + 0.41 * np.cos(
        0.6 * coordinates[:, 0] - 1.3 * coordinates[:, 1]
    )
    signal /= np.max(np.abs(signal))
    return 1.0e-4 * span * signal


def _measure_row(
    case_name: str, requested_cells: int, report_directory: Path
) -> dict[str, Any]:
    """Measure and persist one analytic-flux carrier row."""

    part_path = _part_path(report_directory, case_name, requested_cells)
    progress = {
        "schema": "nova.dual-stencil-census-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "completed": False,
    }
    _write_json(part_path, progress)
    started = perf_counter()
    machine, operator, state, exact = _machine_and_field(case_name, requested_cells)
    pitch = math.sqrt(float(np.median(np.asarray(machine.area, dtype=np.float64))))
    axis_reference = np.asarray(exact.magnetic_axis, dtype=np.float64)
    axis_flux = float(
        limiter_audit._exact_flux(case_name, exact, axis_reference[None, :])[0]
    )
    diverted = certificate._is_diverted_case(case_name)
    x_reference = np.asarray(exact.x_point, dtype=np.float64) if diverted else None
    x_flux = (
        float(limiter_audit._exact_flux(case_name, exact, x_reference[None, :])[0])
        if diverted
        else None
    )
    grid_span = float(np.ptp(state[: len(machine.node)]))
    span = abs(axis_flux - x_flux) if x_flux is not None else grid_span
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("the analytic reference span is not positive")
    structure = _vertex_structure(machine)
    cell_vertices = np.asarray(
        machine.moment_geometry.cell_sample_nodes, dtype=np.int32
    )
    primary_read = _primary_read_function(operator, machine, pitch)
    dual_read = _dual_read_function(operator, structure, pitch, cell_vertices)
    operand = jnp.asarray(state, dtype=jnp.float64)
    primary = jax.block_until_ready(primary_read(operand))
    dual = jax.block_until_ready(dual_read(operand))

    def candidate_rows(output: dict[str, Any], kind: str) -> np.ndarray:
        mask = np.asarray(output[f"typed_{kind}"], dtype=bool)
        position = np.asarray(output["retained_position"], dtype=np.float64)[mask]
        value = np.asarray(output["retained_value"], dtype=np.float64)[mask]
        return np.concatenate((position, value[:, None]), axis=1)

    def null_match(
        output: dict[str, Any],
        reference: np.ndarray,
        reference_flux: float,
    ) -> dict[str, Any]:
        return {
            "saddle": _selected_error(
                candidate_rows(output, "saddle"),
                reference,
                reference_flux,
                pitch,
                span,
            ),
            "extremum": _selected_error(
                candidate_rows(output, "extremum"),
                reference,
                reference_flux,
                pitch,
                span,
            ),
        }

    primary_axis = null_match(primary, axis_reference, axis_flux)
    dual_axis = null_match(dual, axis_reference, axis_flux)
    primary_x = null_match(primary, x_reference, x_flux) if diverted else None
    dual_x = null_match(dual, x_reference, x_flux) if diverted else None

    def polished_difference(
        first: dict[str, Any], second: dict[str, Any]
    ) -> dict[str, Any] | None:
        if (
            first["saddle"]["admitted"]
            and second["saddle"]["admitted"]
            and first["saddle"]["position_rz_m"] is not None
            and second["saddle"]["position_rz_m"] is not None
        ):
            first_position = np.asarray(first["saddle"]["position_rz_m"])
            second_position = np.asarray(second["saddle"]["position_rz_m"])
            separation = float(np.linalg.norm(first_position - second_position))
            return {
                "separation_m": separation,
                "separation_in_pitch": separation / pitch,
            }
        return None

    axis_difference = polished_difference(primary_axis, dual_axis)
    x_difference = polished_difference(primary_x, dual_x) if diverted else None
    axis_proximity = _null_boundary_proximity(machine, structure, axis_reference, pitch)
    x_proximity = (
        _null_boundary_proximity(machine, structure, x_reference, pitch)
        if diverted
        else None
    )
    instrument = _manufactured_controls(
        machine, operator, structure, primary_read, dual_read, pitch
    )
    if not (
        instrument["primary_saddle"]["polished_position_error_m"] <= 1.0e-10
        and instrument["primary_extremum"]["polished_position_error_m"] <= 1.0e-10
        and instrument["dual_saddle"]["polished_position_error_m"] <= 1.0e-10
        and instrument["dual_extremum"]["polished_position_error_m"] <= 1.0e-10
    ):
        raise RuntimeError("a manufactured stationary-point control failed")
    noise_control = None
    if diverted:
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        perturbed_state = state + _smooth_perturbation(coordinates, span)
        perturbed = jax.block_until_ready(
            primary_read(jnp.asarray(perturbed_state, dtype=jnp.float64))
        )
        perturbed_saddle = _selected_error(
            candidate_rows(perturbed, "saddle"), x_reference, x_flux, pitch, span
        )
        noise_control = {
            "amplitude_in_span": 1.0e-4,
            "saddle_admitted_against_reference": perturbed_saddle["admitted"],
            "saddle_position_error_in_pitch": perturbed_saddle[
                "position_error_in_pitch"
            ],
        }

    def false_candidates(
        measured: dict[str, Any],
        axis_true_mask: np.ndarray,
        x_true_mask: np.ndarray,
    ) -> dict[str, Any]:
        raw_extremum = np.asarray(measured["raw_extremum"], dtype=bool)
        raw_saddle = np.asarray(measured["raw_saddle"], dtype=bool)
        stationary_extremum = np.asarray(measured["stationary_extremum"], dtype=bool)
        stationary_saddle = np.asarray(measured["stationary_saddle"], dtype=bool)
        typed_extremum = np.asarray(measured["typed_extremum"], dtype=bool)
        typed_saddle = np.asarray(measured["typed_saddle"], dtype=bool)
        saddles = candidate_rows(measured, "saddle")
        extrema = candidate_rows(measured, "extremum")
        return {
            "before_hessian_and_containment": {
                "saddle": int(
                    np.count_nonzero(raw_saddle & ~np.asarray(x_true_mask, dtype=bool))
                ),
                "extremum": int(
                    np.count_nonzero(
                        raw_extremum & ~np.asarray(axis_true_mask, dtype=bool)
                    )
                ),
            },
            "after_hessian_only": {
                "saddle": int(
                    np.count_nonzero(
                        stationary_saddle & ~np.asarray(x_true_mask, dtype=bool)
                    )
                ),
                "extremum": int(
                    np.count_nonzero(
                        stationary_extremum & ~np.asarray(axis_true_mask, dtype=bool)
                    )
                ),
            },
            "after_hessian_and_containment": {
                "saddle": int(np.count_nonzero(typed_saddle)),
                "extremum": int(np.count_nonzero(typed_extremum)),
            },
            "retained_rows": {
                "saddle": len(saddles),
                "extremum": len(extrema),
            },
        }

    primary_axis_true = _true_cell_mask(machine, axis_reference, pitch)
    primary_x_true = (
        _true_cell_mask(machine, x_reference, pitch)
        if x_reference is not None
        else np.zeros(len(machine.node), dtype=bool)
    )
    dual_axis_true = _vertice_regions_containing(structure, axis_reference, pitch)
    dual_x_true = (
        _vertice_regions_containing(structure, x_reference, pitch)
        if x_reference is not None
        else np.zeros(len(structure["vertex_id"]), dtype=bool)
    )
    row = progress | {
        "allocation": _allocation("cpu"),
        "cache": {
            "semantic_key": machine.cache.get("semantic_key"),
            "hit": machine.cache.get("hit"),
            "build_seconds": machine.cache.get("build_seconds"),
        },
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "vertex_structure": structure,
        "analytic": {
            "axis_rz_m": axis_reference.tolist(),
            "axis_flux_wb": axis_flux,
            "x_point_rz_m": x_reference.tolist() if x_reference is not None else None,
            "x_point_flux_wb": x_flux,
            "reference_span_wb": span,
            "axis_proximity": axis_proximity,
            "x_proximity": x_proximity,
        },
        "primary_cell_census": {
            "criteria": {
                "quadratic_stationary_point": {
                    "axis_admitted": primary_axis["extremum"]["admitted"],
                    "saddle_admitted": primary_x["saddle"]["admitted"]
                    if primary_x
                    else None,
                }
            },
            "axis": primary_axis["extremum"],
            "x_point_saddle": primary_x["saddle"] if primary_x else None,
            "counts": false_candidates(primary, primary_axis_true, primary_x_true),
            "retained_saddle_rows": candidate_rows(primary, "saddle").tolist(),
            "retained_extremum_rows": candidate_rows(primary, "extremum").tolist(),
            "null_position_difference_in_pitch": {
                "axis": axis_difference,
                "x_point": x_difference,
            },
        },
        "dual_vertex_census": {
            "criteria": {
                "quadratic_stationary_point": {
                    "axis_admitted": dual_axis["extremum"]["admitted"],
                    "saddle_admitted": dual_x["saddle"]["admitted"] if dual_x else None,
                }
            },
            "axis": dual_axis["extremum"],
            "x_point_saddle": dual_x["saddle"] if dual_x else None,
            "counts": false_candidates(dual, dual_axis_true, dual_x_true),
            "retained_saddle_rows": candidate_rows(dual, "saddle").tolist(),
            "retained_extremum_rows": candidate_rows(dual, "extremum").tolist(),
            "null_position_difference_in_pitch": {
                "axis": axis_difference,
                "x_point": x_difference,
            },
        },
        "instrument_controls": instrument,
        "smooth_noise_control": noise_control,
        "wall_seconds": perf_counter() - started,
        "completed": True,
    }
    _write_json(part_path, row)
    primary_saddle_admitted = primary_x["saddle"]["admitted"] if primary_x else None
    dual_saddle_admitted = dual_x["saddle"]["admitted"] if dual_x else None
    print(
        "DUAL_STENCIL_ROW "
        f"case={case_name} requested={requested_cells} realised={len(machine.node)} "
        f"interior_vertices={structure['interior_vertex_count']} "
        f"primary_axis={primary_axis['extremum']['admitted']} "
        f"primary_saddle={primary_saddle_admitted} "
        f"dual_saddle={dual_saddle_admitted} "
        f"seconds={row['wall_seconds']:.3f}",
        flush=True,
    )
    return row


def _run_worker(report_directory: Path, shard_index: int, shard_count: int) -> None:
    """Measure one deterministic shard inside the shared CPU allocation."""

    _allocation("cpu")
    for case_name, requested_cells in CPU_ROWS[shard_index::shard_count]:
        _measure_row(case_name, requested_cells, report_directory)


def _render_error_boundary_figure(
    rows: list[dict[str, Any]], figure_directory: Path
) -> dict[str, str]:
    """Draw both censuses' position error against null-to-boundary distance."""

    observations: list[dict[str, Any]] = []
    for row in rows:
        analytic = row["analytic"]
        for null_name, primary, dual, proximity in (
            (
                "axis",
                row["primary_cell_census"]["axis"],
                row["dual_vertex_census"]["axis"],
                analytic["axis_proximity"],
            ),
            (
                "x_point",
                row["primary_cell_census"]["x_point_saddle"],
                row["dual_vertex_census"]["x_point_saddle"],
                analytic.get("x_proximity"),
            ),
        ):
            if dual is None:
                continue
            if (
                primary["position_error_in_pitch"] is None
                or dual["position_error_in_pitch"] is None
            ):
                continue
            observations.append(
                {
                    "realised_cells": int(row["realised_cells"]),
                    "null": null_name,
                    "distance_to_boundary_in_pitch": float(
                        proximity["distance_to_nearest_cell_edge_in_pitch"]
                    ),
                    "primary_error_in_pitch": primary["position_error_in_pitch"],
                    "dual_error_in_pitch": dual["position_error_in_pitch"],
                    "primary_admitted": bool(primary["admitted"]),
                    "dual_admitted": bool(dual["admitted"]),
                }
            )
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for null_name, marker in (("axis", "o"), ("x_point", "s")):
        subset = [item for item in observations if item["null"] == null_name]
        axis.scatter(
            [item["distance_to_boundary_in_pitch"] for item in subset],
            [item["primary_error_in_pitch"] for item in subset],
            marker=marker,
            facecolor=COLOURS["primary"],
            edgecolor="white",
            s=44,
            label=f"cell-centred census · {null_name}"
            if null_name == "axis"
            else "cell-centred census · X-point",
        )
        axis.scatter(
            [item["distance_to_boundary_in_pitch"] for item in subset],
            [item["dual_error_in_pitch"] for item in subset],
            marker=marker,
            facecolor=COLOURS["dual"],
            edgecolor="white",
            s=44,
            label=f"vertex-centred census · {null_name}"
            if null_name == "axis"
            else "vertex-centred census · X-point",
        )
        for item in subset:
            primary_admitted = item["primary_admitted"]
            dual_admitted = item["dual_admitted"]
            axis.plot(
                [item["distance_to_boundary_in_pitch"]] * 2,
                [item["primary_error_in_pitch"], item["dual_error_in_pitch"]],
                color="#bbbbbb",
                linewidth=0.7,
                zorder=1,
            )
            if not primary_admitted:
                axis.scatter(
                    [item["distance_to_boundary_in_pitch"]],
                    [item["primary_error_in_pitch"]],
                    marker="x",
                    color=COLOURS["primary"],
                    s=60,
                )
            if not dual_admitted:
                axis.scatter(
                    [item["distance_to_boundary_in_pitch"]],
                    [item["dual_error_in_pitch"]],
                    marker="x",
                    color=COLOURS["dual"],
                    s=60,
                )
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set_yscale("log")
    axis.set_xlabel("analytic null distance to nearest cell edge / pitch")
    axis.set_ylabel("census position error / pitch")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / "dual-stencil-error-by-boundary-distance.svg"
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/dual-stencil/"
            "dual-stencil-error-by-boundary-distance.svg"
        ),
    }


def _write_report(receipt: dict[str, Any], destination: Path) -> None:
    """Write the human-readable admission, error, boundary, and bin tables."""

    rows = receipt["single_null_rows"]
    lines = [
        "# Dual-stencil stationary-point census",
        "",
        (
            "Two censuses read the analytic single-null flux on the same cached "
            "carriers.  The cell-centred census fits a quadratic on each centroid "
            "and its six sampling vertices; the vertex-centred dual fits a "
            "quadratic on every mesh vertex, its three surrounding cell centroids "
            "and its three adjacent vertices.  Each reports the closed-form "
            "stationary point and Hessian class, admitted inside or within "
            f"{EDGE_TOLERANCE_IN_PITCH} pitch of the owned region (the cell polygon "
            "for the cell census, the triangle of the three surrounding centroids "
            "for the vertex census)."
        ),
        "",
        "## Analytic single-null ladder (X-point)",
        "",
        (
            "| requested | realised | edge_dist/pitch | vertex_dist/pitch | "
            "cell_admitted | dual_admitted | cell_error_m | cell_error/pitch | "
            "dual_error_m | dual_error/pitch | position_difference/pitch |"
        ),
        "|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        primary = row["primary_cell_census"]
        dual = row["dual_vertex_census"]
        advisory = row["analytic"]["x_proximity"]
        primary_admitted = primary["x_point_saddle"]["admitted"]
        dual_admitted = dual["x_point_saddle"]["admitted"]
        primary_error = primary["x_point_saddle"]["position_error_m"]
        dual_error = dual["x_point_saddle"]["position_error_m"]
        difference = primary["null_position_difference_in_pitch"]["x_point"]
        difference_cell = difference["separation_in_pitch"] if difference else "—"
        lines.append(
            f"| {row['requested_cells']} | {row['realised_cells']} | "
            f"{advisory['distance_to_nearest_cell_edge_in_pitch']:.4g} | "
            f"{advisory['distance_to_nearest_vertex_in_pitch']:.4g} | "
            f"{'yes' if primary_admitted else 'no'} | "
            f"{'yes' if dual_admitted else 'no'} | {primary_error} | "
            f"{primary['x_point_saddle']['position_error_in_pitch']} | "
            f"{dual_error} | {dual['x_point_saddle']['position_error_in_pitch']} | "
            f"{difference_cell} |"
        )
    lines.extend(["", "## Axis admission and position error", ""])
    lines.append(
        "| case | requested | realised | edge_dist/pitch | cell_admitted | "
        "dual_admitted | cell_error/pitch | dual_error/pitch | "
        "position_difference/pitch |"
    )
    lines.append("|:---|---:|---:|---:|:---:|:---:|---:|---:|---:|")
    for row in receipt["single_null_rows"] + receipt["static_rows"]:
        primary = row["primary_cell_census"]
        dual = row["dual_vertex_census"]
        advisory = row["analytic"]["axis_proximity"]
        difference = primary["null_position_difference_in_pitch"]["axis"]
        difference_cell = difference["separation_in_pitch"] if difference else "—"
        lines.append(
            f"| {row['case']} | {row['requested_cells']} | {row['realised_cells']} | "
            f"{advisory['distance_to_nearest_cell_edge_in_pitch']:.4g} | "
            f"{'yes' if primary['axis']['admitted'] else 'no'} | "
            f"{'yes' if dual['axis']['admitted'] else 'no'} | "
            f"{primary['axis']['position_error_in_pitch']} | "
            f"{dual['axis']['position_error_in_pitch']} | {difference_cell} |"
        )
    lines.extend(["", "## False candidates before and after the filters", ""])
    lines.append(
        "| case | requested | census | sign_saddle | hessian_saddle | "
        "contained_saddle | sign_extremum | hessian_extremum | contained_extremum |"
    )
    lines.append("|:---|---:|:---|---:|---:|---:|---:|---:|---:|")
    for row in rows + receipt["static_rows"]:
        for census in ("primary_cell_census", "dual_vertex_census"):
            counts = row[census]["counts"]
            before = counts["before_hessian_and_containment"]
            hessian = counts["after_hessian_only"]
            after = counts["after_hessian_and_containment"]
            lines.append(
                f"| {row['case']} | {row['requested_cells']} | "
                f"{'cell' if census == 'primary_cell_census' else 'vertex'} | "
                f"{before['saddle']} | {hessian['saddle']} | {after['saddle']} | "
                f"{before['extremum']} | {hessian['extremum']} | {after['extremum']} |"
            )
    lines.extend(["", "## Position error binned by null-to-boundary distance", ""])
    lines.append(
        "| distance bin (pitch) | nulls | cell mean error/pitch | "
        "dual mean error/pitch | fraction dual smaller |"
    )
    lines.append("|:---|---:|---:|---:|---:|")
    for bin_name, payload in receipt["boundary_bins"]["bins"].items():
        if payload["count"] == 0:
            lines.append(f"| {bin_name} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {bin_name} | {payload['count']} | "
            f"{payload['mean_primary_error_in_pitch']:.5f} | "
            f"{payload['mean_dual_error_in_pitch']:.5f} | "
            f"{payload['fraction_dual_smaller']:.2f} |"
        )
    bin_total = receipt["boundary_bins"]["total"]
    lines.extend(
        [
            "",
            (
                f"Across all {bin_total['nulls']} null observations the dual errors "
                f"were the smaller of the two on "
                f"{bin_total['dual_smaller']} "
                f"({bin_total['dual_smaller'] / max(bin_total['nulls'], 1):.0%})."
            ),
        ]
    )
    lines.extend(["", "## Controls and figures", ""])
    primary_controls = all(
        row["instrument_controls"]["primary_saddle"]["polished_position_error_m"]
        <= 1.0e-10
        and row["instrument_controls"]["primary_extremum"]["polished_position_error_m"]
        <= 1.0e-10
        for row in rows + receipt["static_rows"]
    )
    dual_controls = all(
        row["instrument_controls"]["dual_saddle"]["polished_position_error_m"]
        <= 1.0e-10
        and row["instrument_controls"]["dual_extremum"]["polished_position_error_m"]
        <= 1.0e-10
        for row in rows + receipt["static_rows"]
    )
    lines.append(
        "- Manufactured quadratic stationary points are polished to 1e-10 m by "
        f"both censuses: cell-centred {primary_controls}, vertex-centred "
        f"{dual_controls}."
    )
    noise_admissions = sum(
        row["smooth_noise_control"]["saddle_admitted_against_reference"]
        for row in rows
        if row["smooth_noise_control"] is not None
    )
    lines.append(
        "- Smooth perturbation amplitude is 1e-4 of the analytic axis-to-X span; "
        f"the cell-centred saddle admission survived on {noise_admissions} of "
        f"{len(rows)} rungs."
    )
    for figure in receipt["figures"]:
        lines.append(
            f"- [{Path(figure['path']).stem}]({figure['project_absolute_src']})"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _boundary_bins(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bin all null observations by their distance to the nearest cell edge."""

    observations: list[dict[str, Any]] = []
    for row in rows:
        primary = row["primary_cell_census"]
        dual = row["dual_vertex_census"]
        for null_name, primary_result, dual_result, proximity in (
            ("axis", primary["axis"], dual["axis"], row["analytic"]["axis_proximity"]),
            (
                "x_point",
                primary["x_point_saddle"],
                dual["x_point_saddle"],
                row["analytic"].get("x_proximity"),
            ),
        ):
            if dual_result is None:
                continue
            observations.append(
                {
                    "null": null_name,
                    "distance": float(
                        proximity["distance_to_nearest_cell_edge_in_pitch"]
                    ),
                    "primary_error": float(primary_result["position_error_in_pitch"])
                    if primary_result["position_error_in_pitch"] is not None
                    else math.inf,
                    "dual_error": float(dual_result["position_error_in_pitch"])
                    if dual_result["position_error_in_pitch"] is not None
                    else math.inf,
                    "primary_admitted": bool(primary_result["admitted"]),
                    "dual_admitted": bool(dual_result["admitted"]),
                }
            )
    edges = (0.0, 0.125, 0.25, 0.5, 1.0, math.inf)
    names = ("<0.125", "0.125–0.25", "0.25–0.5", "0.5–1", "≥1")
    bins: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        lower, upper = edges[index], edges[index + 1]
        selected = [item for item in observations if lower <= item["distance"] < upper]
        dual_smaller = sum(
            item["dual_error"] < item["primary_error"] for item in selected
        )
        bins[name] = {
            "count": len(selected),
            "mean_primary_error_in_pitch": float(
                np.mean([item["primary_error"] for item in selected])
            )
            if selected
            else None,
            "mean_dual_error_in_pitch": float(
                np.mean([item["dual_error"] for item in selected])
            )
            if selected
            else None,
            "fraction_dual_smaller": float(dual_smaller / len(selected))
            if selected
            else None,
        }
    return {
        "bins": bins,
        "total": {
            "nulls": len(observations),
            "dual_smaller": int(
                sum(item["dual_error"] < item["primary_error"] for item in observations)
            ),
        },
    }


def aggregate(
    report_directory: Path, figure_directory: Path, *, render_figures: bool
) -> dict[str, Any]:
    """Aggregate parts, require reproduced admissions, and publish evidence."""

    rows = [
        _load_part(_part_path(report_directory, *identity)) for identity in CPU_ROWS
    ]
    single = [row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME]
    static = [row for row in rows if row["case"] != certificate.DIVERTED_CASE_NAME]
    primary_saddle_admitted = sum(
        bool(row["primary_cell_census"]["x_point_saddle"]["admitted"]) for row in single
    )
    primary_axis_admitted = sum(
        bool(row["primary_cell_census"]["axis"]["admitted"]) for row in single + static
    )
    if primary_saddle_admitted != len(single):
        raise RuntimeError(
            "the cell-centred census did not reproduce the admitted analytic "
            f"saddles: {primary_saddle_admitted} of {len(single)}"
        )
    if primary_axis_admitted != len(single) + len(static):
        raise RuntimeError(
            "the cell-centred census did not reproduce the admitted analytic "
            f"axes: {primary_axis_admitted} of {len(single) + len(static)}"
        )
    instrumental = all(
        row["instrument_controls"]["primary_saddle"]["polished_position_error_m"]
        <= 1.0e-10
        and row["instrument_controls"]["primary_extremum"]["polished_position_error_m"]
        <= 1.0e-10
        and row["instrument_controls"]["dual_saddle"]["polished_position_error_m"]
        <= 1.0e-10
        and row["instrument_controls"]["dual_extremum"]["polished_position_error_m"]
        <= 1.0e-10
        for row in rows
    )
    if not instrumental:
        raise RuntimeError("a manufactured stationary-point control failed")
    if render_figures:
        figures = [_render_error_boundary_figure(single + static, figure_directory)]
    else:
        prior_receipt = report_directory / "receipt.json"
        figures = (
            json.loads(prior_receipt.read_text(encoding="utf-8")).get("figures", [])
            if prior_receipt.exists() and prior_receipt.stat().st_size <= 1_000_000
            else []
        )
    dual_saddle_admitted = sum(
        bool(row["dual_vertex_census"]["x_point_saddle"]["admitted"]) for row in single
    )
    dual_axis_admitted = sum(
        bool(row["dual_vertex_census"]["axis"]["admitted"]) for row in single + static
    )
    receipt = {
        "schema": "nova.dual-stencil-census",
        "version": 1,
        "source_revision": _source_revision(),
        "analytic_flux_supplied_directly": True,
        "headline": {
            "cell_census_saddle_admitted_rungs": primary_saddle_admitted,
            "dual_census_saddle_admitted_rungs": dual_saddle_admitted,
            "cell_census_axis_admitted_rows": primary_axis_admitted,
            "dual_census_axis_admitted_rows": dual_axis_admitted,
            "single_null_rung_count": len(single),
            "total_rows": len(rows),
            "dual_vertex_total": sum(
                row["vertex_structure"]["vertex_count"] for row in rows
            ),
            "dual_interior_vertex_total": sum(
                row["vertex_structure"]["interior_vertex_count"] for row in rows
            ),
        },
        "positive_controls": {
            "cell_census_reproduced_9_of_9_saddles": True,
            "cell_census_reproduced_15_of_15_axes": True,
            "manufactured_saddle_and_extremum_polished_every_row": True,
            "analytic_grid_flux_nonuniform_every_row": True,
        },
        "boundary_bins": _boundary_bins(single + static),
        "figures": figures,
        "single_null_rows": single,
        "static_rows": static,
    }
    _write_json(report_directory / "receipt.json", receipt)
    _write_report(receipt, report_directory / "report.md")
    _write_report(receipt, figure_directory / "report.md")
    print(
        "DUAL_STENCIL_AGGREGATE "
        f"cell_saddle={primary_saddle_admitted}/{len(single)} "
        f"dual_saddle={dual_saddle_admitted}/{len(single)} "
        f"cell_axis={primary_axis_admitted}/{len(single) + len(static)} "
        f"dual_axis={dual_axis_admitted}/{len(single) + len(static)}",
        flush=True,
    )
    return receipt


def run_cpu(report_directory: Path, figure_directory: Path, workers: int) -> None:
    """Run all rows in subprocess shards inside one CPU allocation."""

    allocation = _allocation("cpu")
    if workers < 1 or workers > allocation["allocated_cpus"]:
        raise ValueError("worker count must fit within the CPU allocation")
    report_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for shard_index in range(workers):
        stream = (report_directory / f"worker-{shard_index}.log").open(
            "w", encoding="utf-8"
        )
        streams.append(stream)
        environment = os.environ.copy()
        threads = max(1, allocation["allocated_cpus"] // workers)
        environment.update(
            {
                "OMP_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
            }
        )
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "cpu-worker",
                    "--report-directory",
                    str(report_directory),
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(workers),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        )
    failures = []
    for index, process in enumerate(processes):
        status = process.wait()
        streams[index].close()
        if status:
            failures.append((index, status))
    if failures:
        raise RuntimeError(f"CPU worker shards failed: {failures}")
    aggregate(report_directory, figure_directory, render_figures=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-run")
    cpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    cpu.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    cpu.add_argument("--workers", type=int, default=4)
    worker = commands.add_parser("cpu-worker")
    worker.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    aggregate_parser.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    aggregate_parser.add_argument("--render-figures", action="store_true")
    return parser


def main() -> None:
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("the benchmark requires JAX binary64")
    arguments = _parser().parse_args()
    if arguments.command == "cpu-run":
        run_cpu(
            arguments.report_directory, arguments.figure_directory, arguments.workers
        )
    elif arguments.command == "cpu-worker":
        _run_worker(
            arguments.report_directory, arguments.shard_index, arguments.shard_count
        )
    else:
        aggregate(
            arguments.report_directory,
            arguments.figure_directory,
            render_figures=arguments.render_figures,
        )


if __name__ == "__main__":
    main()
