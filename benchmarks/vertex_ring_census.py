#!/usr/bin/env python3
"""Evaluate an own-vertex-ring stationary-point census on hex carriers.

The benchmark is deliberately separate from the production topology read.  It
loads the same cached analytic carriers as the production resolution ladder
and samples every cell centroid and its six authored sampling vertices.  A
quadratic fit over those seven own-node values supplies the closed-form
stationary point and Hessian classification.  The original cyclic vertex signs
about the centroid value remain in the receipt as a negative control, beside
the equivalent signs about the fitted stationary value.  The emitted gate
labels are data identifiers only; the implementation names the mechanisms.

Receipt field glossary:
``raw`` is the sign-change census, ``typed`` adds a finite capped Newton root
and the expected Hessian type, ``contained`` applies the first-wall polygon,
``representative`` deduplicates roots within half a characteristic pitch, and
``qualified`` applies the production axis/private-component qualification.
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
from matplotlib.collections import PolyCollection
from matplotlib.path import Path as PolygonPath
import numpy as np

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/vertex-census"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/vertex-census"
)
PRODUCTION_RECEIPT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/"
    "topology-ladder/receipt.json"
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
TIMING_CELL_COUNTS = (500, 1000, 2500)
PRODUCTION_TIMING_MS = {500: 1.4, 1000: 4.9, 2500: 30.0}
WALL_NODE_COUNT = 121
TIMING_BATCH_SIZE = 16
TIMING_REPEATS = 7
ANALYTIC = certificate.DIVERTED_REFERENCE
ANALYTIC_AXIS = np.asarray(ANALYTIC.magnetic_axis, dtype=np.float64)
ANALYTIC_X = np.asarray(ANALYTIC.x_point, dtype=np.float64)
COLOURS = {
    "analytic": "#3366cc",
    "vertex": "#8a2be2",
    "production": "#d1495b",
    "positive": "#d97706",
    "negative": "#2563eb",
    "cells": "#777777",
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
    """Load one bounded complete receipt part."""

    if path.stat().st_size > 1_000_000:
        raise RuntimeError(f"part receipt exceeds one megabyte: {path}")
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
    elif kind == "gpu":
        if partition != "betelgeuse" or platforms != "cuda,cpu":
            raise RuntimeError(
                "expected betelgeuse and cuda,cpu, received "
                f"{partition!r} and {platforms!r}"
            )
        if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
            raise RuntimeError("the timing allocation must expose exactly one GPU")
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
        "devices": [str(device) for device in jax.devices()],
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


def _deduplicate(
    position: jax.Array,
    valid: jax.Array,
    distance: float,
    priority: jax.Array,
) -> jax.Array:
    """Keep the candidate most securely belonging to its cell in each cluster."""

    order = jnp.argsort(jnp.where(valid, priority, jnp.inf), stable=True)
    ordered_position = position[order]
    ordered_valid = valid[order]
    slot = jnp.arange(ordered_position.shape[0])

    def retain(index, representative):
        separation = jnp.linalg.norm(ordered_position - ordered_position[index], axis=1)
        has_parent = jnp.any(representative & (slot < index) & (separation < distance))
        return representative.at[index].set(ordered_valid[index] & ~has_parent)

    ordered_representative = jax.lax.fori_loop(
        0,
        ordered_position.shape[0],
        retain,
        jnp.zeros(ordered_position.shape[0], dtype=bool),
    )
    return jnp.zeros_like(ordered_representative).at[order].set(ordered_representative)


def _cell_edges(operator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-capacity directed edges for every physical cell polygon."""

    polygons = tuple(
        np.asarray(polygon, dtype=np.float64)
        for polygon in operator.moment_geometry.polygons
    )
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


def _vertex_read_function(operator: Any, pitch: float):
    """Build the jitted own-vertex census and production qualification."""

    stencil = _support_stencil(operator)
    gather = jnp.asarray(stencil.ring_gather_index, dtype=jnp.int32)
    weight = jnp.asarray(stencil.ring_flux_weight, dtype=jnp.float64)
    centre = jnp.asarray(stencil.ring_sampling_centre, dtype=jnp.float64)
    scale = jnp.asarray(stencil.ring_coordinate_scale, dtype=jnp.float64)
    edge_start, edge_end, edge_valid = _cell_edges(operator)
    edge_start = jnp.asarray(edge_start)
    edge_end = jnp.asarray(edge_end)
    edge_valid = jnp.asarray(edge_valid)
    capacity = int(operator._fixed_design_topology.grid.locator.maxsize)
    physical_count = int(operator.physical_node_number)
    axis_kind = operator._fixed_design_topology.grid.extremum_polarity
    if axis_kind is None:
        raise RuntimeError("the production read has no declared axis polarity")

    def read(state: jax.Array) -> dict[str, jax.Array]:
        state = jnp.asarray(state, dtype=jnp.float64)
        physical = state[:physical_count]
        centroid, wall_flux = operator._fixed_design_topology.split_flux_map(physical)
        sample = state[physical_count:]
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
        requested_distance = jnp.linalg.norm(step, axis=1)
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
            position, edge_start, edge_end, edge_valid, 0.25 * pitch
        )
        stationary_delta = values[:, 1:] - value[:, None]
        stationary_above = stationary_delta > 0.0
        stationary_crossing = jnp.sum(
            stationary_above != jnp.roll(stationary_above, -1, axis=1), axis=1
        )
        saddle_type = determinant < -1.0e-12
        extremum_type = (determinant > 1.0e-12) & (
            jnp.where(h00 + h11 < 0.0, 1, -1) == axis_kind
        )
        extremal_centroid = jnp.argmax(operator.polarity * centroid)
        axis_seed = jnp.arange(centroid.shape[0]) == extremal_centroid
        typed_extremum = axis_seed & finite & near_cell & extremum_type
        typed_saddle = finite & near_cell & saddle_type
        candidate_rows = jnp.column_stack((position, value, jnp.zeros_like(value)))
        contained = operator._fixed_design_topology.contained_x_candidates(
            candidate_rows
        )
        contained_extremum = typed_extremum & contained
        contained_saddle = typed_saddle & contained
        representative_extremum = _deduplicate(
            position, contained_extremum, 0.5 * pitch, cell_distance
        )
        representative_saddle = _deduplicate(
            position, contained_saddle, 0.5 * pitch, cell_distance
        )
        extremum_count = jnp.sum(representative_extremum, dtype=jnp.int32)
        saddle_count = jnp.sum(representative_saddle, dtype=jnp.int32)
        extremum_index = jnp.where(
            representative_extremum, size=capacity, fill_value=0
        )[0]
        saddle_index = jnp.where(representative_saddle, size=capacity, fill_value=0)[0]
        slot = jnp.arange(capacity)
        extremum_valid = slot < jnp.minimum(extremum_count, capacity)
        saddle_valid = slot < jnp.minimum(saddle_count, capacity)
        extremum_rows = jnp.where(
            extremum_valid[:, None],
            candidate_rows[extremum_index].at[:, 3].set(float(axis_kind)),
            jnp.nan,
        )
        saddle_rows = jnp.where(
            saddle_valid[:, None], candidate_rows[saddle_index], jnp.nan
        )
        wall_data = operator._fixed_design_topology.wall_anchor_data(
            wall_flux, operator.polarity
        )
        qualified_extremum = operator._fixed_design_topology.qualified_o_candidates(
            extremum_rows,
            saddle_rows,
            wall_data,
            operator.polarity,
            centroid,
            operator.inside_material,
            None,
        )
        qualified_extremum = qualified_extremum & extremum_valid
        return {
            "crossing_count": crossing,
            "stationary_crossing_count": stationary_crossing,
            "raw_extremum": raw_extremum,
            "raw_saddle": raw_saddle,
            "axis_seed": axis_seed,
            "typed_extremum": typed_extremum,
            "typed_saddle": typed_saddle,
            "contained_extremum": contained_extremum,
            "contained_saddle": contained_saddle,
            "representative_extremum": representative_extremum,
            "representative_saddle": representative_saddle,
            "qualified_extremum_rows": jnp.where(
                qualified_extremum[:, None], extremum_rows, jnp.nan
            ),
            "contained_saddle_rows": saddle_rows,
            "raw_position": position,
            "raw_value": value,
            "requested_step_m": requested_distance,
            "inside_cell": inside_cell,
            "distance_to_cell_m": cell_distance,
            "hessian_determinant_local": determinant,
            "extremum_overflow": extremum_count > capacity,
            "saddle_overflow": saddle_count > capacity,
            "extremum_count": extremum_count,
            "saddle_count": saddle_count,
        }

    return jax.jit(read)


def _finite_rows(rows: Any) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    return array[np.all(np.isfinite(array[:, :3]), axis=1)]


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


def _smooth_perturbation(coordinates: np.ndarray, span: float) -> np.ndarray:
    """Return deterministic smooth noise with peak amplitude one ten-thousandth span."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    radial = coordinates[:, 0]
    vertical = coordinates[:, 1]
    signal = np.sin(1.7 * radial + 0.3 * vertical) + 0.41 * np.cos(
        0.6 * radial - 1.3 * vertical
    )
    signal /= np.max(np.abs(signal))
    return 1.0e-4 * span * signal


def _instrument_controls(
    machine: Any, operator: Any, read: Any, pitch: float
) -> dict[str, Any]:
    """Make the census observe manufactured nulls at one cell centroid."""

    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    domain_centre = np.mean(machine.node, axis=0)
    cell = int(np.argmin(np.linalg.norm(machine.node - domain_centre, axis=1)))
    target = np.asarray(machine.node[cell], dtype=np.float64)
    local = (coordinates - target) / pitch
    saddle_state = local[:, 0] ** 2 - local[:, 1] ** 2
    axis_kind = operator._fixed_design_topology.grid.extremum_polarity
    if axis_kind is None:
        raise RuntimeError("the manufactured extremum needs a declared polarity")
    extremum_state = -float(axis_kind) * (local[:, 0] ** 2 + local[:, 1] ** 2)
    saddle = jax.block_until_ready(read(jnp.asarray(saddle_state)))
    extremum = jax.block_until_ready(read(jnp.asarray(extremum_state)))
    saddle_position = np.asarray(saddle["raw_position"])[cell]
    extremum_position = np.asarray(extremum["raw_position"])[cell]
    saddle_detected = bool(
        np.asarray(saddle["raw_saddle"])[cell]
        and np.asarray(saddle["typed_saddle"])[cell]
        and np.linalg.norm(saddle_position - target) <= 1.0e-10
    )
    extremum_detected = bool(
        np.asarray(extremum["raw_extremum"])[cell]
        and np.asarray(extremum["typed_extremum"])[cell]
        and np.linalg.norm(extremum_position - target) <= 1.0e-10
    )
    return {
        "cell_index": cell,
        "target_rz_m": target.tolist(),
        "saddle_crossing_count": int(np.asarray(saddle["crossing_count"])[cell]),
        "saddle_detected_and_polished": saddle_detected,
        "saddle_position_error_m": float(np.linalg.norm(saddle_position - target)),
        "extremum_crossing_count": int(np.asarray(extremum["crossing_count"])[cell]),
        "extremum_detected_and_polished": extremum_detected,
        "extremum_position_error_m": float(np.linalg.norm(extremum_position - target)),
    }


def _reference_cell_probe(
    machine: Any,
    operator: Any,
    state: np.ndarray,
    measured: dict[str, Any],
    reference_mask: np.ndarray,
) -> dict[str, Any] | None:
    """Describe the own-cell signs and unconstrained polish at one true null."""

    indices = np.flatnonzero(reference_mask)
    if not len(indices):
        return None
    cell = int(indices[0])
    stencil = _support_stencil(operator)
    physical = state[: operator.physical_node_number]
    centroid, _wall = operator._fixed_design_topology.split_flux_map(
        jnp.asarray(physical)
    )
    pool = np.concatenate(
        (np.asarray(centroid), state[operator.physical_node_number :])
    )
    values = pool[np.asarray(stencil.ring_gather_index)[cell]]
    delta = values[1:] - values[0]
    position = np.asarray(measured["raw_position"])[cell]
    stationary_value = float(np.asarray(measured["raw_value"])[cell])
    stationary_delta = values[1:] - stationary_value
    determinant = float(np.asarray(measured["hessian_determinant_local"])[cell])
    return {
        "cell_index": cell,
        "cell_centroid_rz_m": np.asarray(machine.node[cell]).tolist(),
        "vertex_minus_centroid_wb": delta.tolist(),
        "above_centroid_bits": "".join(str(int(value > 0.0)) for value in delta),
        "centroid_level_cyclic_sign_change_count": int(
            np.asarray(measured["crossing_count"])[cell]
        ),
        "stationary_value_wb": stationary_value,
        "vertex_minus_stationary_value_wb": stationary_delta.tolist(),
        "above_stationary_value_bits": "".join(
            str(int(value > 0.0)) for value in stationary_delta
        ),
        "stationary_level_cyclic_sign_change_count": int(
            np.asarray(measured["stationary_crossing_count"])[cell]
        ),
        "polished_position_rz_m": position.tolist(),
        "requested_step_m": float(np.asarray(measured["requested_step_m"])[cell]),
        "inside_cell_polygon": bool(np.asarray(measured["inside_cell"])[cell]),
        "distance_to_cell_polygon_m": float(
            np.asarray(measured["distance_to_cell_m"])[cell]
        ),
        "hessian_determinant_local": determinant,
        "hessian_class": (
            "saddle"
            if determinant < 0.0
            else "extremum"
            if determinant > 0.0
            else "singular"
        ),
    }


def _production_read(operator: Any, state: np.ndarray) -> dict[str, Any]:
    """Run the production neighbour-centroid read on the identical state."""

    physical = jnp.asarray(state[: operator.physical_node_number], dtype=jnp.float64)
    _masks, topology, _connected, axis_admitted = jax.block_until_ready(
        operator._fixed_design_read(physical)
    )
    axis = np.asarray(topology.axis, dtype=np.float64)
    saddle = np.asarray(topology.x_point, dtype=np.float64)
    diverted = bool(topology.diverted)
    return {
        "axis_admitted": bool(axis_admitted),
        "axis_rz_m": axis.tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "saddle_admitted": bool(diverted and np.all(np.isfinite(saddle))),
        "saddle_rz_m": saddle.tolist() if np.all(np.isfinite(saddle)) else None,
        "saddle_flux_wb": (
            float(topology.x_point_flux) if np.all(np.isfinite(saddle)) else None
        ),
        "class": "diverted" if diverted else "limited",
    }


def _shift_control(
    result: dict[str, Any], state: np.ndarray, operator: Any
) -> dict[str, Any]:
    """Show that closing the ring makes the count invariant to cyclic origin."""

    stencil = _support_stencil(operator)
    physical = state[: operator.physical_node_number]
    centroid, _wall = operator._fixed_design_topology.split_flux_map(
        jnp.asarray(physical)
    )
    pool = np.concatenate(
        (np.asarray(centroid), state[operator.physical_node_number :])
    )
    values = pool[np.asarray(stencil.ring_gather_index)]
    delta = values[:, 1:] - values[:, :1]
    original = delta > 0.0
    shifted = np.roll(original, 1, axis=1)
    cyclic_original = np.sum(original != np.roll(original, -1, axis=1), axis=1)
    cyclic_shifted = np.sum(shifted != np.roll(shifted, -1, axis=1), axis=1)
    linear_original = np.sum(original[:, 1:] != original[:, :-1], axis=1)
    linear_shifted = np.sum(shifted[:, 1:] != shifted[:, :-1], axis=1)
    observed = np.asarray(result["crossing_count"])
    return {
        "vertex_values_shifted_by_one_cyclic_position": True,
        "cyclic_counts_match_benchmark": bool(
            np.array_equal(cyclic_original, observed)
        ),
        "cyclic_count_changes_after_shift": int(
            np.count_nonzero(cyclic_original != cyclic_shifted)
        ),
        "open_chain_count_changes_after_shift": int(
            np.count_nonzero(linear_original != linear_shifted)
        ),
        "interpretation": (
            "the cyclic count is invariant to the arbitrary first vertex while an "
            "open-chain count changes when the same ordered ring is shifted"
        ),
    }


def _periodic_ring_modes(
    machine: Any,
    operator: Any,
    state: np.ndarray,
    cell: int,
    pitch: float,
) -> dict[str, Any]:
    """Express one six-sample quadratic fit in periodic ring modes."""

    stencil = _support_stencil(operator)
    physical = state[: operator.physical_node_number]
    centroid_flux, _wall = operator._fixed_design_topology.split_flux_map(
        jnp.asarray(physical)
    )
    pool = np.concatenate(
        (np.asarray(centroid_flux), state[operator.physical_node_number :])
    )
    values = pool[np.asarray(stencil.ring_gather_index)[cell]]
    centroid_value = float(values[0])
    vertex_value = np.asarray(values[1:], dtype=np.float64)
    centre = np.asarray(machine.node[cell], dtype=np.float64)
    vertices = np.asarray(machine.sampling_vertices[cell], dtype=np.float64)
    offset = vertices - centre
    radius = np.linalg.norm(offset, axis=1)
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    design = np.column_stack(
        (
            np.ones(6),
            np.cos(angle),
            np.sin(angle),
            np.cos(2.0 * angle),
            np.sin(2.0 * angle),
            np.cos(3.0 * angle),
        )
    )
    mean, first_cos, first_sin, second_cos, second_sin, third_cos = np.linalg.solve(
        design, vertex_value
    )
    ring_radius = float(np.mean(radius))
    first_amplitude = float(np.hypot(first_cos, first_sin))
    second_amplitude = float(np.hypot(second_cos, second_sin))
    isotropic = float(mean - centroid_value)
    gradient = np.asarray((first_cos, first_sin)) / ring_radius
    isotropic_hessian = 2.0 * isotropic / ring_radius**2
    hessian = np.asarray(
        (
            (
                isotropic_hessian + 2.0 * second_cos / ring_radius**2,
                2.0 * second_sin / ring_radius**2,
            ),
            (
                2.0 * second_sin / ring_radius**2,
                isotropic_hessian - 2.0 * second_cos / ring_radius**2,
            ),
        )
    )
    stationary_step = -np.linalg.solve(hessian, gradient)
    ratio_distance = (
        ring_radius * first_amplitude / (2.0 * second_amplitude)
        if second_amplitude > 0.0
        else math.inf
    )
    stationary_value = (
        centroid_value
        + float(gradient @ stationary_step)
        + 0.5 * float(stationary_step @ hessian @ stationary_step)
    )
    stationary_sign = vertex_value - stationary_value
    stationary_bits = stationary_sign > 0.0
    stationary_crossing = int(np.sum(stationary_bits != np.roll(stationary_bits, -1)))
    determinant = float(np.linalg.det(hessian))
    return {
        "cell_index": cell,
        "ring_radius_m": ring_radius,
        "ring_radius_spread_m": float(np.ptp(radius)),
        "mean_wb": float(mean),
        "m1_cos_wb": float(first_cos),
        "m1_sin_wb": float(first_sin),
        "m1_amplitude_wb": first_amplitude,
        "m2_cos_wb": float(second_cos),
        "m2_sin_wb": float(second_sin),
        "m2_amplitude_wb": second_amplitude,
        "m2_phase_rad": float(0.5 * np.arctan2(second_sin, second_cos)),
        "isotropic_mean_minus_centroid_wb": isotropic,
        "isotropic_hessian_per_m2": isotropic_hessian,
        "m3_cos_wb": float(third_cos),
        "m3_amplitude_wb": abs(float(third_cos)),
        "hessian_determinant_per_m4": determinant,
        "hessian_class": "saddle" if determinant < 0.0 else "extremum",
        "stationary_distance_from_modes_m": float(np.linalg.norm(stationary_step)),
        "stationary_distance_from_modes_in_pitch": float(
            np.linalg.norm(stationary_step) / pitch
        ),
        "m1_to_m2_ratio_distance_m": ratio_distance,
        "m1_to_m2_ratio_distance_in_pitch": ratio_distance / pitch,
        "stationary_value_wb": stationary_value,
        "stationary_level_cyclic_sign_change_count": stationary_crossing,
    }


def _stationary_source_cell(
    machine: Any,
    operator: Any,
    state: np.ndarray,
    selected_position: np.ndarray,
) -> int:
    """Resolve the own-node cell whose fitted root supplied a retained row."""

    stencil = _support_stencil(operator)
    physical = state[: operator.physical_node_number]
    centroid_flux, _wall = operator._fixed_design_topology.split_flux_map(
        jnp.asarray(physical)
    )
    pool = np.concatenate(
        (np.asarray(centroid_flux), state[operator.physical_node_number :])
    )
    values = pool[np.asarray(stencil.ring_gather_index)]
    coefficient = np.einsum("nps,ns->np", stencil.ring_flux_weight, values)
    h00 = 2.0 * coefficient[:, 3]
    h01 = coefficient[:, 4]
    h11 = 2.0 * coefficient[:, 5]
    determinant = h00 * h11 - h01 * h01
    safe = np.where(np.abs(determinant) > 1.0e-12, determinant, np.nan)
    local_radial = (h01 * coefficient[:, 2] - h11 * coefficient[:, 1]) / safe
    local_vertical = (h01 * coefficient[:, 1] - h00 * coefficient[:, 2]) / safe
    local = np.column_stack((local_radial, local_vertical))
    position = np.asarray(stencil.ring_sampling_centre) + local * np.asarray(
        stencil.ring_coordinate_scale
    )
    distance = np.linalg.norm(position - selected_position, axis=1)
    cell = int(np.nanargmin(distance))
    if distance[cell] > 1.0e-10:
        raise RuntimeError(
            "the retained stationary point does not match an own-node fit"
        )
    return cell


def _add_periodic_presentation(
    row: dict[str, Any], report_directory: Path
) -> dict[str, Any]:
    """Add mode and local sampling evidence without rerunning admission."""

    case_name = row["case"]
    requested_cells = int(row["requested_cells"])
    machine, operator, state, exact = _machine_and_field(case_name, requested_cells)
    pitch = float(row["characteristic_pitch_m"])
    axis_reference = np.asarray(exact.magnetic_axis, dtype=np.float64)
    centroid_flux = np.asarray(state[: len(machine.node)], dtype=np.float64)
    axis_cell = int(np.argmax(float(operator.polarity) * centroid_flux))
    sample_count = np.asarray(machine.moment_geometry.sample_vertex_count)

    def nearby_four(point: np.ndarray) -> int:
        distance = np.linalg.norm(np.asarray(machine.node) - point, axis=1)
        return int(np.count_nonzero((sample_count == 4) & (distance <= 2.0 * pitch)))

    axis_modes = _periodic_ring_modes(machine, operator, state, axis_cell, pitch)
    axis_kind = operator._fixed_design_topology.grid.extremum_polarity
    axis_modes["declared_extremum_kind"] = int(axis_kind)
    axis_modes["definite_extremum"] = bool(
        axis_modes["hessian_determinant_per_m4"] > 0.0
    )
    axis_modes["extremal_centroid_seed_matches_analytic_axis_cell"] = bool(
        _true_cell_mask(machine, axis_reference, pitch)[axis_cell]
    )
    axis_modes["nearby_four_sample_cells_within_two_pitches"] = nearby_four(
        axis_reference
    )
    saddle_modes = None
    if certificate._is_diverted_case(case_name):
        saddle_reference = np.asarray(exact.x_point, dtype=np.float64)
        saddle_cell = _stationary_source_cell(
            machine,
            operator,
            state,
            np.asarray(row["vertex_read"]["saddle"]["position_rz_m"]),
        )
        saddle_modes = _periodic_ring_modes(
            machine, operator, state, saddle_cell, pitch
        )
        saddle_modes["analytic_reference_cell_index"] = int(
            np.flatnonzero(_true_cell_mask(machine, saddle_reference, pitch))[0]
        )
        saddle_modes["nearby_four_sample_cells_within_two_pitches"] = nearby_four(
            saddle_reference
        )
    row["periodic_ring_presentation"] = {
        "basis": (
            "mean; m=1 cosine and sine gradient; m=2 cosine and sine traceless "
            "Hessian; m=3 cosine; isotropic Hessian from ring mean minus centroid"
        ),
        "axis_cell": axis_modes,
        "saddle_cell": saddle_modes,
        "axis_seed_and_mode_criterion_agree": bool(
            row["vertex_read"]["axis"]["admitted"]
            and axis_modes["definite_extremum"]
            and axis_modes["extremal_centroid_seed_matches_analytic_axis_cell"]
        ),
    }
    _write_json(_part_path(report_directory, case_name, requested_cells), row)
    return row


def _measure_row(
    case_name: str, requested_cells: int, report_directory: Path
) -> dict[str, Any]:
    """Measure and persist one analytic-flux carrier row."""

    part_path = _part_path(report_directory, case_name, requested_cells)
    progress = {
        "schema": "nova.vertex-ring-census-part",
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
    read = _vertex_read_function(operator, pitch)
    measured = jax.block_until_ready(read(jnp.asarray(state, dtype=jnp.float64)))
    extrema = _finite_rows(measured["qualified_extremum_rows"])
    saddles = _finite_rows(measured["contained_saddle_rows"])
    axis = _selected_error(extrema, axis_reference, axis_flux, pitch, span)
    saddle = (
        _selected_error(saddles, x_reference, x_flux, pitch, span) if diverted else None
    )
    raw_axis_true = _true_cell_mask(machine, axis_reference, pitch)
    raw_x_true = (
        _true_cell_mask(machine, x_reference, pitch)
        if x_reference is not None
        else np.zeros(len(machine.node), dtype=bool)
    )
    raw_extremum = np.asarray(measured["raw_extremum"], dtype=bool)
    raw_saddle = np.asarray(measured["raw_saddle"], dtype=bool)
    representative_extremum = np.asarray(
        measured["representative_extremum"], dtype=bool
    )
    representative_saddle = np.asarray(measured["representative_saddle"], dtype=bool)
    instrument = _instrument_controls(machine, operator, read, pitch)
    if not (
        instrument["saddle_detected_and_polished"]
        and instrument["extremum_detected_and_polished"]
    ):
        raise RuntimeError("manufactured stationary-point controls failed")
    axis_probe = _reference_cell_probe(
        machine, operator, state, measured, raw_axis_true
    )
    saddle_probe = (
        _reference_cell_probe(machine, operator, state, measured, raw_x_true)
        if diverted
        else None
    )
    production = _production_read(operator, state)
    production["axis_position_error_m"] = float(
        np.linalg.norm(np.asarray(production["axis_rz_m"]) - axis_reference)
    )
    production["axis_position_error_in_pitch"] = (
        production["axis_position_error_m"] / pitch
    )
    if diverted and production["saddle_admitted"]:
        production["saddle_position_error_m"] = float(
            np.linalg.norm(np.asarray(production["saddle_rz_m"]) - x_reference)
        )
        production["saddle_position_error_in_pitch"] = (
            production["saddle_position_error_m"] / pitch
        )
        production["saddle_level_error_wb"] = abs(production["saddle_flux_wb"] - x_flux)
        production["saddle_level_error_in_span"] = (
            production["saddle_level_error_wb"] / span
        )
    elif diverted:
        production.update(
            {
                "saddle_position_error_m": None,
                "saddle_position_error_in_pitch": None,
                "saddle_level_error_wb": None,
                "saddle_level_error_in_span": None,
            }
        )
    noise_control = None
    if diverted:
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        perturbed_state = state + _smooth_perturbation(coordinates, span)
        perturbed = jax.block_until_ready(
            read(jnp.asarray(perturbed_state, dtype=jnp.float64))
        )
        perturbed_axis = _selected_error(
            _finite_rows(perturbed["qualified_extremum_rows"]),
            axis_reference,
            axis_flux,
            pitch,
            span,
        )
        perturbed_saddle = _selected_error(
            _finite_rows(perturbed["contained_saddle_rows"]),
            x_reference,
            x_flux,
            pitch,
            span,
        )
        noise_control = {
            "amplitude_in_span": 1.0e-4,
            "axis_admitted_against_unperturbed_reference": perturbed_axis["admitted"],
            "saddle_admitted_against_unperturbed_reference": perturbed_saddle[
                "admitted"
            ],
            "axis_position_error_in_pitch": perturbed_axis["position_error_in_pitch"],
            "saddle_position_error_in_pitch": perturbed_saddle[
                "position_error_in_pitch"
            ],
        }
    row = progress | {
        "allocation": _allocation("cpu"),
        "cache": machine.cache,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "analytic": {
            "axis_rz_m": axis_reference.tolist(),
            "axis_flux_wb": axis_flux,
            "x_point_rz_m": x_reference.tolist() if x_reference is not None else None,
            "x_point_flux_wb": x_flux,
            "reference_span_wb": span,
        },
        "vertex_read": {
            "criteria": {
                "centroid_value_sign_change": {
                    "axis_reference_cell_admitted": bool(
                        np.any(raw_extremum & raw_axis_true)
                    ),
                    "saddle_reference_cell_admitted": (
                        bool(np.any(raw_saddle & raw_x_true)) if diverted else None
                    ),
                },
                "quadratic_stationary_point": {
                    "axis_seed": "extremal centroid followed by own-node polish",
                    "saddle_seed": (
                        "indefinite own-node quadratic with stationary point inside "
                        "its cell polygon or within one quarter pitch"
                    ),
                    "axis_admitted": axis["admitted"],
                    "saddle_admitted": saddle["admitted"] if saddle else None,
                    "saddle_stationary_level_has_four_changes": (
                        saddle_probe["stationary_level_cyclic_sign_change_count"] == 4
                        if saddle_probe is not None
                        else None
                    ),
                },
            },
            "axis": axis,
            "saddle": saddle,
            "counts": {
                "raw_extremum": int(np.count_nonzero(raw_extremum)),
                "raw_saddle": int(np.count_nonzero(raw_saddle)),
                "typed_extremum": int(
                    np.count_nonzero(np.asarray(measured["typed_extremum"]))
                ),
                "typed_saddle": int(
                    np.count_nonzero(np.asarray(measured["typed_saddle"]))
                ),
                "contained_extremum": int(
                    np.count_nonzero(np.asarray(measured["contained_extremum"]))
                ),
                "contained_saddle": int(
                    np.count_nonzero(np.asarray(measured["contained_saddle"]))
                ),
                "representative_extremum": int(
                    np.count_nonzero(representative_extremum)
                ),
                "representative_saddle": int(np.count_nonzero(representative_saddle)),
                "qualified_extremum": len(extrema),
            },
            "false_candidates": {
                "before_hessian_and_containment": {
                    "extremum": int(np.count_nonzero(raw_extremum & ~raw_axis_true)),
                    "saddle": int(np.count_nonzero(raw_saddle & ~raw_x_true)),
                },
                "after_hessian_containment_and_dedupe": {
                    "extremum": max(len(extrema) - int(axis["admitted"]), 0),
                    "saddle": max(
                        len(saddles) - int(bool(saddle and saddle["admitted"])), 0
                    ),
                },
            },
            "extremum_rows": extrema.tolist(),
            "saddle_rows": saddles.tolist(),
            "capacity_overflow": {
                "extremum": bool(measured["extremum_overflow"]),
                "saddle": bool(measured["saddle_overflow"]),
            },
            "private_exclusion": (
                "production qualified_o_candidates with its saddle-directed "
                "axis-component flood; candidate saddles use production wall "
                "containment"
            ),
            "analytic_axis_cell_probe": axis_probe,
            "analytic_saddle_cell_probe": saddle_probe,
        },
        "production_read": production,
        "instrument_controls": instrument,
        "smooth_noise_control": noise_control,
        "shift_control": (
            _shift_control(measured, state, operator)
            if case_name == certificate.DIVERTED_CASE_NAME and requested_cells == 300
            else None
        ),
        "device_timing": None,
        "wall_seconds": perf_counter() - started,
        "completed": True,
    }
    _write_json(part_path, row)
    print(
        "VERTEX_CENSUS_ROW "
        f"case={case_name} requested={requested_cells} realised={len(machine.node)} "
        f"axis={axis['admitted']} saddle={saddle and saddle['admitted']} "
        f"production_saddle={production['saddle_admitted']} "
        f"seconds={row['wall_seconds']:.3f}",
        flush=True,
    )
    return row


def _run_worker(report_directory: Path, shard_index: int, shard_count: int) -> None:
    """Measure one deterministic shard inside the shared CPU allocation."""

    _allocation("cpu")
    for case_name, requested_cells in CPU_ROWS[shard_index::shard_count]:
        _measure_row(case_name, requested_cells, report_directory)


def _load_production_ladder() -> dict[int, dict[str, Any]]:
    """Load the bounded banked production ladder without opening it as prose."""

    if PRODUCTION_RECEIPT.stat().st_size > 1_000_000:
        raise RuntimeError("production ladder receipt exceeds one megabyte")
    payload = json.loads(PRODUCTION_RECEIPT.read_text(encoding="utf-8"))
    return {int(row["requested_cells"]): row for row in payload["rows"]}


def _render_error_ladder(
    rows: list[dict[str, Any]], figure_directory: Path
) -> dict[str, str]:
    """Compare vertex and production saddle error with misses made explicit."""

    figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for key, colour, label in (
        ("vertex_read", COLOURS["vertex"], "own-vertex census"),
        ("production_read", COLOURS["production"], "production neighbour census"),
    ):
        admitted_x = []
        admitted_y = []
        missed_x = []
        for row in rows:
            if key == "vertex_read":
                read = row[key]["saddle"]
                admitted = bool(read["admitted"])
                error = read["position_error_in_pitch"]
            else:
                read = row[key]
                admitted = bool(read["saddle_admitted"])
                error = read.get("saddle_position_error_in_pitch")
            if admitted:
                admitted_x.append(row["realised_cells"])
                admitted_y.append(error)
            else:
                missed_x.append(row["realised_cells"])
        axis.plot(admitted_x, admitted_y, marker="o", color=colour, label=label)
        axis.scatter(
            missed_x,
            [-0.004 if key == "vertex_read" else -0.008] * len(missed_x),
            marker="x",
            color=colour,
            s=46,
            label=f"{label} unadmitted",
        )
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set_xscale("log")
    axis.set_xlabel("realised plasma cells")
    axis.set_ylabel("saddle position error / pitch")
    axis.set_ylim(bottom=-0.012)
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / "saddle-error-comparison.svg"
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/vertex-census/"
            "saddle-error-comparison.svg"
        ),
    }


def _render_saddle_panel(row: dict[str, Any], figure_directory: Path) -> dict[str, str]:
    """Draw the analytic saddle neighbourhood and its own-cell vertex signs."""

    requested = int(row["requested_cells"])
    machine, _operator, _state, exact = _machine_and_field(
        certificate.DIVERTED_CASE_NAME, requested
    )
    pitch = float(row["characteristic_pitch_m"])
    contains = _true_cell_mask(machine, ANALYTIC_X, pitch)
    cell = int(np.flatnonzero(contains)[0])
    vertices = np.asarray(machine.sampling_vertices[cell], dtype=np.float64)
    points = np.vstack((machine.node[cell], vertices))
    values = limiter_audit._exact_flux(certificate.DIVERTED_CASE_NAME, exact, points)
    centroid_signs = values[1:] - values[0]
    probe = row["vertex_read"]["analytic_saddle_cell_probe"]
    stationary_signs = values[1:] - probe["stationary_value_wb"]
    extent = max(3.0 * pitch, 0.18)
    radial = np.linspace(ANALYTIC_X[0] - extent, ANALYTIC_X[0] + extent, 241)
    height = np.linspace(ANALYTIC_X[1] - extent, ANALYTIC_X[1] + extent, 241)
    radius_grid, height_grid = np.meshgrid(radial, height)
    field = limiter_audit._exact_flux(
        certificate.DIVERTED_CASE_NAME,
        exact,
        np.column_stack((radius_grid.ravel(), height_grid.ravel())),
    ).reshape(radius_grid.shape)
    levels = poloidal.contour_levels(field, count=14)
    figure, axis = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, height, field, levels, color=COLOURS["analytic"]
    )
    poloidal.draw_wall(axis, units=(np.asarray(machine.wall_node),), linewidth=0.65)
    poloidal.draw_nulls(
        axis,
        magnetic_axis=ANALYTIC_AXIS,
        x_points=ANALYTIC_X[None, :],
        style=DEFAULT_INK.variant(
            axis_marker="^",
            axis_color=COLOURS["analytic"],
            xpoint_color=COLOURS["analytic"],
        ),
        contain=(np.asarray(machine.wall_node),),
    )
    nearby = [
        np.asarray(polygon)
        for polygon in machine.cell_polygons
        if np.linalg.norm(np.mean(polygon, axis=0) - ANALYTIC_X) <= 2.5 * pitch
    ]
    axis.add_collection(
        PolyCollection(
            nearby,
            facecolors="none",
            edgecolors=COLOURS["cells"],
            linewidths=0.5,
        )
    )
    closed = np.vstack((vertices, vertices[0]))
    axis.plot(closed[:, 0], closed[:, 1], color=COLOURS["vertex"], linewidth=1.5)
    for index, (vertex, centroid_sign, stationary_sign) in enumerate(
        zip(vertices, centroid_signs, stationary_signs, strict=True)
    ):
        axis.plot(
            vertex[0],
            vertex[1],
            marker="o",
            markerfacecolor=(
                COLOURS["positive"] if stationary_sign > 0 else COLOURS["negative"]
            ),
            markeredgecolor="white",
            markersize=7,
            linestyle="none",
            zorder=8,
        )
        axis.text(
            vertex[0],
            vertex[1],
            f" {index}:c{'+' if centroid_sign > 0 else '-'}"
            f"/q{'+' if stationary_sign > 0 else '-'}",
            fontsize=7,
        )
    saddle = row["vertex_read"]["saddle"]
    if probe is not None:
        position = probe["polished_position_rz_m"]
        axis.plot(
            position[0],
            position[1],
            marker="X",
            markerfacecolor=COLOURS["vertex"],
            markeredgecolor="white",
            markersize=8,
            linestyle="none",
            zorder=9,
        )
    axis.set_xlim(ANALYTIC_X[0] - extent, ANALYTIC_X[0] + extent)
    axis.set_ylim(ANALYTIC_X[1] - extent, ANALYTIC_X[1] + extent)
    poloidal_axes(axis)
    admission = "admitted" if saddle["admitted"] else "unadmitted"
    centroid_changes = probe["centroid_level_cyclic_sign_change_count"]
    stationary_changes = probe["stationary_level_cyclic_sign_change_count"]
    axis.set_title(
        f"{row['realised_cells']} cells: centroid level {centroid_changes}, "
        f"stationary level {stationary_changes}; {admission}",
        fontsize=9,
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / f"saddle-region-cells-{requested}.svg"
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/vertex-census/"
            + destination.name
        ),
    }


def _write_report(receipt: dict[str, Any], destination: Path) -> None:
    """Write the human-readable admission, error, candidate, and timing table."""

    rows = receipt["single_null_rows"]
    lines = [
        "# Own-vertex-ring stationary-point census",
        "",
        (
            "The production defect reproduced before this independent evaluation: "
            f"the neighbour-centroid read admitted no saddle at "
            f"{rows[0]['realised_cells']} and {rows[2]['realised_cells']} realised "
            "cells, while its 500-requested positive control admitted one."
        ),
        "",
        (
            "The raw vertex-minus-centroid rule is retained as a negative control. "
            "The centroid-value contour generally cuts only one hyperbola branch "
            "pair when the centroid is displaced from the saddle, so it produces "
            "two cyclic changes. The operative criterion instead solves the "
            "own-node quadratic gradient, classifies its Hessian, and requires the "
            "stationary point to lie in or within one quarter pitch of its cell."
        ),
        "",
        "## Analytic single-null ladder",
        "",
        (
            "| requested | realised | centroid_sign_census_admitted | "
            "vertex_census_admitted | vertex_census_position_error_m | "
            "vertex_census_position_error_in_pitch | vertex_census_level_error_wb | "
            "vertex_census_level_error_in_span | production_admitted | "
            "production_position_error_m | production_position_error_in_pitch | "
            "production_level_error_wb | production_level_error_in_span | "
            "centroid_sign_false_saddle_before→vertex_census_false_saddle_after | "
            "centroid_sign_false_extremum_before→vertex_census_false_extremum_after | "
            "noise X |"
        ),
        "|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        vertex = row["vertex_read"]
        saddle = vertex["saddle"]
        production = row["production_read"]
        false = vertex["false_candidates"]
        saddle_pitch_error = saddle["position_error_in_pitch"]
        saddle_level_error = saddle["level_error_in_span"]
        noise_admitted = row["smooth_noise_control"][
            "saddle_admitted_against_unperturbed_reference"
        ]
        raw_admitted = vertex["criteria"]["centroid_value_sign_change"][
            "saddle_reference_cell_admitted"
        ]
        lines.append(
            f"| {row['requested_cells']} | {row['realised_cells']} | "
            f"{'yes' if raw_admitted else 'no'} | "
            f"{'yes' if saddle['admitted'] else 'no'} | "
            f"{saddle['position_error_m']} | "
            f"{saddle_pitch_error if saddle_pitch_error is not None else '—'} | "
            f"{saddle['level_error_wb']} | "
            f"{saddle_level_error if saddle_level_error is not None else '—'} | "
            f"{'yes' if production['saddle_admitted'] else 'no'} | "
            f"{production.get('saddle_position_error_m') or '—'} | "
            f"{production.get('saddle_position_error_in_pitch') or '—'} | "
            f"{production.get('saddle_level_error_wb') or '—'} | "
            f"{production.get('saddle_level_error_in_span') or '—'} | "
            f"{false['before_hessian_and_containment']['saddle']}→"
            f"{false['after_hessian_containment_and_dedupe']['saddle']} | "
            f"{false['before_hessian_and_containment']['extremum']}→"
            f"{false['after_hessian_containment_and_dedupe']['extremum']} | "
            f"{'yes' if noise_admitted else 'no'} |"
        )
    lines.extend(["", "## O-point positive control", ""])
    lines.append(
        "| case | requested | realised | vertex_census_axis_admitted | "
        "vertex_census_axis_position_error_m | "
        "vertex_census_axis_position_error_in_pitch | production_axis_admitted | "
        "production_axis_position_error_m | production_axis_position_error_in_pitch | "
        "extremal_centroid_and_mode_agree |"
    )
    lines.append("|:---|---:|---:|:---:|---:|---:|:---:|---:|---:|:---:|")
    for row in receipt["single_null_rows"] + receipt["static_rows"]:
        vertex_axis = row["vertex_read"]["axis"]
        production_axis = row["production_read"]
        presentation = row.get("periodic_ring_presentation")
        agreement = (
            presentation["axis_seed_and_mode_criterion_agree"]
            if presentation is not None
            else None
        )
        lines.append(
            f"| {row['case']} | {row['requested_cells']} | {row['realised_cells']} | "
            f"{'yes' if vertex_axis['admitted'] else 'no'} | "
            f"{vertex_axis['position_error_m']} | "
            f"{vertex_axis['position_error_in_pitch']} | "
            f"{'yes' if production_axis['axis_admitted'] else 'no'} | "
            f"{production_axis['axis_position_error_m']} | "
            f"{production_axis['axis_position_error_in_pitch']} | "
            f"{'yes' if agreement else 'no' if agreement is not None else '—'} |"
        )
    lines.extend(["", "## Static axis controls", ""])
    lines.append(
        "| case | requested | realised | centroid_sign_census_axis_admitted | "
        "vertex_census_axis_admitted | vertex_census_position_error_m | "
        "vertex_census_position_error_in_pitch | vertex_census_level_error_in_span | "
        "centroid_sign_false_saddle_before→vertex_census_false_saddle_after | "
        "centroid_sign_false_extremum_before→vertex_census_false_extremum_after |"
    )
    lines.append("|:---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|")
    for row in receipt["static_rows"]:
        vertex = row["vertex_read"]
        axis = vertex["axis"]
        false = vertex["false_candidates"]
        raw_admitted = vertex["criteria"]["centroid_value_sign_change"][
            "axis_reference_cell_admitted"
        ]
        lines.append(
            f"| {row['case']} | {row['requested_cells']} | {row['realised_cells']} | "
            f"{'yes' if raw_admitted else 'no'} | "
            f"{'yes' if axis['admitted'] else 'no'} | {axis['position_error_m']} | "
            f"{axis['position_error_in_pitch']} | {axis['level_error_in_span']} | "
            f"{false['before_hessian_and_containment']['saddle']}→"
            f"{false['after_hessian_containment_and_dedupe']['saddle']} | "
            f"{false['before_hessian_and_containment']['extremum']}→"
            f"{false['after_hessian_containment_and_dedupe']['extremum']} |"
        )
    presentation_rows = [
        row
        for row in receipt["single_null_rows"] + receipt["static_rows"]
        if row.get("periodic_ring_presentation") is not None
    ]
    if presentation_rows:
        lines.extend(["", "## Periodic six-vertex mode decomposition", ""])
        lines.append(
            "The six samples are represented by one periodic trigonometric "
            "polynomial: the mean; the cosine and sine m=1 gradient pair; the "
            "cosine and sine m=2 traceless-Hessian pair; the single Nyquist m=3 "
            "cosine; and the isotropic Hessian from ring mean minus centroid."
        )
        lines.extend(
            [
                "",
                "| case | requested | null | m1 amplitude (Wb) | m2 amplitude "
                "(Wb) | m2 phase (rad) | isotropic (Wb) | m3 amplitude (Wb) | "
                "m1:m2 distance / pitch | closed-form distance / pitch | "
                "stationary-level changes | source cell | analytic cell | "
                "four-sample cells within 2 pitch |",
                "|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in presentation_rows:
            presentation = row["periodic_ring_presentation"]
            for null_name in ("axis", "saddle"):
                modes = presentation.get(f"{null_name}_cell")
                if modes is None:
                    continue
                lines.append(
                    f"| {row['case']} | {row['requested_cells']} | {null_name} | "
                    f"{modes['m1_amplitude_wb']:.8g} | "
                    f"{modes['m2_amplitude_wb']:.8g} | "
                    f"{modes['m2_phase_rad']:.8g} | "
                    f"{modes['isotropic_mean_minus_centroid_wb']:.8g} | "
                    f"{modes['m3_amplitude_wb']:.8g} | "
                    f"{modes['m1_to_m2_ratio_distance_in_pitch']:.8g} | "
                    f"{modes['stationary_distance_from_modes_in_pitch']:.8g} | "
                    f"{modes['stationary_level_cyclic_sign_change_count']} | "
                    f"{modes['cell_index']} | "
                    f"{modes.get('analytic_reference_cell_index', '—')} | "
                    f"{modes['nearby_four_sample_cells_within_two_pitches']} |"
                )
        agreement = sum(
            row["periodic_ring_presentation"]["axis_seed_and_mode_criterion_agree"]
            for row in presentation_rows
        )
        lines.extend(
            [
                "",
                "The extremal-centroid seed and definite-Hessian mode criterion "
                f"agree on **{agreement} of {len(presentation_rows)} rows**.",
            ]
        )
    lines.extend(["", "## H200 batch cost", ""])
    timed = [row for row in rows if row.get("device_timing")]
    if not timed:
        lines.append("The accelerator timing job has not completed.")
    else:
        lines.append(
            "| requested | realised | vertex_census_batch16_ms_per_state | "
            "production_batch16_ms_per_state |"
        )
        lines.append("|---:|---:|---:|---:|")
        for row in timed:
            timing = row["device_timing"]
            lines.append(
                f"| {row['requested_cells']} | {row['realised_cells']} | "
                f"{1e3 * timing['batch_seconds_per_state_median']:.6g} | "
                f"{timing['production_ms_per_state']} |"
            )
    lines.extend(["", "## Controls and figures", ""])
    shift = rows[2]["shift_control"]
    lines.append(
        "- Cyclic-origin control at the 300-requested rung: cyclic count changes "
        "after shifting all six values by one position = "
        f"{shift['cyclic_count_changes_after_shift']}; "
        f"open-chain changes = {shift['open_chain_count_changes_after_shift']}."
    )
    noise_admissions = sum(
        row["smooth_noise_control"]["saddle_admitted_against_unperturbed_reference"]
        for row in rows
    )
    lines.append(
        "- Smooth perturbation amplitude is 1e-4 of the analytic axis-to-X span; "
        f"saddle admission survived on {noise_admissions} of {len(rows)} rungs."
    )
    analytic_cell_stationary_four = sum(
        row["vertex_read"]["criteria"]["quadratic_stationary_point"][
            "saddle_stationary_level_has_four_changes"
        ]
        for row in rows
    )
    lines.append(
        "- Vertex signs about the fitted stationary value produced four cyclic "
        "changes in the geometric analytic-X cell on "
        f"{analytic_cell_stationary_four} of {len(rows)} rungs."
    )
    if presentation_rows:
        source_cell_stationary_four = sum(
            row["periodic_ring_presentation"]["saddle_cell"][
                "stationary_level_cyclic_sign_change_count"
            ]
            == 4
            for row in rows
        )
        lines.append(
            "- The retained candidate's source cell produced four stationary-level "
            f"changes on {source_cell_stationary_four} of {len(rows)} rungs. The "
            "quarter-pitch exterior tolerance can admit an indefinite fitted root "
            "whose hyperbola has only one branch pair crossing the sampled ring, so "
            "this sign count is corroborating mode evidence rather than an equivalent "
            "admission rule for tolerance-band roots."
        )
    for figure in receipt["figures"]:
        lines.append(
            f"- [{Path(figure['path']).stem}]({figure['project_absolute_src']})"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(
    report_directory: Path, figure_directory: Path, *, render_figures: bool
) -> dict[str, Any]:
    """Aggregate parts, require positive controls, and publish bounded evidence."""

    rows = [
        _load_part(_part_path(report_directory, *identity)) for identity in CPU_ROWS
    ]
    single = [row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME]
    static = [row for row in rows if row["case"] != certificate.DIVERTED_CASE_NAME]
    by_requested = {row["requested_cells"]: row for row in single}
    if by_requested[110]["production_read"]["saddle_admitted"]:
        raise RuntimeError("the 110-requested production miss did not reproduce")
    if by_requested[300]["production_read"]["saddle_admitted"]:
        raise RuntimeError("the 300-requested production miss did not reproduce")
    if not by_requested[500]["production_read"]["saddle_admitted"]:
        raise RuntimeError("the 500-requested production positive control failed")
    if not all(
        row["instrument_controls"]["saddle_detected_and_polished"]
        and row["instrument_controls"]["extremum_detected_and_polished"]
        for row in rows
    ):
        raise RuntimeError("a manufactured stationary-point control failed")
    banked = _load_production_ladder()
    for row in single:
        prior = banked[row["requested_cells"]]["production_read"]
        if prior["saddle_admitted"] != row["production_read"]["saddle_admitted"]:
            raise RuntimeError(
                "the local production read disagrees with the banked ladder"
            )
    presentation_complete = all(
        row.get("periodic_ring_presentation") is not None for row in rows
    )
    if presentation_complete and not all(
        row["periodic_ring_presentation"]["axis_seed_and_mode_criterion_agree"]
        for row in rows
    ):
        raise RuntimeError("the extremal-centroid and axis-mode control disagrees")
    if render_figures:
        figures = [_render_error_ladder(single, figure_directory)]
        figures.extend(
            _render_saddle_panel(by_requested[cells], figure_directory)
            for cells in (110, 300)
        )
    else:
        prior_receipt = report_directory / "receipt.json"
        figures = (
            json.loads(prior_receipt.read_text(encoding="utf-8")).get("figures", [])
            if prior_receipt.exists() and prior_receipt.stat().st_size <= 1_000_000
            else []
        )
    receipt = {
        "schema": "nova.vertex-ring-census",
        "version": 1,
        "source_revision": _source_revision(),
        "analytic_flux_supplied_directly": True,
        "production_ladder_source": str(PRODUCTION_RECEIPT),
        "headline": {
            "vertex_census_saddle_admitted_rungs": sum(
                bool(row["vertex_read"]["saddle"]["admitted"]) for row in single
            ),
            "centroid_sign_census_saddle_admitted_rungs": sum(
                bool(
                    row["vertex_read"]["criteria"]["centroid_value_sign_change"][
                        "saddle_reference_cell_admitted"
                    ]
                )
                for row in single
            ),
            "production_saddle_admitted_rungs": sum(
                bool(row["production_read"]["saddle_admitted"]) for row in single
            ),
            "single_null_rung_count": len(single),
            "vertex_census_axis_admitted_rows": sum(
                bool(row["vertex_read"]["axis"]["admitted"]) for row in rows
            ),
            "total_rows": len(rows),
            "smooth_noise_saddle_admitted_rungs": sum(
                bool(
                    row["smooth_noise_control"][
                        "saddle_admitted_against_unperturbed_reference"
                    ]
                )
                for row in single
            ),
            "timing_complete": all(
                by_requested[cells].get("device_timing") is not None
                for cells in TIMING_CELL_COUNTS
            ),
        },
        "positive_controls": {
            "production_miss_reproduced_at_requested_110": True,
            "production_miss_reproduced_at_requested_300": True,
            "production_admission_reproduced_at_requested_500": True,
            "banked_production_admission_matches_every_rung": True,
            "analytic_grid_flux_nonuniform_every_row": True,
            "manufactured_saddle_detected_every_row": True,
            "manufactured_extremum_detected_every_row": True,
            "extremal_centroid_and_axis_mode_agree_every_row": (
                True if presentation_complete else None
            ),
        },
        "figures": figures,
        "single_null_rows": single,
        "static_rows": static,
    }
    _write_json(report_directory / "receipt.json", receipt)
    _write_report(receipt, report_directory / "report.md")
    _write_report(receipt, figure_directory / "report.md")
    print(
        "VERTEX_CENSUS_AGGREGATE "
        f"saddle_admitted="
        f"{receipt['headline']['vertex_census_saddle_admitted_rungs']}/9 "
        f"axis_admitted="
        f"{receipt['headline']['vertex_census_axis_admitted_rows']}/{len(rows)} "
        f"timing_complete={receipt['headline']['timing_complete']}",
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


def run_presentation(report_directory: Path, figure_directory: Path) -> None:
    """Enrich completed rows with periodic modes without rerunning admission."""

    _allocation("cpu")
    for identity in CPU_ROWS:
        row = _load_part(_part_path(report_directory, *identity))
        _add_periodic_presentation(row, report_directory)
        print(
            f"VERTEX_CENSUS_PRESENTATION case={identity[0]} requested={identity[1]}",
            flush=True,
        )
    aggregate(report_directory, figure_directory, render_figures=False)


def _timed_samples(function: Any, operand: jax.Array) -> list[float]:
    jax.block_until_ready(function(operand))
    samples = []
    for _ in range(TIMING_REPEATS):
        started = perf_counter()
        jax.block_until_ready(function(operand))
        samples.append(perf_counter() - started)
    return samples


def _device_timing(requested_cells: int, expected: dict[str, Any]) -> dict[str, Any]:
    """Time one jitted vmapped census and polish over sixteen states."""

    machine, operator, state, _exact = _machine_and_field(
        certificate.DIVERTED_CASE_NAME, requested_cells
    )
    pitch = float(expected["characteristic_pitch_m"])
    read_one = _vertex_read_function(operator, pitch)

    def timing_projection(operand):
        result = read_one(operand)
        return (
            result["qualified_extremum_rows"],
            result["contained_saddle_rows"],
            result["extremum_count"],
            result["saddle_count"],
        )

    batch = jax.jit(jax.vmap(timing_projection))
    batched = jnp.broadcast_to(
        jnp.asarray(state, dtype=jnp.float64), (TIMING_BATCH_SIZE, len(state))
    )
    observed = jax.block_until_ready(timing_projection(batched[0]))
    observed_saddles = _finite_rows(observed[1])
    expected_saddles = np.asarray(expected["vertex_read"]["saddle_rows"])
    if not np.allclose(observed_saddles, expected_saddles, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("device census does not reproduce the CPU saddle rows")
    samples = _timed_samples(batch, batched)
    return {
        "allocation": _allocation("gpu"),
        "state_count": TIMING_BATCH_SIZE,
        "repeat_count": TIMING_REPEATS,
        "batch_seconds_samples": samples,
        "batch_seconds_median": float(np.median(samples)),
        "batch_seconds_minimum": float(np.min(samples)),
        "batch_seconds_per_state_median": float(np.median(samples)) / TIMING_BATCH_SIZE,
        "batch_seconds_per_state_minimum": float(np.min(samples)) / TIMING_BATCH_SIZE,
        "production_ms_per_state": PRODUCTION_TIMING_MS[requested_cells],
        "device_matches_cpu_candidate_rows_atol_m": 1.0e-12,
        "realised_cells": len(machine.node),
    }


def run_gpu(report_directory: Path, figure_directory: Path) -> None:
    """Time the three declared carrier sizes and refresh the aggregate receipt."""

    _allocation("gpu")
    for requested_cells in TIMING_CELL_COUNTS:
        path = _part_path(
            report_directory, certificate.DIVERTED_CASE_NAME, requested_cells
        )
        row = _load_part(path)
        row["device_timing"] = _device_timing(requested_cells, row)
        _write_json(path, row)
        timing = row["device_timing"]
        print(
            "VERTEX_CENSUS_TIMING "
            f"requested={requested_cells} realised={row['realised_cells']} "
            f"batch_per_state_ms={1e3 * timing['batch_seconds_per_state_median']:.6g}",
            flush=True,
        )
    aggregate(report_directory, figure_directory, render_figures=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-run")
    cpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    cpu.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    cpu.add_argument("--workers", type=int, default=3)
    presentation = commands.add_parser("presentation-run")
    presentation.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    presentation.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    worker = commands.add_parser("cpu-worker")
    worker.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    gpu = commands.add_parser("gpu-run")
    gpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    gpu.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
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
    elif arguments.command == "presentation-run":
        run_presentation(arguments.report_directory, arguments.figure_directory)
    elif arguments.command == "cpu-worker":
        _run_worker(
            arguments.report_directory, arguments.shard_index, arguments.shard_count
        )
    elif arguments.command == "gpu-run":
        run_gpu(arguments.report_directory, arguments.figure_directory)
    else:
        aggregate(
            arguments.report_directory,
            arguments.figure_directory,
            render_figures=arguments.render_figures,
        )


if __name__ == "__main__":
    main()
