#!/usr/bin/env python3
"""Time a loop-free vertex-ring stationary-point census kernel.

Part one expresses the vertex-ring census as one jitted function with no
Python loop over cells or candidates: a (cells, 7) own-node gather, one
einsum to six quadratic coefficients, the closed-form stationary point and
Hessian, the raw and typed class flags as arrays, a fixed-capacity top-k on
the cell-distance score for dedupe, one vectorised point-in-polygon against
the wall, and a static private-region mask applied elementwise.  The kernel
is vmapped over sixteen states and timed on device at the three realised
rungs, reporting per-state cost, cold-compile time and the HLO instruction
count.

Part two profiles the sequential production census stage by stage on the same
rungs - raw census, polish, containment, dedupe, private exclusion - so the
report can name what carried the reference's two-to-thirty-two millisecond
read.

Receipt field glossary: ``raw`` is the cyclic sign census, ``typed`` adds the
finite capped Newton root and expected Hessian type, ``contained`` applies
the first-wall polygon, ``representative`` is the kernel's fixed-capacity
top-k and the production's half-pitch cluster scan, and ``qualified`` applies
the production axis-component flood on the reference and the static
private-region mask on the kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any
import uuid

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from nova.equilibrium.connectivity_boundary import _points_inside_polygon
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/census-kernel"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/census-kernel"
)
TIMING_CELL_COUNTS = (500, 1000, 2500)
WALL_NODE_COUNT = 121
TIMING_BATCH_SIZE = 16
TIMING_REPEATS = 7
WHOLE_SOLVE_TARGET_MS = 1.0
# Previously banked per-state production timing at the same realised rungs.
PRODUCTION_TIMING_MS = {500: 1.4, 1000: 4.9, 2500: 30.0}
STAGE_NAMES = ("census", "polish", "containment", "dedupe", "private_exclusion")
ANALYTIC = certificate.DIVERTED_REFERENCE
ANALYTIC_AXIS = np.asarray(ANALYTIC.magnetic_axis, dtype=np.float64)
ANALYTIC_X = np.asarray(ANALYTIC.x_point, dtype=np.float64)
COLOURS = {
    "kernel": "#3366cc",
    "reference": "#d1495b",
    "production": "#8aa04a",
    "target": "#777777",
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


def _private_region_mask(
    operator: Any,
    centroid_flux: np.ndarray,
    x_flux: float,
) -> np.ndarray:
    """Return per-cell mask of the wall-closed private flux pocket.

    The analytic separatrix value marks the wall-closed region beyond the
    X point, where the closed poloidal field lines never touch the confined
    core.  The mask is built once on the host per carrier and applied
    elementwise, replacing the sequential axis-component flood.
    """

    wall = np.asarray(operator._fixed_design_topology.wall.coordinate, dtype=np.float64)
    centroid = np.asarray(operator.grid.coordinate, dtype=np.float64)[
        : len(centroid_flux)
    ]
    in_wall = _points_inside_polygon(
        centroid[:, 0], centroid[:, 1], wall[:, 0], wall[:, 1]
    )
    polarity = int(operator.polarity)
    beyond_separatrix = (polarity * centroid_flux) < (polarity * x_flux)
    return in_wall & beyond_separatrix


def _kernel_read_function(operator: Any, pitch: float, private_mask: np.ndarray):
    """Build the loop-free jitted census kernel.

    Every stage is a closed-form vector expression over all cells: the
    seven-value gather and one einsum to six coefficients, the stationary
    point and Hessian in closed form, the raw and typed flags as arrays, a
    fixed-capacity top-k on the cell-distance score for dedupe, one
    point-in-polygon against the wall, and the private-region mask.
    """

    stencil = _support_stencil(operator)
    gather = jnp.asarray(stencil.ring_gather_index, dtype=jnp.int32)
    weight = jnp.asarray(stencil.ring_flux_weight, dtype=jnp.float64)
    centre = jnp.asarray(stencil.ring_sampling_centre, dtype=jnp.float64)
    scale = jnp.asarray(stencil.ring_coordinate_scale, dtype=jnp.float64)
    edge_start, edge_end, edge_valid = _cell_edges(operator)
    edge_start = jnp.asarray(edge_start)
    edge_end = jnp.asarray(edge_end)
    edge_valid = jnp.asarray(edge_valid)
    wall = jnp.asarray(
        operator._fixed_design_topology.wall.coordinate, dtype=jnp.float64
    )
    capacity = int(operator._fixed_design_topology.grid.locator.maxsize)
    physical_count = int(operator.physical_node_number)
    axis_kind = operator._fixed_design_topology.grid.extremum_polarity
    if axis_kind is None:
        raise RuntimeError("the production read has no declared axis polarity")
    private = jnp.asarray(private_mask, dtype=bool)

    def retain_top_k(admitted: jax.Array, score: jax.Array) -> jax.Array:
        _, top = jax.lax.top_k(jnp.where(admitted, score, -jnp.inf), capacity)
        taken = jnp.zeros_like(admitted).at[top].set(True)
        return admitted & taken

    def kernel(state: jax.Array) -> dict[str, jax.Array]:
        state = jnp.asarray(state, dtype=jnp.float64)
        physical = state[:physical_count]
        centroid, _ = operator._fixed_design_topology.split_flux_map(physical)
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
        local = step / scale
        position = centre + step
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
        saddle_type = determinant < -1.0e-12
        extremum_type = (determinant > 1.0e-12) & (
            jnp.where(h00 + h11 < 0.0, 1, -1) == axis_kind
        )
        extremal_centroid = jnp.argmax(operator.polarity * centroid)
        axis_seed = jnp.arange(centroid.shape[0]) == extremal_centroid
        typed_extremum = axis_seed & finite & near_cell & extremum_type
        typed_saddle = finite & near_cell & saddle_type

        contained = finite & _points_inside_polygon(
            position[:, 0], position[:, 1], wall[:, 0], wall[:, 1]
        )
        contained_extremum = typed_extremum & contained
        contained_saddle = typed_saddle & contained
        candidate_rows = jnp.column_stack((position, value, jnp.zeros_like(value)))
        representative_extremum = retain_top_k(contained_extremum, -cell_distance)
        representative_saddle = retain_top_k(contained_saddle, -cell_distance)
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
        representative_private = private[extremum_index]
        qualified_extremum = extremum_valid & ~representative_private
        qualified_extremum_rows = jnp.where(
            qualified_extremum[:, None], extremum_rows, jnp.nan
        )
        return {
            "crossing_count": crossing,
            "raw_extremum": raw_extremum,
            "raw_saddle": raw_saddle,
            "axis_seed": axis_seed,
            "typed_extremum": typed_extremum,
            "typed_saddle": typed_saddle,
            "contained_extremum": contained_extremum,
            "contained_saddle": contained_saddle,
            "representative_extremum": representative_extremum,
            "representative_saddle": representative_saddle,
            "qualified_extremum": qualified_extremum,
            "qualified_extremum_rows": qualified_extremum_rows,
            "contained_saddle_rows": saddle_rows,
            "raw_position": position,
            "raw_value": value,
            "inside_cell": inside_cell,
            "distance_to_cell_m": cell_distance,
            "hessian_determinant_local": determinant,
            "extremum_count": extremum_count,
            "saddle_count": saddle_count,
            "extremum_overflow": extremum_count > capacity,
            "saddle_overflow": saddle_count > capacity,
        }

    return jax.jit(kernel)


def _reference_staged_read(operator: Any, pitch: float, final_stage: int):
    """Build the jitted production census up to one named stage.

    ``final_stage`` is a plain Python integer baked at trace time, so each
    build traces exactly the prefix whose results it returns; the stages
    after the cut are dead-code eliminated.  Stage order follows the
    production read: raw census, polish, containment, dedupe, private
    exclusion.  The full read reproduces the production implementation
    verbatim.
    """

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
        if final_stage <= 0:
            return {
                "crossing_count": crossing,
                "raw_extremum": raw_extremum,
                "raw_saddle": raw_saddle,
            }

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
        if final_stage <= 1:
            return {
                "crossing_count": crossing,
                "raw_extremum": raw_extremum,
                "raw_saddle": raw_saddle,
                "typed_extremum": finite,
                "typed_saddle": finite,
                "raw_position": position,
                "raw_value": value,
                "requested_step_m": requested_distance,
                "hessian_determinant_local": determinant,
            }

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
        if final_stage <= 2:
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
                "raw_position": position,
                "raw_value": value,
                "requested_step_m": requested_distance,
                "inside_cell": inside_cell,
                "distance_to_cell_m": cell_distance,
                "hessian_determinant_local": determinant,
            }

        representative_extremum = _deduplicate(
            position, contained_extremum, 0.5 * pitch, cell_distance
        )
        representative_saddle = _deduplicate(
            position, contained_saddle, 0.5 * pitch, cell_distance
        )
        extremum_count = jnp.sum(representative_extremum, dtype=jnp.int32)
        saddle_count = jnp.sum(representative_saddle, dtype=jnp.int32)
        if final_stage <= 3:
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


def _kernel_instrument_controls(
    kernel: Any, machine: Any, operator: Any, pitch: float
) -> dict[str, Any]:
    """Make the kernel observe manufactured nulls at one cell centroid."""

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
    saddle = jax.block_until_ready(kernel(jnp.asarray(saddle_state)))
    extremum = jax.block_until_ready(kernel(jnp.asarray(extremum_state)))
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


def _cold_compile_seconds(function: Any, operand: jax.Array) -> float:
    started = perf_counter()
    jax.block_until_ready(function(operand))
    return perf_counter() - started


def _timed_samples(function: Any, operand: jax.Array) -> list[float]:
    jax.block_until_ready(function(operand))
    samples = []
    for _ in range(TIMING_REPEATS):
        started = perf_counter()
        jax.block_until_ready(function(operand))
        samples.append(perf_counter() - started)
    return samples


def _hlo_instruction_count(function: Any, operand: jax.Array) -> int:
    """Return total HLO instructions across the compiled module's computations."""

    module = function.lower(operand).compiler_ir(dialect="hlo").as_hlo_module()
    return sum(
        len(list(computation.instructions())) for computation in module.computations()
    )


def _load_machine(requested_cells: int):
    """Load one timing rung and its reference span."""

    case_name = certificate.DIVERTED_CASE_NAME
    machine, operator, state, exact = _machine_and_field(case_name, requested_cells)
    pitch = math.sqrt(float(np.median(np.asarray(machine.area, dtype=np.float64))))
    axis_reference = np.asarray(exact.magnetic_axis, dtype=np.float64)
    axis_flux = float(
        limiter_audit._exact_flux(case_name, exact, axis_reference[None, :])[0]
    )
    x_reference = np.asarray(exact.x_point, dtype=np.float64)
    x_flux = float(limiter_audit._exact_flux(case_name, exact, x_reference[None, :])[0])
    span = abs(axis_flux - x_flux)
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("the analytic reference span is not positive")
    return case_name, machine, operator, state, exact, pitch, x_flux, span


def _verify_census(
    kernel: Any,
    reference: Any,
    state: np.ndarray,
    pitch: float,
    x_flux: float,
    span: float,
) -> dict[str, Any]:
    """Verify the kernel reproduces the production census and admits the nulls."""

    measured = jax.block_until_ready(reference(jnp.asarray(state)))
    result = jax.block_until_ready(kernel(jnp.asarray(state)))
    booleans = (
        "raw_extremum",
        "raw_saddle",
        "typed_extremum",
        "typed_saddle",
        "contained_extremum",
        "contained_saddle",
    )
    kernel_counts = {
        field: int(np.count_nonzero(np.asarray(result[field]))) for field in booleans
    }
    reference_counts = {
        field: int(np.count_nonzero(np.asarray(measured[field]))) for field in booleans
    }
    flags_equal = all(
        bool(np.array_equal(np.asarray(result[field]), np.asarray(measured[field])))
        for field in booleans
    )
    positions_equal = bool(
        np.allclose(
            np.asarray(result["raw_position"]),
            np.asarray(measured["raw_position"]),
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    hessian_equal = bool(
        np.allclose(
            np.asarray(result["hessian_determinant_local"]),
            np.asarray(measured["hessian_determinant_local"]),
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    axis = _selected_error(
        _finite_rows(result["qualified_extremum_rows"]),
        ANALYTIC_AXIS,
        x_flux,
        pitch,
        span,
    )
    saddle = _selected_error(
        _finite_rows(result["contained_saddle_rows"]), ANALYTIC_X, x_flux, pitch, span
    )
    return {
        "kernel_counts": kernel_counts,
        "reference_counts": reference_counts,
        "counts_match_reference": kernel_counts == reference_counts,
        "flag_arrays_equal": flags_equal,
        "positions_equal_atol_1e12": positions_equal,
        "hessian_equal_atol_1e12": hessian_equal,
        "axis_admitted": axis["admitted"],
        "axis_position_error_in_pitch": axis["position_error_in_pitch"],
        "saddle_admitted": saddle["admitted"],
        "saddle_position_error_in_pitch": saddle["position_error_in_pitch"],
        "kernel_representative_saddle_count": int(result["saddle_count"]),
        "kernel_representative_extremum_count": int(result["extremum_count"]),
        "kernel_qualified_extremum_count": int(
            np.count_nonzero(np.asarray(result["qualified_extremum"]))
        ),
    }


def _cpu_part(requested_cells: int, report_directory: Path) -> dict[str, Any]:
    """Measure and persist one CPU verification part."""

    case_name, machine, operator, state, _exact, pitch, x_flux, span = _load_machine(
        requested_cells
    )
    centroid_flux = np.asarray(state[: len(machine.node)], dtype=np.float64)
    private_mask = _private_region_mask(operator, centroid_flux, x_flux)
    kernel = _kernel_read_function(operator, pitch, private_mask)
    reference = _reference_staged_read(operator, pitch, 4)
    verification = _verify_census(kernel, reference, state, pitch, x_flux, span)
    instrument = _kernel_instrument_controls(kernel, machine, operator, pitch)
    part_path = (
        report_directory / "parts" / f"census-kernel-cpu-cells-{requested_cells}.json"
    )
    part = {
        "schema": "nova.census-kernel-cpu-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "allocation": _allocation("cpu"),
        "verification": verification,
        "kernel_instrument_controls": instrument,
        "private_region_mask_cell_count": int(np.count_nonzero(private_mask)),
        "completed": True,
    }
    _write_json(part_path, part)
    instrument_ok = (
        instrument["saddle_detected_and_polished"]
        and instrument["extremum_detected_and_polished"]
    )
    print(
        "CENSUS_KERNEL_CPU "
        f"requested={requested_cells} realised={len(machine.node)} "
        f"counts_match={verification['counts_match_reference']} "
        f"axis={verification['axis_admitted']} "
        f"saddle={verification['saddle_admitted']} instrument={instrument_ok}",
        flush=True,
    )
    return part


def _gpu_part(requested_cells: int, report_directory: Path) -> dict[str, Any]:
    """Time the kernel and profile the reference stages at one rung."""

    case_name, machine, operator, state, _exact, pitch, x_flux, span = _load_machine(
        requested_cells
    )
    centroid_flux = np.asarray(state[: len(machine.node)], dtype=np.float64)
    private_mask = _private_region_mask(operator, centroid_flux, x_flux)
    kernel = _kernel_read_function(operator, pitch, private_mask)
    stages = [_reference_staged_read(operator, pitch, stage) for stage in range(5)]
    reference = stages[4]
    verification = _verify_census(kernel, reference, state, pitch, x_flux, span)
    instrument = _kernel_instrument_controls(kernel, machine, operator, pitch)

    batched = jnp.broadcast_to(
        jnp.asarray(state, dtype=jnp.float64), (TIMING_BATCH_SIZE, len(state))
    )
    kernel_batch = jax.jit(jax.vmap(kernel))
    kernel_compile_seconds = _cold_compile_seconds(kernel_batch, batched)
    kernel_samples = _timed_samples(kernel_batch, batched)
    kernel_hlo = _hlo_instruction_count(kernel_batch, batched)

    stage_batches = [jax.jit(jax.vmap(stage)) for stage in stages]
    reference_compile_seconds = _cold_compile_seconds(stage_batches[4], batched)
    stage_samples = {
        STAGE_NAMES[i]: _timed_samples(batch, batched)
        for i, batch in enumerate(stage_batches)
    }
    accumulated = {
        name: float(np.median(samples)) for name, samples in stage_samples.items()
    }
    order = [(name, accumulated[name]) for name in STAGE_NAMES]
    marginal = {}
    for index, (name, total) in enumerate(order):
        previous = order[index - 1][1] if index else 0.0
        marginal[name] = max(total - previous, 0.0)
    dominant_stage = max(marginal, key=marginal.get)
    reference_median = accumulated[STAGE_NAMES[-1]]

    kernel_median = float(np.median(kernel_samples))
    per_state_ms = 1.0e3 * kernel_median / TIMING_BATCH_SIZE
    per_cell_us = per_state_ms * 1.0e3 / len(machine.node)
    part_path = (
        report_directory / "parts" / f"census-kernel-gpu-cells-{requested_cells}.json"
    )
    part = {
        "schema": "nova.census-kernel-gpu-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "allocation": _allocation("gpu"),
        "verification": verification,
        "kernel_instrument_controls": instrument,
        "private_region_mask_cell_count": int(np.count_nonzero(private_mask)),
        "kernel": {
            "state_count": TIMING_BATCH_SIZE,
            "repeat_count": TIMING_REPEATS,
            "compile_seconds": kernel_compile_seconds,
            "steady_batch_samples": kernel_samples,
            "batch_seconds_median": kernel_median,
            "batch_seconds_per_state_median": 1.0e-3 * per_state_ms,
            "batch_seconds_per_state_ms": per_state_ms,
            "per_cell_per_state_us": per_cell_us,
            "whole_solve_target_ms": WHOLE_SOLVE_TARGET_MS,
            "per_state_ms_over_target": per_state_ms / WHOLE_SOLVE_TARGET_MS,
            "hlo_instruction_count": kernel_hlo,
        },
        "reference": {
            "state_count": TIMING_BATCH_SIZE,
            "repeat_count": TIMING_REPEATS,
            "compile_seconds": reference_compile_seconds,
            "batch_seconds_median": reference_median,
            "batch_seconds_per_state_ms": 1.0e3 * reference_median / TIMING_BATCH_SIZE,
            "hlo_instruction_count": _hlo_instruction_count(stage_batches[4], batched),
            "stage_accumulated_seconds": {
                name: {
                    "median_seconds": accumulated[name],
                    "samples_seconds": stage_samples[name],
                }
                for name in STAGE_NAMES
            },
            "stage_marginal_seconds": marginal,
            "dominant_stage": dominant_stage,
            "dominant_stage_share": marginal[dominant_stage] / reference_median,
        },
        "completed": True,
    }
    _write_json(part_path, part)
    print(
        "CENSUS_KERNEL_GPU "
        f"requested={requested_cells} realised={len(machine.node)} "
        f"kernel_ms_per_state={per_state_ms:.6g} "
        f"reference_ms_per_state={1e3 * reference_median / TIMING_BATCH_SIZE:.6g} "
        f"kernel_hlo={kernel_hlo} dominant={dominant_stage}",
        flush=True,
    )
    return part


def _scaling_exponent(cells: list[int], per_state_ms: list[float]) -> float:
    """Return the least-squares exponent linking per-state time to cell count."""

    log_cells = np.log(np.asarray(cells, dtype=np.float64))
    log_time = np.log(np.asarray(per_state_ms, dtype=np.float64))
    fit = np.polyfit(log_cells, log_time, 1)
    return float(fit[0])


def _render_figure(
    rows: list[dict[str, Any]],
    figure_directory: Path,
    report_directory: Path,
) -> Path:
    """Render the timing ladder and the reference stage attribution."""

    figure_directory.mkdir(parents=True, exist_ok=True)
    cells = [row["realised_cells"] for row in rows]
    kernel_per_state = [row["kernel"]["batch_seconds_per_state_ms"] for row in rows]
    reference_per_state = [
        row["reference"]["batch_seconds_per_state_ms"] for row in rows
    ]
    production_per_state = [
        PRODUCTION_TIMING_MS[row["requested_cells"]] for row in rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].loglog(
        cells, kernel_per_state, "o-", color=COLOURS["kernel"], label="kernel"
    )
    axes[0].loglog(
        cells,
        production_per_state,
        "^-.",
        color=COLOURS["production"],
        label="production read (banked)",
    )
    axes[0].loglog(
        cells,
        reference_per_state,
        "s--",
        color=COLOURS["reference"],
        label="reference read (direct)",
    )
    axes[0].axhline(
        WHOLE_SOLVE_TARGET_MS, color=COLOURS["target"], linestyle=":", linewidth=1.2
    )
    axes[0].text(
        cells[0],
        WHOLE_SOLVE_TARGET_MS * 1.25,
        "1 ms whole-solve target",
        color=COLOURS["target"],
        fontsize=8,
    )
    for cell, kernel_ms, reference_ms, production_ms in zip(
        cells, kernel_per_state, reference_per_state, production_per_state
    ):
        axes[0].annotate(
            f"{kernel_ms:.3g} ms",
            (cell, kernel_ms),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
        )
        axes[0].annotate(
            f"{production_ms:.3g} ms",
            (cell, production_ms),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[0].set_xlabel("realised cells")
    axes[0].set_ylabel("median ms per state")
    axes[0].set_title("census kernel vs reference read")
    axes[0].legend(fontsize=8)

    stage_names = STAGE_NAMES
    width = 0.26
    for offset, (cells_row, row) in enumerate(zip(cells, rows)):
        marginals = [
            row["reference"]["stage_marginal_seconds"][name] for name in stage_names
        ]
        x_positions = np.arange(len(stage_names)) + (offset - 1) * width
        bars = axes[1].bar(
            x_positions,
            marginals,
            width,
            label=f"realised {cells_row}",
            color=COLOURS["kernel"]
            if offset == 0
            else (COLOURS["reference"] if offset == 1 else COLOURS["target"]),
        )
        for bar, margin in zip(bars, marginals):
            axes[1].annotate(
                f"{1e6 * margin / TIMING_BATCH_SIZE:.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 2),
                fontsize=7,
                ha="center",
            )
    axes[1].set_xticks(np.arange(len(stage_names)))
    axes[1].set_xticklabels(stage_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("marginal seconds per batch")
    axes[1].set_title("reference stage attribution")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure_path = figure_directory / "census-kernel-timing.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return figure_path


def _write_report(report_directory: Path, figure_directory: Path) -> dict[str, Any]:
    """Aggregate the banked parts into the terminal receipt and report."""

    rows = []
    for requested_cells in TIMING_CELL_COUNTS:
        part_path = (
            report_directory
            / "parts"
            / f"census-kernel-gpu-cells-{requested_cells}.json"
        )
        payload = json.loads(part_path.read_text(encoding="utf-8"))
        if not payload.get("completed"):
            raise RuntimeError(f"part receipt is incomplete: {part_path}")
        rows.append(payload)
    cells = [row["realised_cells"] for row in rows]
    kernel_ms = [row["kernel"]["batch_seconds_per_state_ms"] for row in rows]
    reference_ms = [row["reference"]["batch_seconds_per_state_ms"] for row in rows]
    figure_path = _render_figure(rows, figure_directory, report_directory)
    rows_table = []
    for row in rows:
        reference = row["reference"]
        rows_table.append(
            {
                "requested_cells": row["requested_cells"],
                "realised_cells": row["realised_cells"],
                "characteristic_pitch_m": row["characteristic_pitch_m"],
                "kernel_per_state_ms": row["kernel"]["batch_seconds_per_state_ms"],
                "kernel_per_cell_per_state_us": row["kernel"]["per_cell_per_state_us"],
                "kernel_state_ms_over_target": row["kernel"][
                    "per_state_ms_over_target"
                ],
                "kernel_compile_seconds": row["kernel"]["compile_seconds"],
                "kernel_hlo_instruction_count": row["kernel"]["hlo_instruction_count"],
                "reference_per_state_ms": reference["batch_seconds_per_state_ms"],
                "reference_hlo_instruction_count": reference["hlo_instruction_count"],
                "dominant_stage": reference["dominant_stage"],
                "dominant_stage_share": reference["dominant_stage_share"],
                "verification": row["verification"],
            }
        )
    receipt = {
        "schema": "nova.census-kernel-timing-receipt",
        "version": 1,
        "source_revision": _source_revision(),
        "rows": rows_table,
        "whole_solve_target_ms": WHOLE_SOLVE_TARGET_MS,
        "kernel_scaling_exponent_in_cells": _scaling_exponent(cells, kernel_ms),
        "reference_scaling_exponent_in_cells": _scaling_exponent(cells, reference_ms),
        "figure_path": str(figure_path),
    }
    _write_json(report_directory / "receipt.json", receipt)
    with open(report_directory / "report.md", "w", encoding="utf-8") as stream:
        stream.write(_markdown_report(receipt))
    print("CENSUS_KERNEL_REPORT written", flush=True)
    return receipt


def _markdown_report(receipt: dict[str, Any]) -> str:
    """Render the terminal report for a reader without opening the receipt."""

    target_ms = receipt["whole_solve_target_ms"]
    kernel_exponent = receipt["kernel_scaling_exponent_in_cells"]
    reference_exponent = receipt["reference_scaling_exponent_in_cells"]
    lines = [
        "# Census kernel timing",
        "",
        f"Whole-solve target: {target_ms} ms.  ",
        f"Kernel scaling exponent in cells: {kernel_exponent:.2f}.  ",
        f"Reference scaling exponent in cells: {reference_exponent:.2f}.",
        "",
        "| requested | realised | kernel ms/state | kernel µs/cell-state | target "
        "share | reference ms/state | reference/kernel | kernel HLO | reference "
        "HLO | dominant stage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in receipt["rows"]:
        reference_ms = row["reference_per_state_ms"]
        kernel_ms = row["kernel_per_state_ms"]
        lines.append(
            f"| {row['requested_cells']} | {row['realised_cells']} | "
            f"{kernel_ms:.4g} | {row['kernel_per_cell_per_state_us']:.4g} | "
            f"{row['kernel_state_ms_over_target']:.3g} | {reference_ms:.4g} | "
            f"{reference_ms / kernel_ms:.1f} | {row['kernel_hlo_instruction_count']} | "
            f"{row['reference_hlo_instruction_count']} | {row['dominant_stage']} |"
        )
    lines.extend(
        [
            "",
            "Verification",
            "",
            "| rung | counts match | axis | saddle | axis err / pitch | saddle "
            "err / pitch |",
        ]
    )
    lines.append("|---|---|---|---|---|---|")
    for row in receipt["rows"]:
        verification = row["verification"]
        lines.append(
            f"| {row['realised_cells']} | {verification['counts_match_reference']} | "
            f"{verification['axis_admitted']} | {verification['saddle_admitted']} | "
            f"{verification['axis_position_error_in_pitch']:.3g} | "
            f"{verification['saddle_position_error_in_pitch']:.3g} |"
        )
    lines.append("")
    lines.append(f"Figure: {receipt['figure_path']}")
    return "\n".join(lines) + "\n"


def _run_cpu(report_directory: Path) -> None:
    """Verify the kernel against the production census at every timing rung."""

    _allocation("cpu")
    report_directory.mkdir(parents=True, exist_ok=True)
    for requested_cells in TIMING_CELL_COUNTS:
        part = _cpu_part(requested_cells, report_directory)
        verification = part["verification"]
        required = (
            verification["counts_match_reference"]
            and verification["flag_arrays_equal"]
            and verification["axis_admitted"]
            and verification["saddle_admitted"]
        )
        if not required:
            raise RuntimeError(
                f"kernel census verification failed at {requested_cells}"
            )
        instrument = part["kernel_instrument_controls"]
        if not (
            instrument["saddle_detected_and_polished"]
            and instrument["extremum_detected_and_polished"]
        ):
            raise RuntimeError(
                f"manufactured stationary-point controls failed at {requested_cells}"
            )


def _run_gpu(report_directory: Path, figure_directory: Path) -> None:
    """Time the kernel and profile the reference stages on one H200."""

    _allocation("gpu")
    report_directory.mkdir(parents=True, exist_ok=True)
    for requested_cells in TIMING_CELL_COUNTS:
        part = _gpu_part(requested_cells, report_directory)
        if not part["verification"]["counts_match_reference"]:
            raise RuntimeError(
                f"device census verification failed at {requested_cells}"
            )
    _write_report(report_directory, figure_directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-run")
    cpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    gpu = commands.add_parser("gpu-run")
    gpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    gpu.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    return parser


def main() -> None:
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("the census kernel requires extended precision")
    arguments = _parser().parse_args()
    if arguments.command == "cpu-run":
        _run_cpu(arguments.report_directory)
    elif arguments.command == "gpu-run":
        _run_gpu(arguments.report_directory, arguments.figure_directory)


if __name__ == "__main__":
    main()
