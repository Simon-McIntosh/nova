#!/usr/bin/env python3
"""Polish-method accuracy ladder on the analytic hex flux oracle.

Every method is seeded at the analytic null's own cell on each carrier rung
and polishes the stationary position.  Four fitting geometries compete:

``ring_quadratic``
    the production ring quadratic, a six-term quadratic fitted to the seed
    centroids' neighbour ring (centre plus six neighbour centroids),
    normalised by the ring half-width; the stationary point is the closed form
    of the fitted gradient.
``own_node_quadratic``
    a six-term quadratic fitted to the seed cell's centroid plus its six
    authored sampling vertices, normalised by the vertex half-width; this is
    the surface the StencilMesh sampling evaluates.
``two_ring_cubic``
    a ten-term cubic fitted by least squares to the centroid, the own six
    vertices, the six neighbour centroids, and as many of their authored
    vertices as are present, propagated to its stationary point by Newton
    iteration from the seed centroid.
``fixed_point_refinement``
    repeated own-node quadratic fits that re-fit on the cell the current
    estimate falls in until the estimate converges to its own fit cell.

Each method reports, per rung and null, position error in metres and in
characteristic pitch, level error in the reference span, the fitted-polynomial
residual over its support, and the number of cells whose samples entered the
fit.  The same four methods run again on the analytic flux evaluated at a
mesh whose carriers are translated by one third of a pitch, so the dependence
of each method on where the null sits inside its cell is measured twice.

Receipt field glossary:
``field`` is the untranslated carrier, ``shifted`` the translated control.
``axis`` and ``saddle`` hold one error record per method per null, ``control``
the manufactured-quadratic positive control.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import socket
import uuid
from time import perf_counter
from typing import Any

import numpy as np
from matplotlib.path import Path as PolygonPath
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/polish-ladder"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/polish-ladder"
)
WALL_NODE_COUNT = 121
SINGLE_NULL_CELL_COUNTS = (500, 750, 1000, 2500)
STATIC_CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
STATIC_CELL_COUNTS = (300, 1000)
ROWS = tuple(
    (certificate.DIVERTED_CASE_NAME, cells) for cells in SINGLE_NULL_CELL_COUNTS
) + tuple(
    (case_name, cells) for case_name in STATIC_CASES for cells in STATIC_CELL_COUNTS
)
SHIFT_FRACTION = 1.0 / 3.0
SHIFT_DIRECTION = np.array([0.91, 0.42], dtype=np.float64)
PITCH_THRESHOLD = 1.0e-3
MAX_POLISH_ITERATIONS = 20
METHODS = (
    "ring_quadratic",
    "own_node_quadratic",
    "two_ring_cubic",
    "fixed_point_refinement",
)
COLOURS = {
    "ring_quadratic": "#3366cc",
    "own_node_quadratic": "#8a2be2",
    "two_ring_cubic": "#d1495b",
    "fixed_point_refinement": "#1a7f37",
    "threshold": "#999999",
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
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _machine_and_field(case_name: str, requested_cells: int):
    """Load one cached carrier and its exact analytic reference."""

    carrier_case, _source_case, exact = certificate._case(case_name)
    machine = limiter_audit._machine(
        case_name, carrier_case, exact, -requested_cells, WALL_NODE_COUNT
    )
    centroid_flux = limiter_audit._exact_flux(case_name, exact, machine.node)
    if float(np.ptp(centroid_flux)) <= 1.0e-10:
        raise RuntimeError("analytic-flux positive control saw a uniform grid field")
    return machine, exact


def _containing_cell(
    node: np.ndarray, polygons: Any, point: np.ndarray, pitch: float
) -> int:
    """Return the cell containing a reference point, tolerant to one pitch."""

    contained = np.asarray(
        [
            PolygonPath(np.asarray(polygon)).contains_point(
                point, radius=64.0 * np.finfo(np.float64).eps
            )
            for polygon in polygons
        ],
        dtype=bool,
    )
    if not np.any(contained):
        nearest = int(np.argmin(np.linalg.norm(node - point, axis=1)))
        if np.linalg.norm(node[nearest] - point) <= pitch:
            contained[nearest] = True
    return int(np.flatnonzero(contained)[0])


class _Geometry:
    """Translation-invariant carrier structure shared by both fields."""

    def __init__(self, node, vertices, polygons, stencil):
        self.node = np.asarray(node, dtype=np.float64)
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.polygons = list(polygons)
        self.ring_neighbours: dict[int, list[int]] = {}
        if stencil is not None:
            ring = np.asarray(stencil, dtype=np.intp)
            self.ring_neighbours = {
                int(row[0]): [int(value) for value in row[1:]] for row in ring
            }

    def is_ring_centre(self, cell: int) -> bool:
        return cell in self.ring_neighbours

    def ring(self, cell: int) -> list[int]:
        if cell not in self.ring_neighbours:
            raise KeyError(cell)
        return [cell] + self.ring_neighbours[cell]


class _Field:
    """Flux values on the (optionally translated) carrier positions."""

    def __init__(self, node, vertices, centroid_flux, vertex_flux):
        self.node = node
        self.vertex_positions = vertices
        self.centroid_flux = np.asarray(centroid_flux, dtype=np.float64)
        self._vertex_flux = np.asarray(vertex_flux, dtype=np.float64)

    def centroid(self, cell: int) -> float:
        return float(self.centroid_flux[cell])

    def vertices(self, cell: int) -> np.ndarray:
        return np.asarray(self._vertex_flux[cell], dtype=np.float64)


def _carrier_field(
    case_name: str,
    exact: Any,
    machine: Any,
    shift: np.ndarray | None,
) -> tuple[_Geometry, _Field, np.ndarray | None]:
    """Build the geometry and analytic flux for one placement of the carrier."""

    node = np.asarray(machine.node, dtype=np.float64)
    vertices = np.asarray(machine.sampling_vertices, dtype=np.float64)
    polygons = list(machine.cell_polygons)
    if shift is not None:
        node = node + shift
        vertices = vertices + shift
        polygons = [
            np.asarray(polygon, dtype=np.float64) + shift for polygon in polygons
        ]
    centroid_flux = limiter_audit._exact_flux(case_name, exact, node)
    vertex_flux = limiter_audit._exact_flux(
        case_name, exact, vertices.reshape(-1, 2)
    ).reshape(len(node), -1)
    geometry = _Geometry(node, vertices, polygons, machine.stencil)
    return geometry, _Field(node, vertices, centroid_flux, vertex_flux), shift


def _quadratic_fit(
    positions: np.ndarray, values: np.ndarray, centre: np.ndarray, scale: float
) -> np.ndarray:
    """Fit the six-term quadratic 1,u,v,u^2,u.v,v^2 by least squares."""

    local = (positions - centre) / scale
    design = np.column_stack(
        (
            np.ones(len(positions)),
            local[:, 0],
            local[:, 1],
            local[:, 0] ** 2,
            local[:, 0] * local[:, 1],
            local[:, 1] ** 2,
        )
    )
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray(values, dtype=np.float64), rcond=None
    )
    return np.asarray(coefficients, dtype=np.float64)


def _quadratic_stationary(
    coefficients: np.ndarray, centre: np.ndarray, scale: float
) -> tuple[np.ndarray, float, float]:
    """Closed-form stationary point of the fitted quadratic."""

    determinant = 4.0 * coefficients[3] * coefficients[5] - coefficients[4] ** 2
    local_radial = (
        coefficients[4] * coefficients[2] - 2.0 * coefficients[5] * coefficients[1]
    ) / determinant
    local_vertical = (
        coefficients[4] * coefficients[1] - 2.0 * coefficients[3] * coefficients[2]
    ) / determinant
    local = np.array([local_radial, local_vertical], dtype=np.float64)
    position = centre + scale * local
    value = (
        coefficients[0]
        + coefficients[1] * local[0]
        + coefficients[2] * local[1]
        + coefficients[3] * local[0] ** 2
        + coefficients[4] * local[0] * local[1]
        + coefficients[5] * local[1] ** 2
    )
    return position, float(value), float(determinant)


def _polynomial_value(coefficients: np.ndarray, local: np.ndarray) -> float:
    """Evaluate a quadratic or cubic model at one local coordinate."""

    if len(coefficients) == 10:
        u, v = local
        return float(
            coefficients[0]
            + coefficients[1] * u
            + coefficients[2] * v
            + coefficients[3] * u * u
            + coefficients[4] * u * v
            + coefficients[5] * v * v
            + coefficients[6] * u * u * u
            + coefficients[7] * u * u * v
            + coefficients[8] * u * v * v
            + coefficients[9] * v * v * v
        )
    u, v = local
    return float(
        coefficients[0]
        + coefficients[1] * u
        + coefficients[2] * v
        + coefficients[3] * u * u
        + coefficients[4] * u * v
        + coefficients[5] * v * v
    )


def _fit_residual(
    positions: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray,
    centre: np.ndarray,
    scale: float,
) -> float:
    """RMS misfit of the fitted model over its support."""

    local = (positions - centre) / scale
    model = np.asarray([_polynomial_value(coefficients, row) for row in local])
    return float(np.sqrt(np.mean((model - np.asarray(values)) ** 2)))


def _cubic_fit(
    positions: np.ndarray, values: np.ndarray, centre: np.ndarray, scale: float
) -> np.ndarray:
    """Fit the ten-term complete cubic by least squares."""

    local = (positions - centre) / scale
    design = np.column_stack(
        (
            np.ones(len(positions)),
            local[:, 0],
            local[:, 1],
            local[:, 0] ** 2,
            local[:, 0] * local[:, 1],
            local[:, 1] ** 2,
            local[:, 0] ** 3,
            local[:, 0] ** 2 * local[:, 1],
            local[:, 0] * local[:, 1] ** 2,
            local[:, 1] ** 3,
        )
    )
    coefficients, *_ = np.linalg.lstsq(
        design, np.asarray(values, dtype=np.float64), rcond=None
    )
    return np.asarray(coefficients, dtype=np.float64)


def _cubic_stationary(
    coefficients: np.ndarray, centre: np.ndarray, scale: float, pitch: float
) -> tuple[np.ndarray | None, float | None]:
    """Propagate the cubic to a stationary point by Newton iteration."""

    def gradient(local):
        u, v = local
        return np.array(
            [
                coefficients[1]
                + 2.0 * coefficients[3] * u
                + coefficients[4] * v
                + 3.0 * coefficients[6] * u * u
                + 2.0 * coefficients[7] * u * v
                + coefficients[8] * v * v,
                coefficients[2]
                + coefficients[4] * u
                + 2.0 * coefficients[5] * v
                + coefficients[7] * u * u
                + 2.0 * coefficients[8] * u * v
                + 3.0 * coefficients[9] * v * v,
            ]
        )

    def hessian(local):
        u, v = local
        return np.array(
            [
                [
                    2.0 * coefficients[3]
                    + 6.0 * coefficients[6] * u
                    + 2.0 * coefficients[7] * v,
                    coefficients[4]
                    + 2.0 * coefficients[7] * u
                    + 2.0 * coefficients[8] * v,
                ],
                [
                    coefficients[4]
                    + 2.0 * coefficients[7] * u
                    + 2.0 * coefficients[8] * v,
                    2.0 * coefficients[5]
                    + 2.0 * coefficients[8] * u
                    + 6.0 * coefficients[9] * v,
                ],
            ]
        )

    local = np.zeros(2, dtype=np.float64)
    for _ in range(50):
        step = np.linalg.solve(hessian(local), -gradient(local))
        local = local + step
        if float(np.linalg.norm(step)) * scale < 1.0e-12 * pitch:
            break
    position = centre + scale * local
    value = _polynomial_value(coefficients, local)
    if not (np.all(np.isfinite(position)) and np.isfinite(value)):
        return None, None
    return position, value


def _own_node_quadratic(
    field: _Field, geometry: _Geometry, cell: int
) -> tuple[np.ndarray, float, float]:
    """Fit and solve the own-node quadratic on one seed cell."""

    centre = geometry.node[cell]
    positions = np.vstack((centre[None, :], geometry.vertices[cell]))
    values = np.concatenate((np.asarray([field.centroid(cell)]), field.vertices(cell)))
    scale = float(np.max(np.linalg.norm(geometry.vertices[cell] - centre, axis=1)))
    coefficients = _quadratic_fit(positions, values, centre, scale)
    position, value, determinant = _quadratic_stationary(coefficients, centre, scale)
    residual = _fit_residual(positions, values, coefficients, centre, scale)
    return position, value, residual


def _polish(method: str, field: _Field, geometry: _Geometry, cell: int, pitch: float):
    """Polish one stationary point with one method, seeded at its own cell."""

    centre = geometry.node[cell]
    if method == "ring_quadratic":
        if not geometry.is_ring_centre(cell):
            raise RuntimeError(f"seed cell {cell} has no neighbour ring")
        ring = geometry.ring(cell)
        positions = geometry.node[np.asarray(ring, dtype=int)]
        values = field.centroid_flux[np.asarray(ring, dtype=int)]
        scale = float(np.max(np.linalg.norm(positions - centre, axis=1)))
        coefficients = _quadratic_fit(positions, values, centre, scale)
        position, value, determinant = _quadratic_stationary(
            coefficients, centre, scale
        )
        residual = _fit_residual(positions, values, coefficients, centre, scale)
        return {
            "position": position,
            "value": value,
            "residual": residual,
            "fit_cells": len(ring),
            "fit_points": len(ring),
            "converged": bool(np.isfinite(determinant)),
        }
    if method == "own_node_quadratic":
        position, value, residual = _own_node_quadratic(field, geometry, cell)
        return {
            "position": position,
            "value": value,
            "residual": residual,
            "fit_cells": 1,
            "fit_points": 7,
            "converged": True,
        }
    if method == "two_ring_cubic":
        if not geometry.is_ring_centre(cell):
            raise RuntimeError(f"seed cell {cell} has no neighbour ring")
        ring = geometry.ring(cell)
        positions = [centre]
        values = [field.centroid(cell)]
        for member in ring[1:]:
            positions.append(geometry.node[member])
            values.append(field.centroid(member))
            for vertex, vertex_flux in zip(
                geometry.vertices[member], field.vertices(member)
            ):
                positions.append(vertex)
                values.append(vertex_flux)
        for vertex, vertex_flux in zip(geometry.vertices[cell], field.vertices(cell)):
            positions.append(vertex)
            values.append(vertex_flux)
        positions = np.asarray(positions, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        scale = float(np.max(np.linalg.norm(positions - centre, axis=1)))
        coefficients = _cubic_fit(positions, values, centre, scale)
        position, value = _cubic_stationary(coefficients, centre, scale, pitch)
        residual = _fit_residual(positions, values, coefficients, centre, scale)
        return {
            "position": position,
            "value": value,
            "residual": residual,
            "fit_cells": len(ring),
            "fit_points": len(positions),
            "converged": position is not None,
        }
    if method == "fixed_point_refinement":
        estimate_cell = cell
        visited: list[int] = []
        position = None
        value = None
        residual = None
        converged = False
        for _ in range(MAX_POLISH_ITERATIONS):
            candidate, candidate_value, candidate_residual = _own_node_quadratic(
                field, geometry, estimate_cell
            )
            visited.append(estimate_cell)
            position, value, residual = candidate, candidate_value, candidate_residual
            contained = _containing_cell(
                geometry.node, geometry.polygons, candidate, pitch
            )
            if contained == estimate_cell:
                converged = True
                break
            if contained in visited:
                break
            estimate_cell = contained
        return {
            "position": position,
            "value": value,
            "residual": residual,
            "fit_cells": len(visited),
            "fit_points": 7 * len(visited),
            "converged": converged,
            "cells_visited": visited,
        }
    raise ValueError(f"unknown method {method!r}")


class _ManufacturedQuadratic:
    """A pure quadratic field whose stationary point is exactly its centre."""

    def __init__(self, centre: np.ndarray, scale: float, level: float):
        self.centre = np.asarray(centre, dtype=np.float64)
        self.scale = float(scale)
        self.level = float(level)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.level + np.sum((points - self.centre) ** 2, axis=1) / self.scale


def _instrument_controls(
    geometry: _Geometry,
    pitch: float,
    span: float,
    seed_cell: int,
) -> dict[str, Any]:
    """Make every styled method recover one manufactured quadratic null."""

    offset = 0.34 * pitch * np.array([0.9, 0.5], dtype=np.float64)
    probe = geometry.node[seed_cell] + offset
    manufactured = _ManufacturedQuadratic(probe, pitch * pitch, 0.0)
    probe_field = _Field(
        geometry.node,
        geometry.vertices,
        manufactured(geometry.node),
        manufactured(geometry.vertices.reshape(-1, 2)).reshape(
            geometry.vertices.shape[0], -1
        ),
    )
    errors: dict[str, float | None] = {}
    for method in METHODS:
        try:
            polished = _polish(method, probe_field, geometry, seed_cell, pitch)
        except RuntimeError:
            errors[method] = None
            continue
        position = polished["position"]
        if position is None:
            errors[method] = None
            continue
        errors[method] = float(np.linalg.norm(position - probe) / pitch)
    admitted = len(errors) == len(METHODS) and all(
        error is not None and error < 1.0e-6 for error in errors.values()
    )
    return {
        "manufactured_quadratic_null_rz_m": probe.tolist(),
        "errors_in_pitch": _strict(errors),
        "all_admitted_within_1e6_pitch": admitted,
        "span_wb": span,
    }


def _error_record(
    polished: dict[str, Any],
    reference_position: np.ndarray,
    reference_flux: float,
    pitch: float,
    span: float,
) -> dict[str, Any]:
    """One method's error record against one analytic stationary point."""

    position = polished["position"]
    if position is None:
        return {
            "admitted": False,
            "position_error_m": None,
            "position_error_in_pitch": None,
            "level_error_wb": None,
            "level_error_in_span": None,
            "fit_residual_wb": polished.get("residual"),
            "fit_residual_in_span": (
                polished["residual"] / span
                if polished.get("residual") is not None
                else None
            ),
            "fit_cells": polished["fit_cells"],
            "fit_points": polished["fit_points"],
            "converged": polished.get("converged"),
            "cells_visited": polished.get("cells_visited"),
        }
    error = float(np.linalg.norm(position - reference_position))
    level_error = abs(float(polished["value"]) - reference_flux)
    return {
        "admitted": bool(np.isfinite(error)),
        "position_rz_m": position.tolist(),
        "position_error_m": error,
        "position_error_in_pitch": error / pitch,
        "level_error_wb": level_error,
        "level_error_in_span": level_error / span,
        "fit_residual_wb": polished["residual"],
        "fit_residual_in_span": polished["residual"] / span,
        "fit_cells": polished["fit_cells"],
        "fit_points": polished["fit_points"],
        "converged": polished.get("converged"),
        "cells_visited": polished.get("cells_visited"),
    }


def _measure_placement(
    case_name: str,
    requested_cells: int,
    machine: Any,
    exact: Any,
    shift: np.ndarray | None,
    pitch: float,
) -> dict[str, Any]:
    """Measure all four methods at one carrier placement."""

    geometry, field, applied_shift = _carrier_field(case_name, exact, machine, shift)
    diverted = certificate._is_diverted_case(case_name)
    axis_reference = np.asarray(exact.magnetic_axis, dtype=np.float64)
    axis_flux = float(
        limiter_audit._exact_flux(case_name, exact, axis_reference[None, :])[0]
    )
    x_reference = np.asarray(exact.x_point, dtype=np.float64) if diverted else None
    x_flux = (
        float(limiter_audit._exact_flux(case_name, exact, x_reference[None, :])[0])
        if x_reference is not None
        else None
    )
    grid_span = float(np.ptp(field.centroid_flux))
    span = abs(axis_flux - x_flux) if x_flux is not None else grid_span
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("the analytic reference span is not positive")
    axis_seed = _containing_cell(
        geometry.node, geometry.polygons, axis_reference, pitch
    )
    saddle_seed = (
        _containing_cell(geometry.node, geometry.polygons, x_reference, pitch)
        if x_reference is not None
        else None
    )
    axis = {
        method: _error_record(
            _polish(method, field, geometry, axis_seed, pitch),
            axis_reference,
            axis_flux,
            pitch,
            span,
        )
        for method in METHODS
    }
    control = _instrument_controls(geometry, pitch, span, axis_seed)
    if not control["all_admitted_within_1e6_pitch"]:
        raise RuntimeError("a manufactured stationary-point control failed")
    saddle = None
    if x_reference is not None:
        saddle = {
            method: _error_record(
                _polish(method, field, geometry, saddle_seed, pitch),
                x_reference,
                x_flux,
                pitch,
                span,
            )
            for method in METHODS
        }
    return {
        "translation_m": applied_shift.tolist() if applied_shift is not None else None,
        "pitch_m": pitch,
        "reference_span_wb": span,
        "axis_seed_cell": axis_seed,
        "saddle_seed_cell": saddle_seed,
        "axis": axis,
        "saddle": saddle,
        "control": control,
    }


def _measure_row(
    case_name: str, requested_cells: int, report_directory: Path
) -> dict[str, Any]:
    """Measure and persist one analytic-flux carrier row with its shift control."""

    part_path = _part_path(report_directory, case_name, requested_cells)
    progress = {
        "schema": "nova.polish-accuracy-ladder-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "completed": False,
    }
    _write_json(part_path, progress)
    started = perf_counter()
    machine, exact = _machine_and_field(case_name, requested_cells)
    pitch = math.sqrt(float(np.median(np.asarray(machine.area, dtype=np.float64))))
    shift = SHIFT_FRACTION * pitch * SHIFT_DIRECTION
    placement = {}
    for key, translation in (("field", None), ("shifted", shift)):
        placement[key] = _measure_placement(
            case_name, requested_cells, machine, exact, translation, pitch
        )
    row = progress | {
        "allocation": _allocation("cpu"),
        "cache": machine.cache,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "shift_fraction_of_pitch": SHIFT_FRACTION,
        "placement": placement,
        "wall_seconds": perf_counter() - started,
        "completed": True,
    }
    _write_json(part_path, row)
    axis_error = placement["field"]["axis"]["own_node_quadratic"][
        "position_error_in_pitch"
    ]
    print(
        "POLISH_ROW "
        f"case={case_name} requested={requested_cells} realised={len(machine.node)} "
        f"own_node_axis_error_pitch={axis_error} "
        f"seconds={row['wall_seconds']:.3f}",
        flush=True,
    )
    return row


def _run_worker(report_directory: Path, shard_index: int, shard_count: int) -> None:
    """Measure one deterministic shard inside the shared CPU allocation."""

    _allocation("cpu")
    for case_name, requested_cells in ROWS[shard_index::shard_count]:
        _measure_row(case_name, requested_cells, report_directory)


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
    aggregate(report_directory, figure_directory)


def _fitted_orders(
    rungs: list[dict[str, Any]], method: str, null_key: str
) -> dict[str, Any]:
    """Fitted order in pitch between and across successive ladder rungs."""

    points = []
    for row in rungs:
        record = row["placement"]["field"][null_key][method]
        error = record["position_error_in_pitch"]
        if error is None:
            continue
        points.append(
            (row["realised_cells"], float(row["characteristic_pitch_m"]), error)
        )
    if len(points) < 2:
        return {
            "successive": [],
            "overall_slope": None,
            "overall_residual_log": None,
            "pairs": [],
        }
    successive = []
    pairs = []
    for (first_cells, first_pitch, first_error), (
        _second_cells,
        second_pitch,
        second_error,
    ) in zip(points, points[1:]):
        order = math.log(second_error / first_error) / math.log(
            second_pitch / first_pitch
        )
        successive.append(order)
        pairs.append(
            {
                "from_cells": first_cells,
                "to_cells": _second_cells,
                "error_ratio": second_error / first_error,
                "pitch_ratio": second_pitch / first_pitch,
                "order": order,
            }
        )
    log_pitch = np.log(np.asarray([p[1] for p in points]))
    log_error = np.log(np.asarray([p[2] for p in points]))
    slope, intercept = np.polyfit(log_pitch, log_error, 1)
    predicted = slope * log_pitch + intercept
    residual = float(np.sqrt(np.mean((log_error - predicted) ** 2)))
    return {
        "successive": successive,
        "overall_slope": float(slope),
        "overall_residual_log": residual,
        "pairs": pairs,
    }


def _render_polish_ladder(
    rows: list[dict[str, Any]], figure_directory: Path
) -> dict[str, str]:
    """One SVG of position error in pitch against cells per method."""

    single = [row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME]
    all_rows = rows
    figure, axes = plt.subplots(
        2, 1, figsize=(7.6, 7.6), constrained_layout=True, sharex=False
    )
    for panel, (null_key, population, title, label) in enumerate(
        (
            ("axis", all_rows, "axis", "axis null"),
            ("saddle", single, "saddle", "saddle null"),
        )
    ):
        axis = axes[panel]
        for method in METHODS:
            x = []
            y = []
            xs = []
            ys = []
            for row in population:
                record = row["placement"]["field"][null_key][method]
                error = record["position_error_in_pitch"]
                if error is not None:
                    x.append(row["realised_cells"])
                    y.append(error)
                shifted = row["placement"]["shifted"][null_key][method]
                shifted_error = shifted["position_error_in_pitch"]
                if shifted_error is not None:
                    xs.append(row["realised_cells"])
                    ys.append(shifted_error)
            axis.plot(x, y, marker="o", color=COLOURS[method], label=method)
            axis.plot(
                xs,
                ys,
                linestyle="--",
                marker=".",
                color=COLOURS[method],
                linewidth=1.2,
                alpha=0.65,
            )
        axis.axhline(
            PITCH_THRESHOLD,
            color=COLOURS["threshold"],
            linewidth=0.9,
            linestyle=":",
            label=f"{PITCH_THRESHOLD:.0e} pitch",
        )
        axis.set_yscale("log")
        axis.set_xscale("log")
        axis.set_ylabel("position error / pitch")
        axis.set_ylim(bottom=1.0e-6, top=3.0)
        axis.grid(True, which="both", alpha=0.22)
        axis.set_title(label, fontsize=9)
        axis.legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("realised plasma cells")
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / "polish-error-ladder.svg"
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/polish-ladder/"
            "polish-error-ladder.svg"
        ),
    }


def _write_report(receipt: dict[str, Any], figure_directory: Path) -> Path:
    """Write the human-readable polish accuracy table."""

    rows = receipt["rows"]
    single = [row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME]
    lines = [
        "# Polish-method accuracy ladder",
        "",
        (
            "Each styled stationary-point polish is seeded at the analytic "
            "null's own cell on every requested rung and reports position "
            "error in characteristic pitch (realised cell count differs from "
            "the requested nominal).  ``field`` is the carrier as authored; "
            "``shifted`` evaluates the same four methods on the analytic flux "
            "with all carriers translated by one third of a pitch in a "
            "non-axis-aligned direction."
        ),
        "",
        "## Position error in pitch per method, rung and null",
        "",
        "| case | requested | realised | method | axis err/pitch | saddle err/pitch | shifted axis err/pitch |",  # noqa: E501
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    null_keys = ("axis", "saddle")
    for row in rows:
        for method in METHODS:
            cells = []
            for null_key in null_keys:
                placement = row["placement"]["field"]
                if placement[null_key] is None:
                    cells.append("—")
                    continue
                error = placement[null_key][method]["position_error_in_pitch"]
                cells.append(f"{error:.6g}" if error is not None else "—")
            shifted_error = row["placement"]["shifted"]["axis"][method][
                "position_error_in_pitch"
            ]
            cells.append(f"{shifted_error:.6g}" if shifted_error is not None else "—")
            lines.append(
                f"| {row['case']} | {row['requested_cells']} | "
                f"{row['realised_cells']} | {method} | {cells[0]} | {cells[1]} | "
                f"{cells[2]} |"
            )
    lines.extend(
        [
            "",
            "## Fitted order in pitch between successive single-null rungs",
            "",
            (
                "For each method and null, the exponent of position error in "
                "pitch against characteristic pitch is reported between each "
                "successive pair of single-null rungs, and as the overall "
                "log-log slope with its residual."
            ),
            "",
            "| method | null | 500→750 | 750→1000 | 1000→2500 | overall slope | log residual |",  # noqa: E501
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for null_key in null_keys:
            orders = _fitted_orders(single, method, null_key)
            successive = list(orders["successive"]) + [None] * 3
            order_cells = " | ".join(
                f"{value:.4f}" if value is not None else "—" for value in successive[:3]
            )
            slope = orders["overall_slope"]
            residual = orders["overall_residual_log"]
            slope_text = f"{slope:.4f}" if slope is not None else "—"
            residual_text = f"{residual:.4f}" if residual is not None else "—"
            lines.append(
                f"| {method} | {null_key} | {order_cells} | {slope_text} | "
                f"{residual_text} |"
            )
    lines.extend(
        [
            "",
            "## Reach of one thousandth of a pitch",
            "",
            (
                f"A method reaches the {PITCH_THRESHOLD:.0e} threshold on a "
                "rung when its position error falls below a thousandth of a "
                "pitch on that rung."
            ),
            "",
            "| requested | realised | method | null | placement | err/pitch | fits threshold | fit points | fit cells |",  # noqa: E501
            "|---:|---:|---|---|---:|---:|:---:|---:|---:|",
        ]
    )
    for requested in (1000, 2500):
        row = next(
            (
                row
                for row in rows
                if (row["case"], row["requested_cells"]) == (rows[0]["case"], requested)
            ),
            None,
        )
        if row is None:
            continue
        for method in METHODS:
            for null_key in null_keys:
                if row["placement"]["field"][null_key] is None:
                    continue
                for placement_key, label in (
                    ("field", "field"),
                    ("shifted", "shifted"),
                ):
                    record = row["placement"][placement_key][null_key][method]
                    error = record["position_error_in_pitch"]
                    error_text = f"{error:.6g}" if error is not None else "—"
                    fit_text = (
                        "yes" if error is not None and error < PITCH_THRESHOLD else "no"
                    )
                    lines.append(
                        f"| {requested} | {row['realised_cells']} | {method} | "
                        f"{null_key} | {label} | {error_text} | {fit_text} | "
                        f"{record['fit_points']} | {record['fit_cells']} |"
                    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / "polish-accuracy-ladder.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def aggregate(report_directory: Path, figure_directory: Path) -> dict[str, Any]:
    """Aggregate parts, require positive controls, and publish evidence."""

    rows = [_load_part(_part_path(report_directory, *identity)) for identity in ROWS]
    for row in rows:
        for placement_key in ("field", "shifted"):
            if not row["placement"][placement_key]["control"][
                "all_admitted_within_1e6_pitch"
            ]:
                raise RuntimeError("a manufactured stationary-point control failed")
    single = [row for row in rows if row["case"] == certificate.DIVERTED_CASE_NAME]
    by_requested = {row["requested_cells"]: row for row in single}

    def gate(requested: int) -> dict[str, Any]:
        row = by_requested[requested]
        result = {}
        for method in METHODS:
            result[method] = {}
            for null_key in ("axis",) + (
                ("saddle",) if row["placement"]["field"]["saddle"] else ()
            ):
                for placement_key in ("field", "shifted"):
                    record = row["placement"][placement_key][null_key][method]
                    error = record["position_error_in_pitch"]
                    result[method][f"{null_key}:{placement_key}"] = {
                        "realised_cells": row["realised_cells"],
                        "position_error_in_pitch": error,
                        "reaches_pitch_threshold": bool(
                            error is not None and error < PITCH_THRESHOLD
                        ),
                        "fit_points": record["fit_points"],
                        "fit_cells": record["fit_cells"],
                    }
        return result

    fitted = {}
    for method in METHODS:
        fitted[method] = {
            "axis": _fitted_orders(single, method, "axis"),
            "saddle": _fitted_orders(single, method, "saddle"),
        }
    figure = _render_polish_ladder(rows, figure_directory)
    report_destination = _write_report({"rows": rows}, figure_directory)
    receipt = {
        "schema": "nova.polish-accuracy-ladder",
        "version": 1,
        "source_revision": rows[0]["source_revision"],
        "pitch_threshold": PITCH_THRESHOLD,
        "shift_fraction_of_pitch": SHIFT_FRACTION,
        "gates_at_1000_cells": gate(1000),
        "gates_at_2500_cells": gate(2500),
        "fitted_orders": fitted,
        "figure": figure,
        "report": str(report_destination),
        "rows": [_slug(row["case"], row["requested_cells"]) for row in rows],
        "completed": True,
    }
    _write_json(report_directory / "receipt.json", receipt)
    own_node_1000 = gate(1000)["own_node_quadratic"]["axis:field"][
        "reaches_pitch_threshold"
    ]
    own_node_2500 = gate(2500)["own_node_quadratic"]["axis:field"][
        "reaches_pitch_threshold"
    ]
    print(
        "POLISH_AGGREGATE "
        f"rows={len(rows)} own_node_1000_axis={own_node_1000} "
        f"own_node_2500_axis={own_node_2500}",
        flush=True,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu-run")
    cpu.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    cpu.add_argument("--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY)
    cpu.add_argument("--workers", type=int, default=3)
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
    return parser


def main() -> None:
    configure_dtypes()
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
        aggregate(arguments.report_directory, arguments.figure_directory)


if __name__ == "__main__":
    main()
