#!/usr/bin/env python3
"""Measure the production hex topology read on the analytic single-null flux.

The benchmark isolates the fixed-design read from the nonlinear solve.  It
loads or builds each analytic oracle carrier, evaluates the closed-form flux at
the carrier coordinates, and records the fixed-design stationary-point census
and final topology classification.  Rows are persisted independently before
aggregation so a scheduler expiry cannot erase completed rungs.

The CPU allocation owns carrier access, the census, reporting, and figures.
The accelerator allocation only measures warmed single-state and vmapped
sixteen-state reads for the rungs whose saddle was admitted by the CPU census.
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
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/topology-ladder"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/topology-ladder"
)
REQUESTED_CELL_COUNTS = (110, 200, 300, 342, 400, 500, 750, 1000, 2500)
WALL_NODE_COUNT = 121
TIMING_REPEATS = 7
TIMING_BATCH_SIZE = 16
ANALYTIC = certificate.DIVERTED_REFERENCE
ANALYTIC_AXIS = np.asarray(ANALYTIC.magnetic_axis, dtype=np.float64)
ANALYTIC_X = np.asarray(ANALYTIC.x_point, dtype=np.float64)
CASE_NAME = certificate.DIVERTED_CASE_NAME
COLOURS = {
    "analytic": "#3366cc",
    "census": "#8a2be2",
    "ring": "#d1495b",
    "cells": "#666666",
    "admitted": "#2a9d8f",
    "unadmitted": "#d1495b",
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


def _part_path(report_directory: Path, requested_cells: int) -> Path:
    """Return the durable result path for one cell-count rung."""

    return report_directory / "parts" / f"cells-{requested_cells}.json"


def _load_part(path: Path) -> dict[str, Any]:
    """Load one bounded JSON part and require it to be complete."""

    if path.stat().st_size > 1_000_000:
        raise RuntimeError(f"part receipt exceeds one megabyte: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("completed"):
        raise RuntimeError(f"part receipt is incomplete: {path}")
    return payload


def _allocation(kind: str) -> dict[str, Any]:
    """Validate and describe the scheduler allocation for one command."""

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
                f"expected all_debug with JAX_PLATFORMS=cpu, received "
                f"{partition!r} and {platforms!r}"
            )
        if jax.default_backend() != "cpu":
            raise RuntimeError("the CPU measurement selected a non-CPU backend")
    elif kind == "gpu":
        if partition != "betelgeuse" or platforms != "cuda,cpu":
            raise RuntimeError(
                f"expected betelgeuse with JAX_PLATFORMS=cuda,cpu, received "
                f"{partition!r} and {platforms!r}"
            )
        if jax.default_backend() != "gpu":
            raise RuntimeError("the device timing selected a non-GPU backend")
        if len(jax.devices("gpu")) != 1:
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


def _machine_and_field(requested_cells: int) -> tuple[Any, Any, np.ndarray]:
    """Load one single-null carrier, its operator, and closed-form flux."""

    carrier_case, source_case, exact = certificate._case(CASE_NAME)
    machine = limiter_audit._machine(
        CASE_NAME,
        carrier_case,
        exact,
        -requested_cells,
        WALL_NODE_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = limiter_audit._exact_flux(CASE_NAME, exact, coordinates)
    if float(np.ptp(analytic[: len(machine.node)])) <= 1.0e-10:
        raise RuntimeError("analytic-flux positive control saw a uniform grid field")
    operator = limiter_audit.oracle_fixture.forward_operator(source_case, machine)
    return machine, operator, analytic


def _finite_candidate_rows(rows: np.ndarray) -> list[dict[str, Any]]:
    """Return the finite published saddle rows."""

    result = []
    for slot, row in enumerate(np.asarray(rows, dtype=np.float64)):
        if np.all(np.isfinite(row[:3])):
            result.append(
                {
                    "slot": slot,
                    "position_rz_m": row[:2].tolist(),
                    "flux_wb": float(row[2]),
                    "kind": float(row[3]),
                    "position_error_m": float(np.linalg.norm(row[:2] - ANALYTIC_X)),
                }
            )
    return result


def _ring_holding_analytic_saddle(
    machine: Any, operator: Any, grid_flux: np.ndarray, pitch: float
) -> dict[str, Any]:
    """Describe the eligible centroid ring nearest the analytic saddle."""

    locator = operator._fixed_design_topology.grid.locator
    stencil = np.asarray(locator.stencil, dtype=np.intp)
    centres = np.asarray(machine.node, dtype=np.float64)
    origin_cells = stencil[:, 0]
    origin_distance = np.linalg.norm(centres[origin_cells] - ANALYTIC_X, axis=1)
    ring_index = int(np.argmin(origin_distance))
    central_cell = int(origin_cells[ring_index])
    central_polygon = PolygonPath(
        np.asarray(machine.cell_polygons[central_cell], dtype=np.float64)
    )
    contains_analytic_x = bool(
        central_polygon.contains_point(
            ANALYTIC_X, radius=64.0 * np.finfo(np.float64).eps
        )
    )
    cell_indices = stencil[ring_index]
    values = np.asarray(grid_flux, dtype=np.float64)[cell_indices]
    bits = values[1:] > values[0]
    sign_count = int(np.sum(bits != np.roll(bits, -1)))
    return {
        "selection": "nearest_eligible_centroid_ring_origin",
        "central_cell_contains_analytic_x": contains_analytic_x,
        "ring_index": ring_index,
        "central_cell_index": int(central_cell),
        "cell_indices": cell_indices.tolist(),
        "centroid_coordinates_rz_m": centres[cell_indices].tolist(),
        "centroid_flux_wb": values.tolist(),
        "neighbour_above_centre_bits": "".join(
            str(int(value)) for value in bits.tolist()
        ),
        "cyclic_sign_change_count": sign_count,
        "required_saddle_sign_change_count": 4,
        "central_cell_centroid_distance_to_analytic_x_m": float(
            np.linalg.norm(centres[central_cell] - ANALYTIC_X)
        ),
        "central_cell_centroid_distance_to_analytic_x_in_pitch": float(
            np.linalg.norm(centres[central_cell] - ANALYTIC_X) / pitch
        ),
    }


def _miss_stage(
    x_candidate_count: int,
    contained_count: int,
    saddle_admitted: bool,
    ring: dict[str, Any],
) -> dict[str, Any] | None:
    """Name the first fixed-design read step that removes the analytic saddle."""

    if saddle_admitted:
        return None
    if x_candidate_count == 0:
        return {
            "step": "sign_change_census",
            "reason": (
                "the eligible centroid ring nearest the analytic X-point produced "
                f"{ring['cyclic_sign_change_count']} cyclic sign changes rather "
                "than the four required to seed a saddle"
            ),
        }
    if contained_count == 0:
        return {
            "step": "first_wall_containment",
            "reason": "the census produced saddle rows but none survived containment",
        }
    return {
        "step": "class_admission",
        "reason": (
            "a contained saddle survived the census but the final fixed-design "
            "classification did not admit it"
        ),
    }


def _measure_row(requested_cells: int, report_directory: Path) -> dict[str, Any]:
    """Measure and persist one analytic-flux cell-count rung."""

    part_path = _part_path(report_directory, requested_cells)
    progress = {
        "schema": "nova.topology-read-resolution-ladder-part",
        "version": 1,
        "source_revision": _source_revision(),
        "requested_cells": requested_cells,
        "completed": False,
    }
    _write_json(part_path, progress)
    started = perf_counter()
    machine, operator, analytic = _machine_and_field(requested_cells)
    physical = jnp.asarray(analytic[: operator.physical_node_number], dtype=jnp.float64)
    grid_flux, _wall_flux = operator._fixed_design_topology.split_flux_map(physical)
    _masks, topology, _connected, axis_admitted = jax.block_until_ready(
        operator._fixed_design_read(physical)
    )
    (_axis_rows, saddle_rows), census = jax.block_until_ready(
        operator._fixed_design_topology.grid.read_census(grid_flux)
    )
    pitch = math.sqrt(float(np.median(np.asarray(machine.area, dtype=np.float64))))
    exact_axis_flux = float(
        limiter_audit._exact_flux(CASE_NAME, ANALYTIC, ANALYTIC_AXIS[None, :])[0]
    )
    exact_x_flux = float(
        limiter_audit._exact_flux(CASE_NAME, ANALYTIC, ANALYTIC_X[None, :])[0]
    )
    span = abs(exact_axis_flux - exact_x_flux)
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("the analytic axis-to-X flux span is not positive")
    saddle = np.asarray(topology.x_point, dtype=np.float64)
    axis = np.asarray(topology.axis, dtype=np.float64)
    final_class = "diverted" if bool(topology.diverted) else "limited"
    saddle_admitted = bool(final_class == "diverted" and np.all(np.isfinite(saddle)))
    x_candidate_count = int(np.asarray(census["same_root_count"])[1])
    finite_saddles = _finite_candidate_rows(np.asarray(saddle_rows))
    contained = np.asarray(
        operator._fixed_design_topology.contained_x_candidates(saddle_rows),
        dtype=bool,
    )
    ring = _ring_holding_analytic_saddle(
        machine, operator, np.asarray(grid_flux, dtype=np.float64), pitch
    )
    saddle_position_error = (
        float(np.linalg.norm(saddle - ANALYTIC_X)) if saddle_admitted else None
    )
    saddle_level_error = (
        abs(float(topology.x_point_flux) - exact_x_flux) if saddle_admitted else None
    )
    axis_error = float(np.linalg.norm(axis - ANALYTIC_AXIS))
    row = progress | {
        "allocation": _allocation("cpu"),
        "cache": machine.cache,
        "realised_cells": len(machine.node),
        "characteristic_pitch_m": pitch,
        "analytic": {
            "axis_rz_m": ANALYTIC_AXIS.tolist(),
            "x_point_rz_m": ANALYTIC_X.tolist(),
            "axis_flux_wb": exact_axis_flux,
            "x_point_flux_wb": exact_x_flux,
            "axis_to_x_flux_span_wb": span,
            "grid_flux_span_wb": float(np.ptp(np.asarray(grid_flux))),
        },
        "census": {
            "x_candidate_count": x_candidate_count,
            "raw_ring_saddle_count": int(np.asarray(census["raw_ring_count"])[1]),
            "retained_saddle_count": int(np.asarray(census["retained_count"])[1]),
            "contained_saddle_count": int(np.count_nonzero(contained)),
            "overflow": bool(np.asarray(census["overflow"])[1]),
            "saddle_candidates": finite_saddles,
            "analytic_saddle_ring": ring,
        },
        "production_read": {
            "axis_admitted": bool(axis_admitted),
            "class": final_class,
            "saddle_admitted": saddle_admitted,
            "axis_rz_m": axis.tolist(),
            "axis_position_error_m": axis_error,
            "axis_position_error_in_pitch": axis_error / pitch,
            "saddle_rz_m": saddle.tolist() if saddle_admitted else None,
            "saddle_position_error_m": saddle_position_error,
            "saddle_position_error_in_pitch": (
                saddle_position_error / pitch
                if saddle_position_error is not None
                else None
            ),
            "saddle_flux_wb": float(topology.x_point_flux) if saddle_admitted else None,
            "saddle_level_error_wb": saddle_level_error,
            "saddle_level_error_in_span": (
                saddle_level_error / span if saddle_level_error is not None else None
            ),
            "boundary_flux_wb": float(topology.boundary_flux),
        },
        "miss": _miss_stage(
            x_candidate_count,
            int(np.count_nonzero(contained)),
            saddle_admitted,
            ring,
        ),
        "device_timing": None,
        "wall_seconds": perf_counter() - started,
        "completed": True,
    }
    _write_json(part_path, row)
    print(
        "TOPOLOGY_LADDER_ROW "
        f"requested={requested_cells} realised={len(machine.node)} "
        f"pitch_m={pitch:.8g} candidates={x_candidate_count} "
        f"admitted={saddle_admitted} class={final_class} "
        f"ring_signs={ring['cyclic_sign_change_count']} "
        f"seconds={row['wall_seconds']:.3f}",
        flush=True,
    )
    return row


def _run_worker(report_directory: Path, shard_index: int, shard_count: int) -> None:
    """Measure one deterministic shard of the cell-count ladder."""

    _allocation("cpu")
    rows = REQUESTED_CELL_COUNTS[shard_index::shard_count]
    for requested_cells in rows:
        _measure_row(requested_cells, report_directory)


def _draw_cell_context(axis: Any, machine: Any, pitch: float) -> None:
    """Outline cells around the analytic saddle."""

    polygons = []
    for polygon in machine.cell_polygons:
        vertices = np.asarray(polygon, dtype=np.float64)
        if np.linalg.norm(vertices.mean(axis=0) - ANALYTIC_X) <= 2.8 * pitch:
            polygons.append(vertices)
    if polygons:
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors="none",
                edgecolors=COLOURS["cells"],
                linewidths=0.6,
                zorder=3,
            )
        )


def _render_unadmitted_panel(
    row: dict[str, Any], figure_directory: Path
) -> dict[str, str]:
    """Render the analytic field and failed centroid ring for one missed rung."""

    requested_cells = int(row["requested_cells"])
    machine, _operator, _analytic = _machine_and_field(requested_cells)
    pitch = float(row["characteristic_pitch_m"])
    extent = max(4.0 * pitch, 0.27)
    radial = np.linspace(ANALYTIC_X[0] - extent, ANALYTIC_X[0] + extent, 241)
    height = np.linspace(ANALYTIC_X[1] - extent, ANALYTIC_X[1] + extent, 241)
    radius_grid, height_grid = np.meshgrid(radial, height)
    points = np.column_stack((radius_grid.ravel(), height_grid.ravel()))
    field = limiter_audit._exact_flux(CASE_NAME, ANALYTIC, points).reshape(
        height_grid.shape
    )
    levels = poloidal.contour_levels(field, count=14)
    figure, axis = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    poloidal.draw_flux_contours(
        axis, radial, height, field, levels, color=COLOURS["analytic"]
    )
    poloidal.draw_wall(axis, units=(np.asarray(machine.wall_node),), linewidth=0.65)
    analytic_style = DEFAULT_INK.variant(
        axis_marker="^",
        axis_color=COLOURS["analytic"],
        xpoint_color=COLOURS["analytic"],
    )
    poloidal.draw_nulls(
        axis,
        magnetic_axis=ANALYTIC_AXIS,
        x_points=ANALYTIC_X[None, :],
        style=analytic_style,
        contain=(np.asarray(machine.wall_node),),
    )
    for candidate in row["census"]["saddle_candidates"]:
        point = candidate["position_rz_m"]
        axis.plot(
            point[0],
            point[1],
            marker="o",
            markerfacecolor="none",
            markeredgecolor=COLOURS["census"],
            markersize=5,
            linestyle="none",
            zorder=8,
        )
    _draw_cell_context(axis, machine, pitch)
    ring = row["census"]["analytic_saddle_ring"]
    ring_coordinates = np.asarray(ring["centroid_coordinates_rz_m"], dtype=float)
    neighbours = ring_coordinates[1:]
    closed = np.vstack((neighbours, neighbours[0]))
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color=COLOURS["ring"],
        linestyle="--",
        linewidth=1.2,
        zorder=6,
    )
    axis.plot(
        ring_coordinates[0, 0],
        ring_coordinates[0, 1],
        marker="s",
        markerfacecolor="none",
        markeredgecolor=COLOURS["ring"],
        markersize=5,
        linestyle="none",
        zorder=7,
    )
    axis.set_xlim(ANALYTIC_X[0] - extent, ANALYTIC_X[0] + extent)
    axis.set_ylim(ANALYTIC_X[1] - extent, ANALYTIC_X[1] + extent)
    poloidal_axes(axis)
    axis.set_title(
        f"{requested_cells} requested / {row['realised_cells']} realised cells\n"
        f"analytic X ring: {ring['cyclic_sign_change_count']} sign changes; "
        "four required",
        fontsize=9,
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    name = f"unadmitted-cells-{requested_cells}.svg"
    destination = figure_directory / name
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/topology-ladder/" + name
        ),
    }


def _render_error_ladder(
    rows: list[dict[str, Any]], figure_directory: Path
) -> dict[str, str]:
    """Plot admitted saddle error in pitch with missed rungs marked distinctly."""

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    admitted = [row for row in rows if row["production_read"]["saddle_admitted"]]
    missed = [row for row in rows if not row["production_read"]["saddle_admitted"]]
    if admitted:
        axis.plot(
            [row["realised_cells"] for row in admitted],
            [
                row["production_read"]["saddle_position_error_in_pitch"]
                for row in admitted
            ],
            marker="o",
            color=COLOURS["admitted"],
            label="admitted saddle error",
        )
    marker_level = -0.002
    if missed:
        axis.scatter(
            [row["realised_cells"] for row in missed],
            [marker_level] * len(missed),
            marker="x",
            s=48,
            color=COLOURS["unadmitted"],
            label="unadmitted (not zero error)",
            zorder=5,
        )
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set_xscale("log")
    axis.set_xlabel("realised plasma cells")
    axis.set_ylabel("admitted saddle position error / pitch")
    axis.set_ylim(bottom=-0.004)
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False)
    figure_directory.mkdir(parents=True, exist_ok=True)
    destination = figure_directory / "admitted-saddle-error.svg"
    figure.savefig(destination)
    plt.close(figure)
    return {
        "path": str(destination),
        "project_absolute_src": (
            "/nova/figures/cut-cell-current-attribution/topology-ladder/"
            "admitted-saddle-error.svg"
        ),
    }


def _headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the admission threshold and accelerator timing."""

    admitted = [row for row in rows if row["production_read"]["saddle_admitted"]]
    smallest = (
        min(admitted, key=lambda row: row["requested_cells"]) if admitted else None
    )
    timings = [
        row["device_timing"]["vmapped_batch_seconds_per_state_median"]
        for row in admitted
        if row.get("device_timing") is not None
    ]
    return {
        "smallest_admitted_requested_cells": smallest["requested_cells"]
        if smallest
        else None,
        "smallest_admitted_realised_cells": smallest["realised_cells"]
        if smallest
        else None,
        "unadmitted_requested_cells": [
            row["requested_cells"]
            for row in rows
            if not row["production_read"]["saddle_admitted"]
        ],
        "all_unadmitted_miss_steps": sorted(
            {row["miss"]["step"] for row in rows if row["miss"] is not None}
        ),
        "device_timing_complete": bool(admitted)
        and all(row.get("device_timing") is not None for row in admitted),
        "median_vmapped_batch_seconds_per_state_across_admitted_rungs": (
            float(np.median(timings)) if timings else None
        ),
    }


def _write_report(receipt: dict[str, Any], destination: Path) -> None:
    """Write the human-readable threshold and timing report."""

    headline = receipt["headline"]
    rows = receipt["rows"]
    lines = [
        "# Production topology-read resolution ladder",
        "",
        (
            "The benchmark supplies the closed-form single-null flux directly to "
            "the production fixed-design hex read; no nonlinear solve or persisted "
            "terminal state is in the path."
        ),
        "",
    ]
    if headline["smallest_admitted_requested_cells"] is None:
        lines.append("No measured rung admitted the analytic saddle.")
    else:
        lines.append(
            "The smallest measured rung that admits the analytic saddle is "
            f"**{headline['smallest_admitted_requested_cells']} requested cells "
            f"({headline['smallest_admitted_realised_cells']} realised)**."
        )
    lines.extend(["", "## Resolution census", ""])
    lines.append(
        "| requested | realised | pitch (m) | X candidates | admitted | class | "
        "saddle error (m) | error / pitch | level error / span | miss |"
    )
    lines.append("|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---|")
    for row in rows:
        read = row["production_read"]
        miss = row["miss"]
        position_error = read["saddle_position_error_m"]
        pitch_error = read["saddle_position_error_in_pitch"]
        level_error = read["saddle_level_error_in_span"]
        lines.append(
            f"| {row['requested_cells']} | {row['realised_cells']} | "
            f"{row['characteristic_pitch_m']:.8g} | "
            f"{row['census']['x_candidate_count']} | "
            f"{'yes' if read['saddle_admitted'] else 'no'} | {read['class']} | "
            f"{position_error if position_error is not None else '—'} | "
            f"{pitch_error if pitch_error is not None else '—'} | "
            f"{level_error if level_error is not None else '—'} | "
            f"{miss['step'] if miss else '—'} |"
        )
    lines.extend(["", "## Miss mechanism", ""])
    for row in rows:
        if row["miss"] is None:
            continue
        ring = row["census"]["analytic_saddle_ring"]
        lines.append(
            f"- {row['requested_cells']} requested / {row['realised_cells']} "
            f"realised: `{row['miss']['step']}`; ring {ring['ring_index']} "
            f"(central cell {ring['central_cell_index']}) produced "
            f"{ring['cyclic_sign_change_count']} cyclic sign changes with pattern "
            f"`{ring['neighbour_above_centre_bits']}`. {row['miss']['reason']}"
        )
    lines.extend(["", "## H200 read cost", ""])
    timed = [row for row in rows if row.get("device_timing") is not None]
    if not timed:
        lines.append(
            "Device timing is pending; the CPU census and figures are complete."
        )
    else:
        lines.append(
            "| requested | realised | one state median (ms) | batch 16 median "
            "(ms) | batched per state (ms) |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for row in timed:
            timing = row["device_timing"]
            lines.append(
                f"| {row['requested_cells']} | {row['realised_cells']} | "
                f"{1e3 * timing['single_state_seconds_median']:.6g} | "
                f"{1e3 * timing['vmapped_batch_seconds_median']:.6g} | "
                f"{1e3 * timing['vmapped_batch_seconds_per_state_median']:.6g} |"
            )
        median_batch_ms = (
            1e3
            * headline["median_vmapped_batch_seconds_per_state_across_admitted_rungs"]
        )
        lines.extend(
            [
                "",
                "Across admitted rungs, the median vmapped batch cost is "
                f"**{median_batch_ms:.6g} ms per state**.",
            ]
        )
    lines.extend(["", "## Build evidence", ""])
    for requested_cells in (200, 400):
        row = next(item for item in rows if item["requested_cells"] == requested_cells)
        cache = row["cache"]
        lines.append(
            f"- {requested_cells} requested / {row['realised_cells']} realised: "
            f"cache hit `{cache['hit']}`, build {cache['build_seconds']:.6g} s, "
            f"load {cache['load_seconds']:.6g} s, store {cache['store_seconds']:.6g} s."
        )
    lines.extend(
        [
            "",
            "The 500-cell positive control was required to retain at least one X "
            "candidate and admit the saddle; this protects a uniform or empty "
            "instrument from being reported as a resolution threshold.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def aggregate(
    report_directory: Path,
    figure_directory: Path,
    *,
    render_figures: bool,
) -> dict[str, Any]:
    """Aggregate completed parts, assert controls, and optionally render figures."""

    rows = [
        _load_part(_part_path(report_directory, cells))
        for cells in REQUESTED_CELL_COUNTS
    ]
    positive = next(row for row in rows if row["requested_cells"] == 500)
    if positive["census"]["x_candidate_count"] < 1:
        raise RuntimeError("the 500-cell positive control retained no X candidate")
    if not positive["production_read"]["saddle_admitted"]:
        raise RuntimeError("the 500-cell positive control did not admit its saddle")
    figures: list[dict[str, str]] = []
    if render_figures:
        figures.append(_render_error_ladder(rows, figure_directory))
        figures.extend(
            _render_unadmitted_panel(row, figure_directory)
            for row in rows
            if not row["production_read"]["saddle_admitted"]
        )
    else:
        existing = report_directory / "receipt.json"
        if existing.exists() and existing.stat().st_size <= 1_000_000:
            figures = json.loads(existing.read_text(encoding="utf-8")).get(
                "figures", []
            )
    receipt = {
        "schema": "nova.topology-read-resolution-ladder",
        "version": 1,
        "source_revision": _source_revision(),
        "analytic_case": CASE_NAME,
        "analytic_flux_supplied_directly": True,
        "requested_cell_counts": list(REQUESTED_CELL_COUNTS),
        "wall_node_count": WALL_NODE_COUNT,
        "headline": _headline(rows),
        "positive_controls": {
            "analytic_grid_flux_is_nonuniform": True,
            "cells_500_retains_x_candidate": True,
            "cells_500_admits_saddle": True,
            "cells_300_reproduces_zero_x_candidates": bool(
                next(row for row in rows if row["requested_cells"] == 300)["census"][
                    "x_candidate_count"
                ]
                == 0
            ),
        },
        "figures": figures,
        "rows": rows,
    }
    _write_json(report_directory / "receipt.json", receipt)
    _write_report(receipt, report_directory / "report.md")
    smallest_requested = receipt["headline"]["smallest_admitted_requested_cells"]
    smallest_realised = receipt["headline"]["smallest_admitted_realised_cells"]
    print(
        "TOPOLOGY_LADDER_AGGREGATE "
        f"smallest_requested={smallest_requested} "
        f"smallest_realised={smallest_realised} "
        f"timing_complete={receipt['headline']['device_timing_complete']}",
        flush=True,
    )
    return receipt


def run_cpu(report_directory: Path, figure_directory: Path, workers: int) -> None:
    """Run all rungs in subprocess shards inside one CPU allocation."""

    allocation = _allocation("cpu")
    if workers < 1 or workers > allocation["allocated_cpus"]:
        raise ValueError("worker count must fit within the CPU allocation")
    report_directory.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    for shard_index in range(workers):
        log_path = report_directory / f"worker-{shard_index}.log"
        stream = log_path.open("w", encoding="utf-8")
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


def _timed_samples(function: Any, operand: jax.Array) -> list[float]:
    """Measure warmed device completion for one compiled read."""

    jax.block_until_ready(function(operand))
    samples = []
    for _ in range(TIMING_REPEATS):
        started = perf_counter()
        jax.block_until_ready(function(operand))
        samples.append(perf_counter() - started)
    return samples


def _device_timing(requested_cells: int, expected: dict[str, Any]) -> dict[str, Any]:
    """Measure one admitted read as a single state and a vmapped batch."""

    machine, operator, analytic = _machine_and_field(requested_cells)
    physical = jnp.asarray(analytic[: operator.physical_node_number], dtype=jnp.float64)

    def read_one(state: jax.Array) -> tuple[jax.Array, ...]:
        _masks, topology, _connected, admitted = operator._fixed_design_read(state)
        return (
            topology.axis,
            topology.x_point,
            topology.axis_flux,
            topology.x_point_flux,
            topology.boundary_flux,
            topology.diverted,
            admitted,
        )

    single = jax.jit(read_one)
    batch = jax.jit(jax.vmap(read_one))
    batched = jnp.broadcast_to(physical, (TIMING_BATCH_SIZE,) + physical.shape)
    observed = jax.block_until_ready(single(physical))
    observed_saddle = np.asarray(observed[1], dtype=np.float64)
    expected_saddle = np.asarray(
        expected["production_read"]["saddle_rz_m"], dtype=np.float64
    )
    if not np.allclose(observed_saddle, expected_saddle, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("device read does not reproduce the CPU saddle")
    single_samples = _timed_samples(single, physical)
    batch_samples = _timed_samples(batch, batched)
    return {
        "allocation": _allocation("gpu"),
        "state_count": TIMING_BATCH_SIZE,
        "repeat_count": TIMING_REPEATS,
        "single_state_seconds_samples": single_samples,
        "single_state_seconds_median": float(np.median(single_samples)),
        "single_state_seconds_minimum": float(np.min(single_samples)),
        "vmapped_batch_seconds_samples": batch_samples,
        "vmapped_batch_seconds_median": float(np.median(batch_samples)),
        "vmapped_batch_seconds_minimum": float(np.min(batch_samples)),
        "vmapped_batch_seconds_per_state_median": float(np.median(batch_samples))
        / TIMING_BATCH_SIZE,
        "vmapped_batch_seconds_per_state_minimum": float(np.min(batch_samples))
        / TIMING_BATCH_SIZE,
        "device_matches_cpu_saddle_atol_m": 1.0e-12,
        "realised_cells": len(machine.node),
    }


def run_gpu(report_directory: Path, figure_directory: Path) -> None:
    """Time admitted rungs on one accelerator and refresh bounded summaries."""

    _allocation("gpu")
    for requested_cells in REQUESTED_CELL_COUNTS:
        path = _part_path(report_directory, requested_cells)
        row = _load_part(path)
        if not row["production_read"]["saddle_admitted"]:
            continue
        row["device_timing"] = _device_timing(requested_cells, row)
        _write_json(path, row)
        single_ms = 1e3 * row["device_timing"]["single_state_seconds_median"]
        batch_per_state_ms = (
            1e3 * row["device_timing"]["vmapped_batch_seconds_per_state_median"]
        )
        print(
            "TOPOLOGY_LADDER_TIMING "
            f"requested={requested_cells} realised={row['realised_cells']} "
            f"single_ms={single_ms:.6g} "
            f"batch_per_state_ms={batch_per_state_ms:.6g}",
            flush=True,
        )
    aggregate(report_directory, figure_directory, render_figures=False)


def _parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

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
    row = commands.add_parser("row")
    row.add_argument("--cells", type=int, choices=REQUESTED_CELL_COUNTS, required=True)
    row.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY)
    return parser


def main() -> None:
    """Run the selected benchmark command."""

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
    elif arguments.command == "gpu-run":
        run_gpu(arguments.report_directory, arguments.figure_directory)
    elif arguments.command == "aggregate":
        aggregate(
            arguments.report_directory,
            arguments.figure_directory,
            render_figures=arguments.render_figures,
        )
    else:
        _allocation("cpu")
        _measure_row(arguments.cells, arguments.report_directory)


if __name__ == "__main__":
    main()
