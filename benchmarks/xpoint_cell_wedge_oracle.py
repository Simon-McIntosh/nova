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
REQUESTED_CELLS = (110, 300, 1000, 2500)
REFERENCE_CELLS = 110
MU_0 = 4.0e-7 * math.pi
CORE_CURRENT_RELATIVE_LIMIT = 1.0e-6
COLOURS = {
    "analytic": "#2563eb",
    "read": "#d97706",
    "cell": "#111827",
    "core": "#0f766e",
    "private": "#7c3aed",
    "sol": "#dc2626",
}


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


def _xpoint_cell(machine: Any, x_point: np.ndarray) -> int:
    containing = [
        index
        for index, polygon in enumerate(machine.cell_polygons)
        if PlotPath(np.asarray(polygon)).contains_point(x_point, radius=1.0e-12)
    ]
    if len(containing) != 1:
        raise RuntimeError(
            f"expected one cell containing the analytic saddle, found {containing}"
        )
    return containing[0]


def _vertical_bounds(vertices: np.ndarray, radius: float) -> tuple[float, float]:
    heights = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        span = float(second[0] - first[0])
        if span == 0.0:
            continue
        fraction = (radius - float(first[0])) / span
        if 0.0 <= fraction <= 1.0:
            heights.append(float(first[1] + fraction * (second[1] - first[1])))
    if len(heights) < 2:
        return 0.0, 0.0
    return min(heights), max(heights)


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
            bottom, top = _vertical_bounds(vertices, radius)
            current = density(radius)
            if moment == 0:
                return current * (top - bottom)
            if moment == 1:
                return current * (radius - centre[0]) * (top - bottom)
            return 0.5 * current * ((top - centre[1]) ** 2 - (bottom - centre[1]) ** 2)

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


def _measure_row(requested_cells: int, output: Path) -> dict[str, Any]:
    started = perf_counter()
    machine, operator, analytic = topology_ladder._machine_and_field(requested_cells)
    exact = topology_ladder.ANALYTIC
    x_point = np.asarray(exact.x_point, dtype=np.float64)
    axis = np.asarray(exact.magnetic_axis, dtype=np.float64)
    cell = _xpoint_cell(machine, x_point)
    polygon = np.asarray(machine.cell_polygons[cell], dtype=np.float64)
    centre = np.asarray(machine.node[cell], dtype=np.float64)
    mesh = AtomicCellMesh.from_cells([polygon], centroids=centre[None, :])
    boundary_flux = float(exact.flux(x_point[None, :])[0])
    axis_flux = float(exact.flux(axis[None, :])[0])
    node_flux = np.asarray(exact.flux(mesh.node_coordinates), dtype=np.float64)
    polarity = math.copysign(1.0, axis_flux - boundary_flux)
    signed_flux = jnp.asarray(polarity * (node_flux - boundary_flux))
    wedges = jax.jit(
        lambda values: mesh.traced_saddle_wedges(
            values,
            saddle_vertex=jnp.asarray(x_point),
            core_reference=jnp.asarray(axis),
        )
    )(signed_flux)
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
    expected = np.asarray(
        [
            _analytic_polygon_moments(_support_vertices(wedges, slot), centre, item)
            for slot, item in enumerate(profiles)
        ]
    )
    scale = np.maximum(np.abs(expected), 1.0e-12)
    relative_error = np.abs(actual - expected) / scale
    core_current_relative_error = float(relative_error[0, 0])
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
        np.array_equal(vertices[slot, count:], 0.0) for slot, count in enumerate(counts)
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
    return {
        "requested_cells": requested_cells,
        "reference": requested_cells == REFERENCE_CELLS,
        "realised_cells": len(machine.node),
        "machine_cache": machine.cache,
        "xpoint_cell": cell,
        "wedge_vertex_capacity": int(wedges.support_vertices.shape[2]),
        "wedge_vertex_count": counts,
        "wedge_area_m2": np.asarray(wedges.area)[0],
        "wedge_area_closure_m2": area_closure,
        "saddle_inserted_as_first_vertex": saddle_inserted,
        "exact_zero_padding": exact_zero_padding,
        "profile_order": ["confined-core", "zero-private", "zero-sol", "zero-sol"],
        "measured_moments": actual,
        "analytic_moments": expected,
        "relative_moment_error": relative_error,
        "core_current_relative_error": core_current_relative_error,
        "private_flux_current_a": float(actual[1, 0]),
        "common_sol_current_a": [float(actual[2, 0]), float(actual[3, 0])],
        "observed_nulls": observed,
        "figure_src": figure_src,
        "wall_seconds": perf_counter() - started,
    }


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    allocation = _allocation()
    rows = [_measure_row(cells, output) for cells in REQUESTED_CELLS]
    tested = [row for row in rows if not row["reference"]]
    receipt = {
        "schema": "nova.xpoint-cell-wedge-oracle",
        "version": 1,
        "source_revision": _source_revision(),
        "allocation": allocation,
        "reference_requested_cells": REFERENCE_CELLS,
        "tested_requested_cells": [row["requested_cells"] for row in tested],
        "core_current_relative_limit": CORE_CURRENT_RELATIVE_LIMIT,
        "max_core_current_relative_error": max(
            row["core_current_relative_error"] for row in tested
        ),
        "private_flux_current_exact_zero": all(
            row["private_flux_current_a"] == 0.0 for row in tested
        ),
        "common_sol_current_exact_zero": all(
            row["common_sol_current_a"] == [0.0, 0.0] for row in tested
        ),
        "all_saddles_inserted": all(
            row["saddle_inserted_as_first_vertex"] for row in rows
        ),
        "all_padding_exact_zero": all(row["exact_zero_padding"] for row in rows),
        "rows": rows,
        "completed": True,
    }
    _write_json(output / "receipt.json", receipt)
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
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
