"""Measure the local positional stiffness of the analytic Solovev fixture.

The measurement translates the analytic flux without moving either the target
current or the exterior.  It compares the analytic-clipped exterior with the
former whole-cell exterior, applies the production map once, and then starts
the typed production iteration from the translated state.  Every displacement
is persisted before the next one starts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.biot.target import FluxTarget
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "docs/figures/cut-cell-current-attribution/positional-stiffness"
)
DEFAULT_REPORT_PATH = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "s19-handoff/positional-stiffness/report.md"
)
ROWS = (
    ("weak-rotation-reactor-static", -110),
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -300),
)
DISPLACEMENTS_M = np.asarray((0.0, 0.002, 0.005, 0.010, 0.020, 0.040, 0.060))
DIRECTIONS = {
    "outboard": np.asarray((1.0, 0.0), dtype=np.float64),
    "vertical": np.asarray((0.0, 1.0), dtype=np.float64),
}
POSINGS = {
    "analytic_clipped": np.asarray((1.0, 0.0), dtype=np.float64),
    "whole_cell": np.asarray((0.0, 1.0), dtype=np.float64),
}
REQUESTED_CLASS = int(TopologyClass.LIMITED)
LOW_STATE_DISPLACEMENT_M = 0.036
LOW_STATE_RESIDUAL_FRACTION = 8.6e-2
RENDER_INPUT_DIRNAME = "render"
RENDER_RECEIPT_NAME = "render-receipt.json"
TRANSLATED_DISPLACEMENT_M = 0.040
POSING_INKS = {"analytic_clipped": "#3366cc", "whole_cell": "#cc7722"}
POSING_LABELS = {
    "analytic_clipped": "analytic-clipped exterior",
    "whole_cell": "whole-cell exterior",
}
# The flag is read off the production iteration this measurement already ran: the
# state drawn at 40 mm is the iteration's start, so "converged" is whether that
# iteration settles back inside the input displacement, not whether a residual
# taken from a state this panel does not draw passed a tolerance.
CONVERGED_CRITERION = (
    "the production map started from the drawn 40 mm state returns the magnetic "
    "axis inside the input displacement within four trips"
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _pixel_digest(path: Path) -> str:
    """Digest the decoded pixels, so two files that differ only in metadata match,
    and two files that differ on the canvas do not."""

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def _figure_titles(figure) -> list[str]:
    lines = []
    if figure._suptitle is not None and figure._suptitle.get_text():
        lines.extend(figure._suptitle.get_text().splitlines())
    for index, axis in enumerate(figure.axes):
        for line in axis.get_title().splitlines():
            if line:
                lines.append(line)
    return [line for line in lines if line.strip()]


def _null_glyph_panels(figure, counts_by_axis: dict[int, dict[str, dict[str, int]]]):
    panels = []
    for index, axis in enumerate(figure.axes):
        counts = counts_by_axis.get(index)
        if counts is None:
            continue
        panels.append({"panel": index, "null_glyphs": counts})
    return panels


def _lane() -> dict[str, Any]:
    device = jax.devices()[0]
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raise RuntimeError("the stiffness measure requires one scheduler allocation")
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"the stiffness measure requires one H200, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the stiffness measure requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the stiffness measure requires the declared reservation")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the stiffness measure requires eight requested CPUs")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp inside the allocation")
    return {
        "job_id": int(job_id),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "reservation": os.environ["SLURM_JOB_RESERVATION"],
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": device.device_kind,
        "platform": device.platform,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "tmpdir": os.environ["TMPDIR"],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _target_with_exteriors(
    target: FluxTarget, first: np.ndarray, second: np.ndarray
) -> FluxTarget:
    return replace(
        target,
        source_target=jnp.asarray(np.column_stack((first, second))),
    )


def _operator_with_both_exteriors(
    source_case: Any,
    machine: Any,
    analytic_clipped: np.ndarray,
    whole_cell: np.ndarray,
):
    operator = oracle_fixture.forward_operator(
        source_case, machine, np.zeros_like(analytic_clipped)
    )
    grid_count = len(machine.node)
    wall_count = len(machine.wall_node)
    grid_slice = slice(0, grid_count)
    wall_slice = slice(grid_count, grid_count + wall_count)
    sample_slice = slice(grid_count + wall_count, None)
    return replace(
        operator,
        grid=_target_with_exteriors(
            operator.grid, analytic_clipped[grid_slice], whole_cell[grid_slice]
        ),
        wall=_target_with_exteriors(
            operator.wall, analytic_clipped[wall_slice], whole_cell[wall_slice]
        ),
        sample=_target_with_exteriors(
            operator.sample, analytic_clipped[sample_slice], whole_cell[sample_slice]
        ),
        external_current=jnp.asarray(POSINGS["analytic_clipped"]),
    )


def _build_context(case_name: str, requested_cells: int) -> dict[str, Any]:
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    exact_moments, analytic_clipped, clipped_cache = (
        oracle_fixture.cached_fixture_exterior(
            source_case, exact, machine, empty, analytic
        )
    )
    whole_moments = oracle_fixture.whole_cell_current_moments(
        source_case, empty, analytic
    )
    whole_coefficients = empty.coupling_current_moments(whole_moments)
    whole_cell = np.asarray(analytic) - oracle_fixture._internal_flux_image(
        empty, whole_coefficients
    )
    operator = _operator_with_both_exteriors(
        source_case, machine, analytic_clipped, whole_cell
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_moments
    )
    analytic_topology = oracle_probe._topology(operator, analytic)
    span = abs(
        float(analytic_topology["axis_flux_wb"])
        - float(analytic_topology["boundary_flux_wb"])
    )
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("analytic topology did not provide a finite span")
    return {
        "case_name": case_name,
        "requested_cells": requested_cells,
        "source_case": source_case,
        "exact": exact,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": np.asarray(analytic, dtype=np.float64),
        "analytic_topology": analytic_topology,
        "span": span,
        "operator": operator,
        "target_current": float(target_current),
        "current_centroid": np.asarray(centroid, dtype=np.float64),
        "current_receipt": current_receipt,
        "grid_count": len(machine.node),
        "exteriors": {
            "analytic_clipped": {
                "sha256_binary64": _array_digest(analytic_clipped),
                "cache": clipped_cache,
            },
            "whole_cell": {
                "sha256_binary64": _array_digest(whole_cell),
                "definition": (
                    "analytic total flux minus the frozen-block image of the "
                    "analytic density integrated over the former whole-cell support"
                ),
                "moment_sha256_binary64": _array_digest(whole_moments.cell_current),
            },
        },
    }


def _translated_state(context: dict[str, Any], displacement: np.ndarray) -> np.ndarray:
    shifted_coordinates = context["coordinates"] - displacement[None, :]
    return np.asarray(
        certificate._exact_state(
            context["case_name"], context["exact"], shifted_coordinates
        ),
        dtype=np.float64,
    )


def _residual(mapped: np.ndarray, state: np.ndarray, context: dict[str, Any]):
    difference = mapped[: context["grid_count"]] - state[: context["grid_count"]]
    return {
        "rms_fraction_of_span": float(
            np.sqrt(np.mean(difference**2)) / context["span"]
        ),
        "sup_fraction_of_span": float(np.max(np.abs(difference)) / context["span"]),
        "rms_wb": float(np.sqrt(np.mean(difference**2))),
        "sup_wb": float(np.max(np.abs(difference))),
    }


def _axis_displacement(
    topology: dict[str, Any], context: dict[str, Any], direction: np.ndarray
) -> dict[str, Any]:
    axis = np.asarray(topology["axis_rz_m"], dtype=np.float64)
    analytic_axis = np.asarray(
        context["analytic_topology"]["axis_rz_m"], dtype=np.float64
    )
    delta = axis - analytic_axis
    return {
        "axis_rz_m": axis,
        "from_analytic_rz_m": delta,
        "magnitude_m": float(np.linalg.norm(delta)),
        "projected_on_input_direction_m": float(delta @ direction),
    }


def _measure_displacement(
    context: dict[str, Any],
    direction_name: str,
    direction: np.ndarray,
    displacement_m: float,
    posing: str,
    current: np.ndarray,
    part_path: Path,
) -> dict[str, Any]:
    displacement = displacement_m * direction
    state = _translated_state(context, displacement)
    input_topology = oracle_probe._topology(context["operator"], state)
    operator = context["operator"]
    external = operator.external(jnp.asarray(current))
    production_map = jax.jit(
        operator.traced_flux_map(REQUESTED_CLASS, context["target_current"])
    )
    map_started = perf_counter()
    mapped = np.asarray(
        jax.block_until_ready(production_map(jnp.asarray(state), external)),
        dtype=np.float64,
    )
    mapped_topology = oracle_probe._topology(operator, mapped)
    moments, amplitude = operator.normalised_current_moments(
        jnp.asarray(state), context["target_current"], REQUESTED_CLASS
    )
    booked = float(jnp.sum(jax.block_until_ready(moments.cell_current)))
    partial = {
        "case": context["case_name"],
        "requested_cells": context["requested_cells"],
        "realised_cells": context["grid_count"],
        "direction": direction_name,
        "direction_rz": direction,
        "displacement_m": displacement_m,
        "posing": posing,
        "current_selector": current,
        "input": {
            "state_sha256_binary64": _array_digest(state),
            "topology": input_topology,
        },
        "one_map": {
            "residual": _residual(mapped, state, context),
            "axis": _axis_displacement(mapped_topology, context, direction),
            "boundary_level_wb": mapped_topology["boundary_flux_wb"],
            "contact_rz_m": mapped_topology["wall_contact_rz_m"],
            "state_sha256_binary64": _array_digest(mapped),
            "wall_seconds": perf_counter() - map_started,
        },
        "booking": {
            "target_current_a": context["target_current"],
            "booked_current_a": booked,
            "unit_amplitude_booked_current_a": booked / float(amplitude),
            "normalisation_amplitude": float(amplitude),
        },
        "completed": False,
    }
    _write_json(part_path, partial)
    trip_state = state
    partial["iteration"] = {"trips": []}
    for trip in range(1, 5):
        trip_started = perf_counter()
        previous_state = trip_state
        trip_state = np.asarray(
            jax.block_until_ready(
                production_map(jnp.asarray(previous_state), external)
            ),
            dtype=np.float64,
        )
        trip_topology = oracle_probe._topology(operator, trip_state)
        measured_trip = {
            "trip": trip,
            "residual_from_previous": _residual(trip_state, previous_state, context),
            "axis": _axis_displacement(trip_topology, context, direction),
            "boundary_level_wb": trip_topology["boundary_flux_wb"],
            "contact_rz_m": trip_topology["wall_contact_rz_m"],
            "wall_seconds": perf_counter() - trip_started,
            "state_sha256_binary64": _array_digest(trip_state),
        }
        partial["iteration"]["trips"].append(measured_trip)
        if trip == 1:
            partial["iteration"]["after_one_trip"] = measured_trip
        if trip == 4:
            partial["iteration"]["after_four_trips"] = measured_trip
        _write_json(part_path, partial)
    partial["completed"] = True
    _write_json(part_path, partial)
    residual_rms = partial["one_map"]["residual"]["rms_fraction_of_span"]
    print(
        "POSITIONAL_STIFFNESS_PART "
        f"case={context['case_name']} cells={context['requested_cells']} "
        f"direction={direction_name} displacement_mm={1e3 * displacement_m:g} "
        f"posing={posing} rms={residual_rms:.6e}",
        flush=True,
    )
    return partial


def _fit(rows: list[dict[str, Any]], measure: str) -> dict[str, Any]:
    displacement = np.asarray([row["displacement_m"] for row in rows])
    residual = np.asarray(
        [row["one_map"]["residual"][f"{measure}_fraction_of_span"] for row in rows]
    )
    coefficients = np.polyfit(displacement, residual, 2)
    fitted = np.polyval(coefficients, displacement)
    floor = float(residual[0])
    threshold = 10.0 * floor
    exceeded = displacement[residual > threshold]
    return {
        "measure": f"{measure}_fraction_of_span",
        "quadratic_coefficients_descending": coefficients,
        "curvature_span_per_m2": float(2.0 * coefficients[0]),
        "fit_rms": float(np.sqrt(np.mean((fitted - residual) ** 2))),
        "zero_displacement_floor": floor,
        "ten_times_floor": threshold,
        "first_displacement_above_ten_times_floor_m": (
            float(exceeded[0]) if exceeded.size else None
        ),
        "fitted_residual_at_low_state_displacement": float(
            np.polyval(coefficients, LOW_STATE_DISPLACEMENT_M)
        ),
    }


def _draw_residual_figure(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Draw one residual figure from its own per-figure receipt."""

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for axis, direction_name in zip(axes, DIRECTIONS, strict=True):
        for posing in POSINGS:
            series = payload["series"][direction_name][posing]
            displacement_mm = series["displacement_mm"]
            rms = series["rms_fraction_of_span"]
            sup = series["sup_fraction_of_span"]
            axis.plot(
                displacement_mm,
                rms,
                marker="o",
                color=POSING_INKS[posing],
                label=f"{POSING_LABELS[posing]} rms",
            )
            axis.plot(
                displacement_mm,
                sup,
                marker="s",
                linestyle="--",
                color=POSING_INKS[posing],
                label=f"{POSING_LABELS[posing]} sup",
            )
            axis.axhline(rms[0], color=POSING_INKS[posing], alpha=0.28, linewidth=0.8)
        axis.axvline(
            1.0e3 * LOW_STATE_DISPLACEMENT_M,
            color="#7a3e9d",
            linestyle=":",
            linewidth=1.2,
            label="low state: 36 mm",
        )
        axis.axhline(
            LOW_STATE_RESIDUAL_FRACTION,
            color="#7a3e9d",
            linestyle="-.",
            linewidth=1.0,
            label="low state: 8.6e-2 span",
        )
        axis.set_yscale("log")
        axis.set_xlabel(f"{direction_name} displacement [mm]")
        axis.set_ylabel("one-map residual / analytic span")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=6)
    figure.suptitle(
        f"{payload['case']} · {abs(payload['requested_cells'])} requested cells "
        f"({payload['realised_cells']} realised)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    titles = _figure_titles(figure)
    plt.close(figure)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": _file_digest(path),
        "pixel_sha256": _pixel_digest(path),
        "titles": titles,
        "poloidal_panels": [],
    }


def _null_glyph_counts(
    tally: dict[str, int], magnetic_axis: Any, contain: Any
) -> dict[str, int]:
    """The axis marker is not tallied by ``draw_nulls``, so count it here.

    ``draw_nulls`` reports what it drew for every set except the magnetic axis,
    and the panel receipt has to name a count for each set; the predicate below
    is the one that function applies before it plots the axis.
    """

    counted = dict(tally)
    axis_drawn = 0
    if magnetic_axis is not None:
        point = np.asarray(magnetic_axis, dtype=float).reshape(-1)[:2]
        if np.all(np.isfinite(point)):
            inside = poloidal.inside_wall_units(point[None, :], contain)
            axis_drawn = int(bool(np.asarray(inside)[0]))
    counted["axis_drawn"] = axis_drawn
    return counted


def _draw_translated_panel(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Draw the forty-millimetre translated comparison from its own receipt."""

    rows = payload["rows"]
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(8.4, 10.6), constrained_layout=True
    )
    panel_nulls: dict[int, dict[str, dict[str, int]]] = {}
    for row_index, row in enumerate(rows):
        wall = np.asarray(row["wall_units_rz_m"], dtype=np.float64)
        coordinates = np.asarray(row["coordinates_rz_m"], dtype=np.float64)
        radial, height, analytic_field = certificate._raster_field(
            coordinates, np.asarray(row["analytic_flux_wb"], dtype=np.float64), wall
        )
        level_array = row["levels_wb"]
        for column, direction_name in enumerate(DIRECTIONS):
            direction = row["directions"][direction_name]
            _, _, translated_field = certificate._raster_field(
                coordinates,
                np.asarray(direction["state_flux_wb"], dtype=np.float64),
                wall,
            )
            axis = axes[row_index, column]
            poloidal.draw_flux_contours(
                axis, radial, height, analytic_field, level_array, color="#3366cc"
            )
            poloidal.draw_flux_contours(
                axis, radial, height, translated_field, level_array, color="#cc7722"
            )
            poloidal.draw_wall(axis, units=(wall,))
            analytic_set = poloidal.draw_nulls(
                axis,
                magnetic_axis=row["analytic_topology"]["axis_rz_m"],
                x_points=row["analytic_topology"]["x_point_rz_m"],
                style=DEFAULT_INK.variant(
                    axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
                ),
                contain=(wall,),
            )
            translated_set = poloidal.draw_nulls(
                axis,
                magnetic_axis=direction["topology"]["axis_rz_m"],
                x_points=direction["topology"]["x_point_rz_m"],
                style=DEFAULT_INK.variant(
                    axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
                ),
                contain=(wall,),
            )
            panel_nulls.setdefault(row_index * 2 + column, {})
            panel_nulls[row_index * 2 + column]["analytic"] = _null_glyph_counts(
                analytic_set, row["analytic_topology"]["axis_rz_m"], (wall,)
            )
            panel_nulls[row_index * 2 + column]["translated"] = _null_glyph_counts(
                translated_set, direction["topology"]["axis_rz_m"], (wall,)
            )
            poloidal_axes(axis)
            axis.set_title(
                f"{row['case']} · {abs(row['requested_cells'])} requested cells "
                f"({row['realised_cells']} realised)\n"
                f"40 mm {direction_name}: analytic blue / translated ochre\n"
                f"levels Wb (shared): {_levels_text(level_array)}\n"
                f"residual={direction['residual_fraction_of_span']:.3e} of span · "
                f"converged={'yes' if direction['converged'] else 'no'}",
                fontsize=6,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    titles = _figure_titles(figure)
    plt.close(figure)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": _file_digest(path),
        "pixel_sha256": _pixel_digest(path),
        "titles": titles,
        "poloidal_panels": _null_glyph_panels(figure, panel_nulls),
    }


def _levels_text(levels: Any) -> str:
    return "[" + ", ".join(f"{float(level):.4g}" for level in np.asarray(levels)) + "]"


def _residual_slug(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-cells-{abs(requested_cells)}"


def _residual_payload(
    rows: list[dict[str, Any]],
    case_name: str,
    requested_cells: int,
    source_receipt: Path,
) -> dict[str, Any]:
    """The per-figure input for one residual figure: its own rows, nothing else."""

    case_rows = [
        row
        for row in rows
        if row["case"] == case_name and row["requested_cells"] == requested_cells
    ]
    series: dict[str, Any] = {}
    for direction_name in DIRECTIONS:
        series[direction_name] = {}
        for posing in POSINGS:
            ordered = sorted(
                (
                    row
                    for row in case_rows
                    if row["direction"] == direction_name and row["posing"] == posing
                ),
                key=lambda row: row["displacement_m"],
            )
            series[direction_name][posing] = {
                "displacement_mm": [1e3 * row["displacement_m"] for row in ordered],
                "rms_fraction_of_span": [
                    row["one_map"]["residual"]["rms_fraction_of_span"]
                    for row in ordered
                ],
                "sup_fraction_of_span": [
                    row["one_map"]["residual"]["sup_fraction_of_span"]
                    for row in ordered
                ],
            }
    return {
        "slug": _residual_slug(case_name, requested_cells),
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": case_rows[0]["realised_cells"],
        "source_receipt": str(source_receipt.relative_to(ROOT)),
        "series": series,
    }


def _residual_payloads(
    receipt: dict[str, Any], source_receipt: Path
) -> list[dict[str, Any]]:
    return [
        _residual_payload(receipt["rows"], case_name, requested_cells, source_receipt)
        for case_name, requested_cells in ROWS
    ]


def _render_geometry(case_name, requested_cells):
    """The machine and analytic state on its nodes, no operator built."""

    carrier_case, _source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = np.asarray(
        certificate._exact_state(case_name, exact, coordinates), dtype=np.float64
    )
    return machine, coordinates, analytic, exact


def _persisted_topology(output_root, slug, direction_name, posing, displacement_m):
    """Read a topology the committed measurement already recorded."""

    part = (
        output_root
        / "parts"
        / (
            f"{slug}-{direction_name}-{posing}-"
            f"{int(round(1e3 * displacement_m)):03d}mm.json"
        )
    )
    return json.loads(part.read_text(encoding="utf-8"))["input"]["topology"]


def _translated_payload(output_root, rows, source_receipt):
    """The per-figure input for the translated panel: the states it draws.

    The states are stored on the machine coordinates and rasterised at
    render time, so the input carries a few hundred samples per field
    rather than the three-hundred-odd thousand a stored raster needs.
    Both topologies are the ones the committed measurement already read
    for these same states, so nothing is evaluated here.
    """

    entries = []
    for case_name, requested_cells in ROWS:
        machine, coordinates, analytic, exact = _render_geometry(
            case_name, requested_cells
        )
        slug = _residual_slug(case_name, requested_cells)
        wall = np.asarray(machine.wall_node, dtype=np.float64)
        _, _, analytic_raster = certificate._raster_field(coordinates, analytic, wall)
        measured = {
            row["direction"]: row
            for row in rows
            if row["case"] == case_name
            and row["requested_cells"] == requested_cells
            and row["posing"] == "analytic_clipped"
            and abs(row["displacement_m"] - TRANSLATED_DISPLACEMENT_M) < 1e-12
        }
        directions = {}
        for direction_name, direction in DIRECTIONS.items():
            shifted = coordinates - (TRANSLATED_DISPLACEMENT_M * direction[None, :])
            state = np.asarray(
                certificate._exact_state(case_name, exact, shifted),
                dtype=np.float64,
            )
            row = measured[direction_name]
            settled = abs(
                row["iteration"]["after_four_trips"]["axis"][
                    "projected_on_input_direction_m"
                ]
            )
            directions[direction_name] = {
                "state_flux_wb": state,
                "topology": _persisted_topology(
                    output_root,
                    slug,
                    direction_name,
                    "analytic_clipped",
                    TRANSLATED_DISPLACEMENT_M,
                ),
                "residual_fraction_of_span": row["one_map"]["residual"][
                    "rms_fraction_of_span"
                ],
                "converged": bool(settled < TRANSLATED_DISPLACEMENT_M),
            }
        entries.append(
            {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": len(machine.node),
                "coordinates_rz_m": coordinates,
                "wall_units_rz_m": wall,
                "analytic_flux_wb": analytic,
                "analytic_topology": _persisted_topology(
                    output_root, slug, "outboard", "analytic_clipped", 0.0
                ),
                "levels_wb": poloidal.contour_levels(analytic_raster, count=12),
                "directions": directions,
            }
        )
    return {
        "source_receipt": str(source_receipt.relative_to(ROOT)),
        "converged_criterion": CONVERGED_CRITERION,
        "displacement_m": TRANSLATED_DISPLACEMENT_M,
        "rows": entries,
    }


def _render_from_inputs(output_root: Path, render_input: Path) -> dict[str, Any]:
    """Draw every named figure from the committed receipts, with no solve.

    The residual payloads are read back out of the committed measurement
    receipt, one disjoint row selection per figure, and the translated payload
    out of the render input this measurement routes through.
    """

    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    translated_payload = json.loads(render_input.read_text(encoding="utf-8"))
    digest = _file_digest(receipt_path)
    relative_receipt = str(receipt_path.relative_to(ROOT))
    figures = []
    for payload in _residual_payloads(receipt, receipt_path):
        drawn = _draw_residual_figure(
            output_root / "figures" / f"{payload['slug']}-residual.png", payload
        )
        drawn.update(
            {
                "figure": f"{payload['slug']}-residual.png",
                "case": payload["case"],
                "requested_cells": payload["requested_cells"],
                "realised_cells": payload["realised_cells"],
                "source_receipt": relative_receipt,
                "source_receipt_sha256": digest,
            }
        )
        figures.append(drawn)
    translated = _draw_translated_panel(
        output_root / "figures" / "translated-40mm.png", translated_payload
    )
    translated.update(
        {
            "figure": "translated-40mm.png",
            "case": None,
            "source_receipt": relative_receipt,
            "source_receipt_sha256": digest,
        }
    )
    render_receipt = {
        "$id": "nova.fixture-positional-stiffness-render-receipt",
        "revision": _revision(),
        "source_receipt": relative_receipt,
        "source_receipt_sha256": digest,
        "render_input": str(render_input.relative_to(ROOT)),
        "residual_figures": figures,
        "translated_figure": translated,
    }
    _write_json(output_root / RENDER_RECEIPT_NAME, render_receipt)
    return render_receipt


def render_only(output_root: Path, render_input: Path) -> dict[str, Any]:
    """Rebuild the named figures from the committed receipts alone."""

    return _render_from_inputs(output_root, render_input)


def write_render_inputs(output_root: Path) -> Path:
    """Persist the translated render input from one fixture build, with no solve.

    This builds the machine and the analytic state and reads the two translated
    topologies; it never runs the production map or the four-trip iteration.
    """

    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the render input requires extended precision")
    receipt_path = output_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    path = output_root / RENDER_INPUT_DIRNAME / "translated-40mm.json"
    _write_json(path, _translated_payload(output_root, receipt["rows"], receipt_path))
    print("POSITIONAL_STIFFNESS_RENDER_INPUT written", flush=True)
    return path


def _verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_rows = [row for row in rows if row["posing"] == "analytic_clipped"]
    restoration = []
    for row in current_rows:
        initial = row["displacement_m"]
        after_four = abs(
            row["iteration"]["after_four_trips"]["axis"][
                "projected_on_input_direction_m"
            ]
        )
        if initial > 0.0:
            restoration.append(after_four < initial)
    fits = [
        item["fits"]["rms"]
        for item in rows_by_group(rows)
        if item["posing"] == "analytic_clipped"
    ]
    steeper = all(
        item["fitted_residual_at_low_state_displacement"] > LOW_STATE_RESIDUAL_FRACTION
        for item in fits
    )
    restoring = bool(restoration) and all(restoration)
    holds = restoring and steeper
    return {
        "restoring_after_four_trips_at_every_nonzero_displacement": restoring,
        "residual_valley_steeper_than_low_state_at_36mm": steeper,
        "fixture_holds_position": holds,
        "alternative_reposing": (
            "Retain the analytic oracle and pose a non-vacuum exterior family from "
            "the analytic total minus analytically clipped plasma images at nearby "
            "rigid displacements; its displacement derivatives supply an explicit "
            "restoring Taylor term. A fitted vacuum-coil exterior is not admissible "
            "because the Solovev exterior is not a vacuum field."
        ),
    }


def rows_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = []
    identities = dict.fromkeys(
        (row["case"], row["requested_cells"], row["direction"], row["posing"])
        for row in rows
    )
    for case_name, requested_cells, direction, posing in identities:
        selected = sorted(
            (
                row
                for row in rows
                if row["case"] == case_name
                and row["requested_cells"] == requested_cells
                and row["direction"] == direction
                and row["posing"] == posing
            ),
            key=lambda row: row["displacement_m"],
        )
        grouped.append(
            {
                "case": case_name,
                "requested_cells": requested_cells,
                "direction": direction,
                "posing": posing,
                "fits": {"rms": _fit(selected, "rms"), "sup": _fit(selected, "sup")},
            }
        )
    return grouped


def _report(receipt: dict[str, Any]) -> str:
    verdict = receipt["verdict"]
    decision = "holds" if verdict["fixture_holds_position"] else "does not hold"
    lines = [
        "# Fixture positional stiffness",
        "",
        f"The analytic-clipped fixture **{decision} the plasma position** under the "
        "registered criterion: every nonzero displacement must restore after four "
        "production trips and the fitted one-map residual at 36 mm must exceed "
        "8.6e-2 of span in every row and direction. "
        "Restoring="
        f"{verdict['restoring_after_four_trips_at_every_nonzero_displacement']}; "
        f"steeper={verdict['residual_valley_steeper_than_low_state_at_36mm']}.",
        "",
        "| case | cells | direction | posing | rms curvature [span/m2] | "
        "sup curvature [span/m2] | first rms > 10x floor [mm] | rms at 36 mm |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for group in receipt["groups"]:
        rms = group["fits"]["rms"]
        sup = group["fits"]["sup"]
        threshold = rms["first_displacement_above_ten_times_floor_m"]
        threshold_mm = "" if threshold is None else f"{1e3 * threshold:.6g}"
        lines.append(
            f"| {group['case']} | {abs(group['requested_cells'])} | "
            f"{group['direction']} | {group['posing']} | "
            f"{rms['curvature_span_per_m2']:.6g} | "
            f"{sup['curvature_span_per_m2']:.6g} | "
            f"{threshold_mm} | "
            f"{rms['fitted_residual_at_low_state_displacement']:.6g} |"
        )
    lines.extend(["", "## Figures", ""])
    for figure in receipt["residual_figures"]:
        figure_link = figure["project_absolute_src"]
        lines.extend(
            [
                f"![Residual stiffness for {figure['case']}]({figure_link})",
                "",
            ]
        )
    translated = receipt["translated_figure"]
    translated_link = translated["project_absolute_src"]
    lines.extend(
        [
            f"![Forty-millimetre translated analytic states]({translated_link})",
            "",
            "Blue contours and solid triangles are the analytic state; ochre contours "
            "and solid triangles are the rigidly translated state. Every panel uses "
            "shared analytic Wb levels and draws the machine wall.",
            "",
            "## Interpretation",
            "",
            receipt["interpretation"],
            "",
            verdict["alternative_reposing"],
            "",
            "No Nova source file was changed by this measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_root: Path, report_path: Path) -> dict[str, Any]:
    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the stiffness measure requires extended precision")
    set_support_clip_mode("exact")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    receipt_path = output_root / "receipt.json"
    receipt: dict[str, Any] = {
        "$id": "nova.fixture-positional-stiffness",
        "revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _file_digest(Path(__file__)),
        },
        "lane": _lane(),
        "clip_mode": "exact",
        "persistent_compilation_cache": cache.receipt(),
        "displacements_m": DISPLACEMENTS_M,
        "low_state_reference": {
            "displacement_m": LOW_STATE_DISPLACEMENT_M,
            "residual_fraction_of_span": LOW_STATE_RESIDUAL_FRACTION,
        },
        "rows": [],
        "completed": False,
    }
    _write_json(receipt_path, receipt)
    contexts = []
    for case_name, requested_cells in ROWS:
        context = _build_context(case_name, requested_cells)
        contexts.append(context)
        row_slug = f"{case_name}-cells-{abs(requested_cells)}"
        for direction_name, direction in DIRECTIONS.items():
            for posing, current in POSINGS.items():
                for displacement_m in DISPLACEMENTS_M:
                    part_path = (
                        output_root
                        / "parts"
                        / (
                            f"{row_slug}-{direction_name}-{posing}-"
                            f"{int(round(1e3 * displacement_m)):03d}mm.json"
                        )
                    )
                    measured = _measure_displacement(
                        context,
                        direction_name,
                        direction,
                        float(displacement_m),
                        posing,
                        current,
                        part_path,
                    )
                    receipt["rows"].append(measured)
                    _write_json(receipt_path, receipt)
    receipt["groups"] = rows_by_group(receipt["rows"])
    receipt["verdict"] = _verdict(receipt["rows"])
    verdict = receipt["verdict"]
    if verdict["fixture_holds_position"]:
        receipt["interpretation"] = (
            "The analytic-clipped exterior supplies a restoring positional basin: "
            "all measured starts move toward the analytic axis by the fourth trip, "
            "and each fitted residual valley is steeper at 36 mm than the observed "
            "low state's 8.6e-2-of-span separation. The whole-cell posing is retained "
            "only as the historical comparison."
        )
    else:
        receipt["interpretation"] = (
            "The analytic-clipped exterior does not supply the registered restoring "
            "basin. At least one measured start fails to move toward the analytic "
            "axis by the fourth trip or its fitted residual at 36 mm remains below "
            "the low state's 8.6e-2-of-span separation. The two fixed points are "
            "therefore consistent with weak positional holding, not a trusted "
            "certificate basin."
        )
    receipt["completed"] = True
    receipt["elapsed_seconds"] = perf_counter() - started
    _write_json(receipt_path, receipt)
    render_input = output_root / RENDER_INPUT_DIRNAME / "translated-40mm.json"
    _write_json(
        render_input, _translated_payload(output_root, receipt["rows"], receipt_path)
    )
    rendered = _render_from_inputs(output_root, render_input)
    report = _report(
        {
            **receipt,
            "residual_figures": rendered["residual_figures"],
            "translated_figure": rendered["translated_figure"],
        }
    )
    _write_text(output_root / "report.md", report)
    _write_text(report_path, report)
    print(
        f"POSITIONAL_STIFFNESS_EXIT rows={len(receipt['rows'])} completed=True",
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the named figures from the committed receipts, with no solve",
    )
    parser.add_argument(
        "--write-render-inputs",
        action="store_true",
        help="persist the translated render input from one fixture build, no solve",
    )
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    render_input = output_root / RENDER_INPUT_DIRNAME / "translated-40mm.json"
    if arguments.write_render_inputs:
        write_render_inputs(output_root)
        return 0
    if arguments.render_only:
        render_only(output_root, render_input)
        return 0
    receipt = run(output_root, arguments.report_path.resolve())
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
