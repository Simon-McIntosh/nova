"""Discriminate the low exact-clip fixed point on two weak rows.

The emitted receipt keeps four evidence arms.  Arm ``A`` starts the production
iteration at the analytic flux and records the first, second, fourth, and
terminal active-set states.  Arm ``B`` reads the independently persisted
production solve from the current-aligned cold seed.  Arm ``C`` rigidly
translates the analytic flux by arm B's axis displacement and applies the
production limiter read.  Arm ``D`` compares production current booking with
the analytic density integrated over the same terminal support polygons and
with the fixture's analytic clipped moments.

The arm letters are receipt data required by the discriminator contract; code
symbols are named for the mechanism they measure.
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

from benchmarks import oracle_start_newton_probe as oracle_probe
from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-low-state"
)
DEFAULT_REFERENCE_PARTS = (
    ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-seed/parts"
)
DEFAULT_REPORT_PATH = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/"
    "s19-handoff/exact-clip-low-state/report.md"
)
CASE_NAME = "weak-rotation-reactor-static"
REQUESTED_ROWS = (-110, -300)
TRIP_SNAPSHOTS = (1, 2, 4)
REQUESTED_CLASS = int(TopologyClass.LIMITED)
RESIDUAL_BOUND = 1.0e-12
FIXED_DIFFERENCE_LEVELS = np.asarray(
    [
        -1.0e-1,
        -1.0e-2,
        -1.0e-3,
        -3.0e-4,
        -1.0e-4,
        -3.0e-5,
        -1.0e-5,
        1.0e-5,
        3.0e-5,
        1.0e-4,
        3.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
    ],
    dtype=np.float64,
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


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    device = jax.devices()[0]
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raise RuntimeError("the discriminator requires one scheduler allocation")
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"the discriminator requires one H200, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the discriminator requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the discriminator requires the declared reservation")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the discriminator requires eight requested CPUs")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp inside the allocation")
    return {
        "job_id": int(job_id),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": device.device_kind,
        "platform": device.platform,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _reference_part_path(reference_parts: Path, requested_cells: int) -> Path:
    suffix = (
        "production-route-reduced.json"
        if requested_cells == -110
        else "production-route-cells-300.json"
    )
    return reference_parts / f"{CASE_NAME}-{suffix}"


def _load_reference_part(
    reference_parts: Path, requested_cells: int
) -> tuple[dict[str, Any], Path]:
    path = _reference_part_path(reference_parts, requested_cells)
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["case"] != CASE_NAME or int(row["requested_cells"]) != requested_cells:
        raise RuntimeError(f"reference part identity mismatch in {path}")
    if "render_data" not in row:
        raise RuntimeError(f"reference part has no terminal arrays: {path}")
    terminal = np.asarray(row["render_data"]["terminal_flux_wb"], dtype=np.float64)
    analytic = np.asarray(row["render_data"]["analytic_flux_wb"], dtype=np.float64)
    if terminal.shape != analytic.shape or not np.all(np.isfinite(terminal)):
        raise RuntimeError(f"reference terminal arrays are invalid in {path}")
    if np.array_equal(terminal, analytic):
        raise RuntimeError(
            "the reference-part instrument cannot see the known terminal difference"
        )
    return row, path


def _build_context(requested_cells: int) -> dict[str, Any]:
    carrier_case, source_case, exact = certificate._case(CASE_NAME)
    machine = certificate._case_machine(CASE_NAME, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(CASE_NAME, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    analytic_moments, exterior, exterior_cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(source_case, machine, exterior)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        CASE_NAME, source_case, operator, analytic_moments
    )
    analytic_topology = oracle_probe._topology(operator, analytic)
    span = abs(
        float(analytic_topology["axis_flux_wb"])
        - float(analytic_topology["boundary_flux_wb"])
    )
    if not np.isfinite(span) or span <= 0.0:
        raise RuntimeError("analytic topology did not provide a finite flux span")
    return {
        "carrier_case": carrier_case,
        "source_case": source_case,
        "exact": exact,
        "machine": machine,
        "coordinates": coordinates,
        "analytic": analytic,
        "analytic_moments": analytic_moments,
        "exterior_cache": exterior_cache,
        "operator": operator,
        "profile": profile,
        "target_current": float(target_current),
        "current_centroid": np.asarray(centroid, dtype=np.float64),
        "current_receipt": current_receipt,
        "analytic_topology": analytic_topology,
        "span": span,
        "grid_count": len(machine.node),
    }


def _state_metrics(
    context: dict[str, Any],
    state: np.ndarray,
    *,
    residual: float,
    converged: bool,
    trip_count: int,
) -> dict[str, Any]:
    operator = context["operator"]
    analytic = context["analytic"]
    grid_count = context["grid_count"]
    span = context["span"]
    topology = oracle_probe._topology(operator, state)
    difference = np.asarray(state, dtype=np.float64) - analytic
    grid_difference = difference[:grid_count]
    moments, amplitude = operator.normalised_current_moments(
        jnp.asarray(state), context["target_current"], REQUESTED_CLASS
    )
    jax.block_until_ready(moments.cell_current)
    return {
        "trip_count": int(trip_count),
        "terminal_residual": float(residual),
        "converged": bool(converged),
        "axis_flux_wb": topology["axis_flux_wb"],
        "axis_rz_m": topology.get("axis_rz_m"),
        "boundary_level_wb": topology["boundary_flux_wb"],
        "max_solved_minus_analytic_over_span": float(
            np.max(np.abs(grid_difference)) / span
        ),
        "rms_solved_minus_analytic_over_span": float(
            np.sqrt(np.mean(grid_difference**2)) / span
        ),
        "normalisation_amplitude": float(amplitude),
        "state_sha256_binary64": _array_digest(state),
    }


def _solve_prefix(
    context: dict[str, Any], active_set_steps: int
) -> tuple[np.ndarray, dict[str, Any]]:
    request = certificate._certificate_solve_request(
        context["profile"],
        jnp.asarray(context["analytic"]),
        context["target_current"],
        carrier_identity=(
            f"exact-clip-low-state:{CASE_NAME}:"
            f"{context['grid_count']}:{active_set_steps}"
        ),
    )
    request = replace(
        request,
        policy=replace(request.policy, active_set_steps=active_set_steps),
    )
    started = perf_counter()
    solve_receipt = context["profile"].solve(request)
    history = solve_receipt.equilibrium.fixed_point
    jax.block_until_ready(history.state)
    state = np.asarray(solve_receipt.equilibrium.flux, dtype=np.float64)
    metrics = _state_metrics(
        context,
        state,
        residual=float(history.residual),
        converged=bool(history.converged),
        trip_count=int(history.active_set_iterations),
    )
    metrics.update(
        {
            "configured_active_set_steps": active_set_steps,
            "termination": FixedPointTerminationReason(
                int(history.termination_reason)
            ).name.lower(),
            "solve_wall_seconds": perf_counter() - started,
            "active_set_residuals": np.asarray(
                history.active_set_residuals, dtype=np.float64
            ),
            "active_set_mask_differences": np.asarray(
                history.active_set_mask_differences, dtype=np.int64
            ),
        }
    )
    return state, metrics


def _analytic_start_arm(
    context: dict[str, Any], part_path: Path
) -> tuple[dict[str, Any], np.ndarray]:
    snapshots: dict[int, dict[str, Any]] = {}
    states: dict[int, np.ndarray] = {}
    first_state, first_metrics = _solve_prefix(context, 1)
    snapshots[1] = first_metrics
    states[1] = first_state
    partial = {
        "arm": "A",
        "mechanism": "production iteration started at analytic flux",
        "snapshots": {"1": first_metrics},
        "completed": False,
    }
    _write_json(part_path, partial)
    print(
        f"LOW_STATE_ARM_A_PREFIX cells={context['grid_count']} trips=1 "
        f"residual={first_metrics['terminal_residual']:.6e}",
        flush=True,
    )

    terminal_state, terminal_metrics = _solve_prefix(context, 16)
    terminal_trips = terminal_metrics["trip_count"]
    for trip in (2, 4):
        if terminal_trips <= trip:
            states[trip] = terminal_state
            snapshots[trip] = {
                **terminal_metrics,
                "snapshot_rule": (
                    f"production converged or terminated after {terminal_trips} "
                    f"trips, so the state at trip {trip} is the terminal state"
                ),
            }
        else:
            state, metrics = _solve_prefix(context, trip)
            states[trip] = state
            snapshots[trip] = metrics
    arm = {
        "arm": "A",
        "mechanism": "production iteration started at analytic flux",
        "initial": _state_metrics(
            context,
            context["analytic"],
            residual=float("nan"),
            converged=False,
            trip_count=0,
        ),
        "snapshots": {str(trip): snapshots[trip] for trip in TRIP_SNAPSHOTS},
        "terminal": terminal_metrics,
        "terminal_state_sha256_binary64": _array_digest(terminal_state),
        "completed": True,
    }
    _write_json(part_path, arm)
    return arm, terminal_state


def _reference_seed_arm(
    context: dict[str, Any], reference: dict[str, Any], reference_path: Path
) -> tuple[dict[str, Any], np.ndarray]:
    terminal = np.asarray(
        reference["render_data"]["terminal_flux_wb"], dtype=np.float64
    )
    if terminal.shape != context["analytic"].shape:
        raise RuntimeError("reference terminal shape differs from rebuilt operator")
    measured = _state_metrics(
        context,
        terminal,
        residual=float(reference["solver"]["terminal_fixed_point_residual"]),
        converged=bool(reference["solver"]["production_telemetry"]["converged"]),
        trip_count=int(reference["solver"]["production_telemetry"]["trip_count"]),
    )
    arm = {
        "arm": "B",
        "mechanism": "production iteration from current-aligned cold seed",
        "source_part": str(reference_path),
        "source_part_sha256": _file_digest(reference_path),
        "seed": reference["solver"]["seed"],
        "terminal": measured,
        "source_terminal": {
            "axis_error_m": reference["geometry"]["magnetic_axis_position_error_m"],
            "boundary_flux_error_wb": reference["geometry"]["boundary_flux_error_wb"],
            "psi_norms": reference["norms"]["psi"],
        },
        "completed": True,
    }
    return arm, terminal


def _translated_analytic_arm(
    context: dict[str, Any], reference_terminal: np.ndarray
) -> dict[str, Any]:
    reference_topology = oracle_probe._topology(context["operator"], reference_terminal)
    analytic_axis = np.asarray(
        context["analytic_topology"]["axis_rz_m"], dtype=np.float64
    )
    reference_axis = np.asarray(reference_topology["axis_rz_m"], dtype=np.float64)
    displacement = reference_axis - analytic_axis
    shifted_coordinates = context["coordinates"] - displacement[None, :]
    translated = certificate._exact_state(
        CASE_NAME, context["exact"], shifted_coordinates
    )
    translated_topology = oracle_probe._topology(context["operator"], translated)
    reference_boundary = float(reference_topology["boundary_flux_wb"])
    translated_boundary = float(translated_topology["boundary_flux_wb"])
    analytic_boundary = float(context["analytic_topology"]["boundary_flux_wb"])
    mismatch = abs(translated_boundary - reference_boundary)
    reference_offset = abs(reference_boundary - analytic_boundary)
    return {
        "arm": "C",
        "mechanism": (
            "analytic flux translated rigidly by arm B axis displacement, then "
            "read by the production limiter"
        ),
        "axis_displacement_rz_m": displacement,
        "axis_displacement_m": float(np.linalg.norm(displacement)),
        "translated_topology": translated_topology,
        "arm_b_boundary_level_wb": reference_boundary,
        "analytic_boundary_level_wb": analytic_boundary,
        "translated_boundary_level_wb": translated_boundary,
        "translated_minus_arm_b_boundary_wb": translated_boundary - reference_boundary,
        "boundary_mismatch_fraction_of_span": mismatch / context["span"],
        "match_fraction_of_arm_b_offset": (
            mismatch / reference_offset if reference_offset > 0.0 else None
        ),
        "supports_displaced_equilibrium": bool(
            reference_offset > 0.0 and mismatch <= 0.25 * reference_offset
        ),
        "translated_state_sha256_binary64": _array_digest(translated),
        "completed": True,
    }


def _support_classes(support: Any) -> np.ndarray:
    area = np.asarray(support.area, dtype=np.float64)
    full = np.asarray(support.full_area, dtype=np.float64)
    included = np.asarray(support.included, dtype=bool)
    scale = max(float(np.max(np.abs(full))), np.finfo(np.float64).tiny)
    tolerance = 4096.0 * np.finfo(np.float64).eps * scale
    active = included & (area > tolerance)
    whole = active & (np.abs(area - full) <= tolerance)
    values = np.full(len(area), "inactive", dtype=object)
    values[active & ~whole] = "cut"
    values[whole] = "whole"
    return values


def _analytic_current_on_support(source_case: Any, support: Any) -> np.ndarray:
    counts = np.asarray(support.vertex_count, dtype=np.int64)
    vertices = np.asarray(support.support_vertices, dtype=np.float64)
    current = np.zeros(len(counts), dtype=np.float64)
    for cell, count in enumerate(counts):
        if count < 3:
            continue
        points, weights = oracle_fixture._polygon_rule(vertices[cell, :count])
        density = np.asarray(
            source_case.toroidal_current_density(points[:, 0], points[:, 1]),
            dtype=np.float64,
        )
        current[cell] = float(np.sum(weights * density))
    return current


def _class_census(
    classes: np.ndarray, booked: np.ndarray, analytic: np.ndarray
) -> dict[str, Any]:
    difference = booked - analytic
    absolute_total = float(np.sum(np.abs(difference)))
    result: dict[str, Any] = {}
    for name in ("whole", "cut", "inactive"):
        selected = classes == name
        absolute = float(np.sum(np.abs(difference[selected])))
        result[name] = {
            "cell_count": int(np.count_nonzero(selected)),
            "booked_current_a": float(np.sum(booked[selected])),
            "analytic_current_a": float(np.sum(analytic[selected])),
            "booked_minus_analytic_a": float(np.sum(difference[selected])),
            "absolute_difference_a": absolute,
            "absolute_difference_fraction": (
                absolute / absolute_total if absolute_total > 0.0 else 0.0
            ),
        }
    result["all"] = {
        "cell_count": len(classes),
        "booked_current_a": float(np.sum(booked)),
        "analytic_current_a": float(np.sum(analytic)),
        "booked_minus_analytic_a": float(np.sum(difference)),
        "absolute_difference_a": absolute_total,
    }
    return result


def _booking_census_arm(
    context: dict[str, Any], reference_terminal: np.ndarray
) -> dict[str, Any]:
    operator = context["operator"]
    terminal_partition = operator._support_partition(
        jnp.asarray(reference_terminal), REQUESTED_CLASS
    )
    analytic_partition = operator._support_partition(
        jnp.asarray(context["analytic"]), REQUESTED_CLASS
    )
    terminal_masks, _topology, _sample, terminal_support = terminal_partition
    analytic_masks, _analytic_topology, _analytic_sample, analytic_support = (
        analytic_partition
    )
    terminal_classes = _support_classes(terminal_support)
    analytic_classes = _support_classes(analytic_support)
    moments, amplitude = operator.normalised_current_moments(
        jnp.asarray(reference_terminal), context["target_current"], REQUESTED_CLASS
    )
    booked = np.asarray(moments.cell_current, dtype=np.float64)
    support_matched = _analytic_current_on_support(
        context["source_case"], terminal_support
    )
    fixture_analytic = np.asarray(
        context["analytic_moments"].cell_current, dtype=np.float64
    )
    centres = np.asarray(
        operator.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    )
    terminal_psi_norm = np.asarray(terminal_masks.psi_norm, dtype=np.float64)
    analytic_psi_norm = np.asarray(analytic_masks.psi_norm, dtype=np.float64)
    changed = np.flatnonzero(terminal_classes != analytic_classes)
    changed_rows = [
        {
            "cell": int(cell),
            "centre_rz_m": centres[cell],
            "analytic_class": str(analytic_classes[cell]),
            "terminal_class": str(terminal_classes[cell]),
            "analytic_psi_norm": float(analytic_psi_norm[cell]),
            "terminal_psi_norm": float(terminal_psi_norm[cell]),
            "booked_current_a": float(booked[cell]),
            "support_matched_analytic_current_a": float(support_matched[cell]),
            "fixture_analytic_current_a": float(fixture_analytic[cell]),
        }
        for cell in changed
    ]
    fixture_census = _class_census(terminal_classes, booked, fixture_analytic)
    boundary_fraction = (
        fixture_census["cut"]["absolute_difference_fraction"]
        + fixture_census["inactive"]["absolute_difference_fraction"]
    )
    return {
        "arm": "D",
        "mechanism": (
            "target-normalised production booked current against analytic current "
            "on the same terminal support and against analytic clipped fixture moments"
        ),
        "normalisation_amplitude": float(amplitude),
        "terminal_support_census": _class_census(
            terminal_classes, booked, support_matched
        ),
        "fixture_reference_census_by_terminal_class": fixture_census,
        "participation": {
            "analytic_active_cells": int(
                np.count_nonzero(analytic_classes != "inactive")
            ),
            "terminal_active_cells": int(
                np.count_nonzero(terminal_classes != "inactive")
            ),
            "changed_class_count": len(changed_rows),
            "changed_cells": changed_rows,
        },
        "boundary_class_absolute_difference_fraction": boundary_fraction,
        "supports_booking_error_in_cut_or_inactive_cells": bool(
            boundary_fraction >= 0.90
            and fixture_census["whole"]["absolute_difference_fraction"] <= 0.10
        ),
        "completed": True,
    }


def _draw_row_figure(
    path: Path,
    context: dict[str, Any],
    analytic_terminal: np.ndarray,
    reference_terminal: np.ndarray,
    analytic_arm: dict[str, Any],
    reference_arm: dict[str, Any],
) -> dict[str, Any]:
    coordinates = context["coordinates"]
    analytic = context["analytic"]
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    boundary = certificate._boundary(CASE_NAME, context["exact"])
    radial, height, analytic_field = certificate._raster_field(
        coordinates, analytic, wall
    )
    _, _, analytic_terminal_field = certificate._raster_field(
        coordinates, analytic_terminal, wall
    )
    _, _, reference_terminal_field = certificate._raster_field(
        coordinates, reference_terminal, wall
    )
    span = context["span"]
    shared_levels = poloidal.contour_levels(
        np.concatenate(
            (
                analytic_field.ravel(),
                analytic_terminal_field.ravel(),
                reference_terminal_field.ravel(),
            )
        ),
        count=12,
    )
    states = (
        (
            "analytic-start terminal",
            analytic_terminal_field,
            analytic_terminal,
            analytic_arm,
        ),
        (
            "current-aligned-seed terminal",
            reference_terminal_field,
            reference_terminal,
            reference_arm,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.5), constrained_layout=True)
    wall_units = (wall,)
    analytic_topology = context["analytic_topology"]
    for row_index, (label, field, state, arm) in enumerate(states):
        overlay_axis = axes[row_index, 0]
        difference_axis = axes[row_index, 1]
        poloidal.draw_flux_contours(
            overlay_axis, radial, height, analytic_field, shared_levels, color="#3366cc"
        )
        poloidal.draw_flux_contours(
            overlay_axis, radial, height, field, shared_levels, color="#cc7722"
        )
        difference = (field - analytic_field) / span
        poloidal.draw_flux_contours(
            difference_axis,
            radial,
            height,
            difference,
            FIXED_DIFFERENCE_LEVELS,
            color="#7a3e9d",
        )
        terminal_topology = oracle_probe._topology(context["operator"], state)
        for axis in (overlay_axis, difference_axis):
            poloidal.draw_wall(axis, units=wall_units)
            poloidal.draw_nulls(
                axis,
                magnetic_axis=analytic_topology["axis_rz_m"],
                x_points=analytic_topology["x_point_rz_m"],
                style=DEFAULT_INK.variant(
                    axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
                ),
                contain=wall_units,
            )
            poloidal.draw_nulls(
                axis,
                magnetic_axis=terminal_topology["axis_rz_m"],
                x_points=terminal_topology["x_point_rz_m"],
                style=DEFAULT_INK.variant(
                    axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
                ),
                contain=wall_units,
            )
            poloidal_axes(axis)
        poloidal.draw_boundary(
            overlay_axis, boundary[:, 0], boundary[:, 1], color="#3366cc"
        )
        terminal_metrics = arm["terminal"]
        overlay_axis.set_title(
            f"{label}: analytic blue / solved ochre\n"
            f"residual={terminal_metrics['terminal_residual']:.3e}; "
            f"converged={terminal_metrics['converged']}",
            fontsize=8,
        )
        difference_axis.set_title(
            f"({label} - analytic) / span\n"
            f"fixed levels {FIXED_DIFFERENCE_LEVELS.tolist()}",
            fontsize=7,
        )
    figure.suptitle(
        f"{CASE_NAME} · {abs(int(context['requested_cells']))} requested cells",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": _file_digest(path),
        "shared_flux_levels_wb": shared_levels,
        "difference_levels_fraction_of_span": FIXED_DIFFERENCE_LEVELS,
    }


def _row_verdict(
    context: dict[str, Any],
    analytic_arm: dict[str, Any],
    analytic_terminal: np.ndarray,
    reference_arm: dict[str, Any],
    reference_terminal: np.ndarray,
    translated_arm: dict[str, Any],
    booking_arm: dict[str, Any],
) -> dict[str, Any]:
    grid_count = context["grid_count"]
    span = context["span"]
    analytic = context["analytic"]
    reference_distance = float(
        np.max(np.abs(reference_terminal[:grid_count] - analytic[:grid_count])) / span
    )
    analytic_start_distance = float(
        np.max(np.abs(analytic_terminal[:grid_count] - analytic[:grid_count])) / span
    )
    state_separation = float(
        np.max(np.abs(analytic_terminal[:grid_count] - reference_terminal[:grid_count]))
        / span
    )
    distinguishable_reference = reference_distance > 1.0e-8
    analytic_start_stays = (
        analytic_start_distance <= max(1.0e-4, 0.10 * reference_distance)
        if distinguishable_reference
        else analytic_start_distance <= 1.0e-4
    )
    analytic_start_matches_reference = state_separation <= max(
        1.0e-4, 0.10 * reference_distance
    )
    readings = {
        "seed_selected_second_self_consistent_state": bool(
            distinguishable_reference and analytic_start_stays
        ),
        "displaced_equilibrium": bool(translated_arm["supports_displaced_equilibrium"]),
        "booking_error_in_cut_or_inactive_cells": bool(
            booking_arm["supports_booking_error_in_cut_or_inactive_cells"]
        ),
    }
    if readings["seed_selected_second_self_consistent_state"]:
        supported = "seed-selected second self-consistent state"
        seam = "cold-seed basin selection before the converged fixed point"
    elif (
        analytic_start_matches_reference
        and readings["booking_error_in_cut_or_inactive_cells"]
    ):
        supported = "booking error in cut or inactive cells"
        seam = "participation in _profile_support and the confinement test"
    elif readings["displaced_equilibrium"]:
        supported = "displaced equilibrium"
        seam = "production limiter contact read"
    elif readings["booking_error_in_cut_or_inactive_cells"]:
        supported = "booking error in cut or inactive cells"
        seam = "moment conversion and participation of clipped cells"
    else:
        supported = "unresolved by the three candidate readings"
        seam = "no single named seam"
    return {
        "reference_distance_from_analytic_sup_fraction_of_span": reference_distance,
        "analytic_start_distance_from_analytic_sup_fraction_of_span": (
            analytic_start_distance
        ),
        "analytic_start_distance_from_reference_sup_fraction_of_span": (
            state_separation
        ),
        "analytic_start_stays_near_oracle": analytic_start_stays,
        "analytic_start_matches_reference_low_state": analytic_start_matches_reference,
        "candidate_readings": readings,
        "supported_reading": supported,
        "code_seam": seam,
        "analytic_start_terminal_residual": analytic_arm["terminal"][
            "terminal_residual"
        ],
        "reference_terminal_residual": reference_arm["terminal"]["terminal_residual"],
    }


def _report(receipt: dict[str, Any]) -> str:
    lines = [
        "# Exact-clip low-state discriminator",
        "",
        receipt["headline"],
        "",
        "| requested cells | arm A residual | arm A axis flux [Wb] | "
        "arm B residual | arm B axis flux [Wb] | arm C minus B boundary [Wb] | "
        "arm D cut+inactive absolute fraction | supported reading |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in receipt["rows"]:
        arm_a = row["arms"]["A"]["terminal"]
        arm_b = row["arms"]["B"]["terminal"]
        arm_c = row["arms"]["C"]
        arm_d = row["arms"]["D"]
        lines.append(
            f"| {abs(int(row['requested_cells']))} | "
            f"{arm_a['terminal_residual']:.3e} | {arm_a['axis_flux_wb']:.8g} | "
            f"{arm_b['terminal_residual']:.3e} | {arm_b['axis_flux_wb']:.8g} | "
            f"{arm_c['translated_minus_arm_b_boundary_wb']:.5g} | "
            f"{arm_d['boundary_class_absolute_difference_fraction']:.5f} | "
            f"{row['verdict']['supported_reading']} |"
        )
    lines.extend(["", "## Trip snapshots", ""])
    for row in receipt["rows"]:
        lines.extend(
            [
                f"### {abs(int(row['requested_cells']))} requested cells",
                "",
                "| state | residual | axis flux [Wb] | boundary level [Wb] | "
                "max difference / span |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        snapshots = row["arms"]["A"]["snapshots"]
        for trip in ("1", "2", "4"):
            item = snapshots[trip]
            lines.append(
                f"| after {trip} trip(s) | {item['terminal_residual']:.3e} | "
                f"{item['axis_flux_wb']:.8g} | {item['boundary_level_wb']:.8g} | "
                f"{item['max_solved_minus_analytic_over_span']:.6g} |"
            )
        item = row["arms"]["A"]["terminal"]
        lines.append(
            f"| terminal ({item['trip_count']} trips) | "
            f"{item['terminal_residual']:.3e} | {item['axis_flux_wb']:.8g} | "
            f"{item['boundary_level_wb']:.8g} | "
            f"{item['max_solved_minus_analytic_over_span']:.6g} |"
        )
        lines.extend(
            [
                "",
                f"![Arm A and B terminals]({row['figure']['project_absolute_src']})",
                "",
                (
                    "Blue contours and solid triangles are analytic; ochre contours "
                    "and solid triangles are solved. Both difference panels use the "
                    "same fixed signed fractions of the analytic flux span. "
                    "Arm A residual "
                    f"{row['arms']['A']['terminal']['terminal_residual']:.3e}, "
                    f"converged {row['arms']['A']['terminal']['converged']}; "
                    "arm B residual "
                    f"{row['arms']['B']['terminal']['terminal_residual']:.3e}, "
                    f"converged {row['arms']['B']['terminal']['converged']}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            receipt["interpretation"],
            "",
            "The receipt retains every changed participation cell with its centre, "
            "analytic and terminal class, and both normalized-flux values. No source "
            "file was changed by this measurement.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    output_root: Path,
    reference_parts: Path,
    report_path: Path,
) -> dict[str, Any]:
    started = perf_counter()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the discriminator requires extended precision")
    set_support_clip_mode("exact")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    lane = _lane()
    receipt_path = output_root / "receipt.json"
    receipt: dict[str, Any] = {
        "$id": "nova.exact-clip-low-state-discriminator",
        "revision": _revision(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _file_digest(Path(__file__)),
        },
        "clip_mode": "exact",
        "lane": lane,
        "persistent_compilation_cache": cache.receipt(),
        "reference_parts": str(reference_parts),
        "rows": [],
        "completed": False,
    }
    _write_json(receipt_path, receipt)
    for requested_cells in REQUESTED_ROWS:
        context = _build_context(requested_cells)
        context["requested_cells"] = requested_cells
        reference, reference_path = _load_reference_part(
            reference_parts, requested_cells
        )
        reference_coordinates = np.asarray(
            reference["render_data"]["coordinates_rz_m"], dtype=np.float64
        )
        if not np.array_equal(context["coordinates"], reference_coordinates):
            raise RuntimeError("rebuilt coordinates differ from the reference part")
        row_slug = f"weak-static-{abs(requested_cells)}"
        row_directory = output_root / "parts"
        analytic_part = row_directory / f"{row_slug}-analytic-start.json"
        analytic_arm, analytic_terminal = _analytic_start_arm(context, analytic_part)
        reference_arm, reference_terminal = _reference_seed_arm(
            context, reference, reference_path
        )
        _write_json(row_directory / f"{row_slug}-seed-reference.json", reference_arm)
        translated_arm = _translated_analytic_arm(context, reference_terminal)
        _write_json(
            row_directory / f"{row_slug}-translated-analytic.json",
            translated_arm,
        )
        booking_arm = _booking_census_arm(context, reference_terminal)
        _write_json(row_directory / f"{row_slug}-booking-census.json", booking_arm)
        figure = _draw_row_figure(
            output_root / "panels" / f"{row_slug}-terminals.png",
            context,
            analytic_terminal,
            reference_terminal,
            analytic_arm,
            reference_arm,
        )
        verdict = _row_verdict(
            context,
            analytic_arm,
            analytic_terminal,
            reference_arm,
            reference_terminal,
            translated_arm,
            booking_arm,
        )
        row = {
            "case": CASE_NAME,
            "requested_cells": requested_cells,
            "realised_cells": context["grid_count"],
            "analytic_flux_span_wb": context["span"],
            "analytic_topology": context["analytic_topology"],
            "fixture_exterior_cache": context["exterior_cache"],
            "arms": {
                "A": analytic_arm,
                "B": reference_arm,
                "C": translated_arm,
                "D": booking_arm,
            },
            "figure": figure,
            "verdict": verdict,
        }
        receipt["rows"].append(row)
        _write_json(receipt_path, receipt)
        print(
            f"LOW_STATE_ROW cells={requested_cells} "
            f"reading={verdict['supported_reading']}",
            flush=True,
        )
    readings = [row["verdict"]["supported_reading"] for row in receipt["rows"]]
    unique_readings = list(dict.fromkeys(readings))
    receipt["headline"] = (
        "The two weak rows jointly support " + ", then ".join(unique_readings) + "."
    )
    receipt["interpretation"] = (
        "Arm A decides whether the analytic state is retained by the production map; "
        "arm C prices a rigid displacement through the same limiter read; arm D "
        "locates current disagreement by support class. The per-row verdicts are "
        + "; ".join(
            f"{abs(int(row['requested_cells']))} cells: "
            f"{row['verdict']['supported_reading']} "
            f"({row['verdict']['code_seam']})"
            for row in receipt["rows"]
        )
        + "."
    )
    receipt["completed"] = True
    receipt["elapsed_seconds"] = perf_counter() - started
    _write_json(receipt_path, receipt)
    report = _report(receipt)
    _write_text(output_root / "report.md", report)
    _write_text(report_path, report)
    print(f"LOW_STATE_EXIT rows={len(receipt['rows'])} completed=True", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-parts", type=Path, default=DEFAULT_REFERENCE_PARTS)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args()
    receipt = run(
        arguments.output_root.resolve(),
        arguments.reference_parts.resolve(),
        arguments.report_path.resolve(),
    )
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
