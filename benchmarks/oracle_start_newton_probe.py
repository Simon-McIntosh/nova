"""Probe the production fixed-point iteration from an analytic near-root state.

The analytic flux, exterior completion, current target, mesh interaction
matrices, clip modes, and Newton controls all come from the production
Solov'ev certificate stack.  The benchmark changes no production behavior.
It persists each row and allocation mode before advancing, so a scheduler
expiry preserves every completed measurement.
"""

from __future__ import annotations

import argparse
import hashlib
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
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import fixed_point
from nova.equilibrium.forward_operator import (
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.jax.config import (
    Precision,
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/cut-cell-current-attribution/oracle-start"
    / "oracle-start-newton-probe.json"
)
DEFAULT_REPORT_DIRECTORY = DEFAULT_OUTPUT.parent
PART_DIRECTORY_NAME = "parts"
PERTURBATION_FRACTIONS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
FINITE_DIFFERENCE_STEPS = (1.0e-5, 1.0e-7)
RANDOM_DIRECTION_COUNT = 4
RANDOM_SEED = 271828
MODES = ("exact", "chord")
ROWS = (
    ("weak-rotation-reactor-static", -110),
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -110),
    ("moderate-rotation-conventional-static", -300),
    (certificate.DIVERTED_CASE_NAME, -300),
    (certificate.DIVERTED_CASE_NAME, -500),
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("the measurement requires one scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if reservation != "gpu_0003_grpA":
        raise RuntimeError(f"unexpected reservation {reservation!r}")
    if platforms != "cuda,cpu":
        raise RuntimeError(f"expected JAX_PLATFORMS=cuda,cpu, received {platforms!r}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "H200" not in gpu:
        raise RuntimeError(f"the measurement requires an H200, received {gpu!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "1")),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "gpu": gpu,
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "jax_cuda_devices": [str(device) for device in jax.devices("gpu")],
        "jax_cpu_devices": [str(device) for device in jax.devices("cpu")],
    }


def _row_slug(case_name: str, requested_cells: int) -> str:
    return f"{case_name}-cells-{abs(requested_cells)}"


def _part_path(output: Path, case_name: str, requested_cells: int, mode: str) -> Path:
    name = f"{_row_slug(case_name, requested_cells)}-{mode}.json"
    return output.parent / PART_DIRECTORY_NAME / name


def _smooth_directions(
    coordinates: np.ndarray, count: int, seed: int
) -> list[np.ndarray]:
    points = np.asarray(coordinates, dtype=np.float64)
    centre = np.mean(points, axis=0)
    scale = np.maximum(np.ptp(points, axis=0), 1.0e-12)
    normalized = (points - centre) / scale
    radius = normalized[:, 0]
    height = normalized[:, 1]
    basis = np.column_stack(
        (
            np.ones(len(points)),
            radius,
            height,
            radius * height,
            radius**2 - np.mean(radius**2),
            height**2 - np.mean(height**2),
            np.sin(np.pi * radius),
            np.cos(np.pi * height),
            np.sin(np.pi * (radius + height)),
            np.exp(-5.0 * (radius**2 + height**2)),
        )
    )
    generator = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    for _ in range(count):
        direction = basis @ generator.normal(size=basis.shape[1])
        direction -= np.mean(direction)
        norm = float(np.max(np.abs(direction)))
        if not np.isfinite(norm) or norm == 0.0:
            raise RuntimeError("the smooth-direction instrument produced no signal")
        directions.append(direction / norm)
    return directions


def _norms(delta: np.ndarray, span: float, grid_count: int) -> dict[str, float]:
    grid = np.asarray(delta, dtype=np.float64)[:grid_count]
    return {
        "absolute_rms_wb": float(np.sqrt(np.mean(grid**2))),
        "absolute_sup_wb": float(np.max(np.abs(grid))),
        "relative_rms_of_span": float(np.sqrt(np.mean(grid**2)) / span),
        "relative_sup_of_span": float(np.max(np.abs(grid)) / span),
    }


def _topology(operator: Any, state: np.ndarray) -> dict[str, Any]:
    try:
        _masks, topology = operator.read(jnp.asarray(state))
    except NoQualifiedAxisError as error:
        return {
            "read_status": "no_qualified_axis",
            "axis_rz_m": None,
            "x_point_rz_m": None,
            "boundary_flux_wb": None,
            "axis_flux_wb": None,
            "exception_text": str(error),
        }
    axis = np.asarray(topology.axis, dtype=np.float64)
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    return {
        "read_status": "qualified_axis",
        "axis_rz_m": axis.tolist() if np.all(np.isfinite(axis)) else None,
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        "boundary_flux_wb": float(topology.boundary_flux),
        "axis_flux_wb": float(topology.axis_flux),
        "exception_text": None,
    }


def _booked_current(
    operator: Any,
    state: np.ndarray,
    requested_class: int,
    target: float,
) -> dict[str, Any]:
    try:
        moments = operator.cell_current_moments(jnp.asarray(state), requested_class)
        booked = float(jnp.sum(moments.cell_current))
        amplitude = float(operator.current_normalisation_amplitude(target, booked))
    except Exception as error:
        return {
            "booked_plasma_current_a": None,
            "analytic_plasma_current_a": target,
            "booked_over_analytic": None,
            "normalisation_amplitude": None,
            "status": "unavailable",
            "exception_text": f"{type(error).__name__}: {error}",
        }
    return {
        "booked_plasma_current_a": booked,
        "analytic_plasma_current_a": target,
        "booked_over_analytic": booked / target,
        "normalisation_amplitude": amplitude,
        "status": "finite" if np.isfinite(amplitude) else "nonfinite",
        "exception_text": None,
    }


def _solver_functions(operator: Any, requested_class: int, target_current: float):
    mapped = operator.flux_map(
        requested_class=requested_class, target_current=target_current
    )
    shadowed = operator.flux_map_with_shadow(
        requested_class=requested_class, target_current=target_current
    )

    def shadow_mask(state):
        return operator.residual_shadow_mask(state, requested_class)

    def promoted_shadow_mask(state, previous):
        return operator.residual_shadow_mask(
            state, requested_class, previous_shadow=previous
        )

    return mapped, shadowed, shadow_mask, promoted_shadow_mask


def _solve_prefix(
    mapped: Any,
    shadowed: Any,
    shadow_mask: Any,
    promoted_shadow_mask: Any,
    initial: np.ndarray,
    newton_steps: int,
):
    history = fixed_point.newton_krylov(
        mapped,
        jnp.asarray(initial),
        newton_steps=newton_steps,
        gmres_iterations=recovery.KRYLOV_ITERATIONS,
        warmup=0,
        shadow_mask_fn=shadow_mask,
        promoted_shadow_mask_fn=promoted_shadow_mask,
        shadowed_map_fn=shadowed,
        precision=Precision.DOUBLE,
    )
    jax.block_until_ready(history.state)
    return history


def _state_record(
    operator: Any,
    state: np.ndarray,
    analytic: np.ndarray,
    span: float,
    grid_count: int,
    requested_class: int,
    target_current: float,
    axis_reference: np.ndarray,
) -> dict[str, Any]:
    topology = _topology(operator, state)
    axis = topology["axis_rz_m"]
    return {
        "distance_to_analytic": _norms(state - analytic, span, grid_count),
        "axis_position_error_m": (
            None
            if axis is None
            else float(np.linalg.norm(np.asarray(axis) - axis_reference))
        ),
        "topology": topology,
        "current": _booked_current(operator, state, requested_class, target_current),
    }


def _iteration(
    operator: Any,
    analytic: np.ndarray,
    direction: np.ndarray,
    fraction: float,
    span: float,
    grid_count: int,
    requested_class: int,
    target_current: float,
    axis_reference: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    initial = analytic + fraction * span * direction
    mapped, shadowed, shadow_mask, promoted_shadow_mask = _solver_functions(
        operator, requested_class, target_current
    )
    initial_mapped = np.asarray(jax.block_until_ready(mapped(jnp.asarray(initial))))
    initial_relative_residual = float(
        fixed_point._relative_residual(
            jnp.asarray(initial_mapped), jnp.asarray(initial)
        )
    )
    prefixes = [
        {
            "newton_step": 0,
            "relative_fixed_point_residual": initial_relative_residual,
            "active_set_trip_count": 0,
            "accepted_newton_promotions": 0,
            **_state_record(
                operator,
                initial,
                analytic,
                span,
                grid_count,
                requested_class,
                target_current,
                axis_reference,
            ),
        }
    ]
    terminal_state = initial
    terminal_history = None
    for step in range(1, recovery.NEWTON_STEPS + 1):
        history = _solve_prefix(
            mapped,
            shadowed,
            shadow_mask,
            promoted_shadow_mask,
            initial,
            step,
        )
        terminal_history = history
        terminal_state = np.asarray(history.state, dtype=np.float64)
        prefixes.append(
            {
                "newton_step": step,
                "relative_fixed_point_residual": float(history.residual),
                "active_set_trip_count": int(history.active_set_iterations),
                "accepted_newton_promotions": int(history.accepted_newton_promotions),
                **_state_record(
                    operator,
                    terminal_state,
                    analytic,
                    span,
                    grid_count,
                    requested_class,
                    target_current,
                    axis_reference,
                ),
            }
        )
    assert terminal_history is not None
    initial_distance = prefixes[0]["distance_to_analytic"]["relative_sup_of_span"]
    terminal_distance = prefixes[-1]["distance_to_analytic"]["relative_sup_of_span"]
    return (
        {
            "requested_relative_perturbation": fraction,
            "realised_relative_sup_perturbation": initial_distance,
            "smooth_direction_sha256": hashlib.sha256(
                np.ascontiguousarray(direction, dtype="<f8").tobytes()
            ).hexdigest(),
            "steps": prefixes,
            "terminal": {
                "contracted_toward_analytic": bool(
                    terminal_distance < initial_distance
                ),
                "left_analytic_neighbourhood": bool(
                    terminal_distance >= initial_distance
                ),
                "trip_count": int(terminal_history.active_set_iterations),
                "attempted_newton_promotions": int(
                    terminal_history.attempted_newton_promotions
                ),
                "accepted_newton_promotions": int(
                    terminal_history.accepted_newton_promotions
                ),
                "converged": bool(terminal_history.converged),
                "termination": fixed_point.FixedPointTerminationReason(
                    int(terminal_history.termination_reason)
                ).name.lower(),
                "relative_fixed_point_residual": float(terminal_history.residual),
                "relative_sup_distance_to_analytic": terminal_distance,
            },
        },
        terminal_state,
    )


def _relative_discrepancy(reference: np.ndarray, candidate: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference))
    denominator = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def _jacobian_probe(
    operator: Any,
    analytic: np.ndarray,
    span: float,
    requested_class: int,
    target_current: float,
    random_directions: list[np.ndarray],
) -> dict[str, Any]:
    mapped, shadowed, shadow_mask, _promoted_shadow_mask = _solver_functions(
        operator, requested_class, target_current
    )
    state = jnp.asarray(analytic)
    frozen_shadow = shadow_mask(state)

    def fixed_residual(candidate):
        return candidate - shadowed(candidate, frozen_shadow)

    residual, tangent = jax.linearize(fixed_residual, state)
    mapped_state = mapped(state)
    nonlinear_residual = fixed_point._relative_residual(mapped_state, state)

    def linear_action(vector):
        return tangent(vector)

    qualified = fixed_point._qualified_krylov_step(
        linear_action,
        mapped_state - state,
        nonlinear_residual,
        gmres_iterations=recovery.KRYLOV_ITERATIONS,
        condition_ratio_limit=math.e,
        preceding_condition_baseline=jnp.asarray(jnp.nan, dtype=state.dtype),
    )
    jax.block_until_ready(qualified.step)
    newton_direction = np.asarray(qualified.step, dtype=np.float64)
    directions = [
        ("newton", newton_direction),
        *[
            (f"random_{index + 1}", direction)
            for index, direction in enumerate(random_directions)
        ],
    ]
    records = []
    for name, unscaled in directions:
        norm = float(np.max(np.abs(unscaled)))
        if not np.isfinite(norm) or norm == 0.0:
            records.append(
                {
                    "direction": name,
                    "status": "zero_or_nonfinite_direction",
                    "relative_steps": [],
                }
            )
            continue
        direction = np.asarray(unscaled / norm, dtype=np.float64)
        exact_jvp = np.asarray(tangent(jnp.asarray(direction)), dtype=np.float64)
        step_records = []
        for relative_step in FINITE_DIFFERENCE_STEPS:
            absolute_step = relative_step * span
            plus = np.asarray(
                fixed_residual(state + absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            minus = np.asarray(
                fixed_residual(state - absolute_step * jnp.asarray(direction)),
                dtype=np.float64,
            )
            finite_difference = (plus - minus) / (2.0 * absolute_step)
            step_records.append(
                {
                    "relative_step_of_flux_span": relative_step,
                    "absolute_step_wb": absolute_step,
                    "relative_jvp_discrepancy": _relative_discrepancy(
                        exact_jvp, finite_difference
                    ),
                    "exact_jvp_rms": float(np.sqrt(np.mean(exact_jvp**2))),
                    "finite_difference_rms": float(
                        np.sqrt(np.mean(finite_difference**2))
                    ),
                }
            )
        records.append(
            {
                "direction": name,
                "status": "measured",
                "unit_sup_direction_sha256": hashlib.sha256(
                    np.ascontiguousarray(direction, dtype="<f8").tobytes()
                ).hexdigest(),
                "relative_steps": step_records,
            }
        )
    return {
        "residual_definition": (
            "state minus the production shadow-frozen fixed-point map; this is "
            "the I-minus-J linear action used by fixed_point.newton_krylov"
        ),
        "analytic_residual_rms": float(
            np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2))
        ),
        "newton_direction_qualification": fixed_point.KrylovActionQualification(
            int(qualified.qualification)
        ).name.lower(),
        "projected_krylov_condition": float(qualified.projected_condition),
        "directions": records,
    }


def _analytic_topology(case_name: str, exact: Any) -> dict[str, Any]:
    if certificate._is_diverted_case(case_name):
        return certificate._analytic_diverted_topology(exact)
    return {
        "axis_rz_m": np.asarray(exact.magnetic_axis, dtype=np.float64).tolist(),
        "x_point_rz_m": None,
    }


def _render_row(
    path: Path,
    case_name: str,
    requested_cells: int,
    coordinates: np.ndarray,
    analytic: np.ndarray,
    wall: np.ndarray,
    boundary: np.ndarray,
    analytic_topology: dict[str, Any],
    terminals: dict[str, tuple[np.ndarray, dict[str, Any]]],
) -> None:
    figure, axes = plt.subplots(
        1, len(MODES), figsize=(10.8, 5.1), constrained_layout=True
    )
    analytic_radial, analytic_height, analytic_raster = certificate._raster_field(
        coordinates, analytic, wall
    )
    terminal_rasters = {
        mode: certificate._raster_field(coordinates, terminals[mode][0], wall)
        for mode in MODES
    }
    all_values = [analytic_raster.ravel()]
    all_values.extend(raster[2].ravel() for raster in terminal_rasters.values())
    levels = poloidal.contour_levels(np.concatenate(all_values), count=12)
    wall_units = (wall,)
    for axis, mode in zip(np.atleast_1d(axes), MODES, strict=True):
        terminal, summary = terminals[mode]
        radial, height, terminal_raster = terminal_rasters[mode]
        poloidal.draw_flux_contours(
            axis,
            analytic_radial,
            analytic_height,
            analytic_raster,
            levels,
            color="#3366cc",
        )
        poloidal.draw_flux_contours(
            axis, radial, height, terminal_raster, levels, color="#cc7722"
        )
        poloidal.draw_boundary(axis, boundary[:, 0], boundary[:, 1], color="#3366cc")
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
        terminal_topology = summary["steps"][-1]["topology"]
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
        terminal_record = summary["terminal"]
        axis.set_title(
            f"{mode}: analytic blue / terminal ochre\n"
            f"residual={terminal_record['relative_fixed_point_residual']:.3e}; "
            f"converged={terminal_record['converged']}; "
            f"trips={terminal_record['trip_count']}",
            fontsize=8,
        )
    figure.suptitle(
        f"{case_name} · {abs(requested_cells)} cells · 1e-2 analytic perturbation"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _measure_mode(
    case_name: str,
    requested_cells: int,
    mode: str,
    operator: Any,
    analytic: np.ndarray,
    coordinates: np.ndarray,
    grid_count: int,
    span: float,
    requested_class: int,
    target_current: float,
    axis_reference: np.ndarray,
    output: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    started = perf_counter()
    set_support_clip_mode(mode)
    if support_clip_mode() != mode:
        raise RuntimeError(f"clip-mode setter did not select {mode}")
    booked = _booked_current(operator, analytic, requested_class, target_current)
    mapped, _shadowed, _shadow_mask, _promoted_shadow_mask = _solver_functions(
        operator, requested_class, target_current
    )
    mapped_analytic = np.asarray(
        jax.block_until_ready(mapped(jnp.asarray(analytic))), dtype=np.float64
    )
    map_floor = _norms(mapped_analytic - analytic, span, grid_count)
    directions = _smooth_directions(
        coordinates, RANDOM_DIRECTION_COUNT + 1, RANDOM_SEED
    )
    iterations: list[dict[str, Any]] = []
    terminal_for_figure: np.ndarray | None = None
    figure_summary: dict[str, Any] | None = None
    row = {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": grid_count,
        "mode": mode,
        "mode_semantics": (
            "signed-flux spline-chain exact clip"
            if mode == "exact"
            else "production whole-cell booking control"
        ),
        "analytic_map_floor": {
            **map_floor,
            "booked_current": booked,
        },
        "perturbations": iterations,
        "jacobian": None,
        "wall_seconds": None,
        "completed": False,
    }
    part = _part_path(output, case_name, requested_cells, mode)
    _write_json(part, row)
    perturbation_direction = directions[0]
    for fraction in PERTURBATION_FRACTIONS:
        result, terminal = _iteration(
            operator,
            analytic,
            perturbation_direction,
            fraction,
            span,
            grid_count,
            requested_class,
            target_current,
            axis_reference,
        )
        iterations.append(result)
        if fraction == 1.0e-2:
            terminal_for_figure = terminal
            figure_summary = result
        _write_json(part, row)
        print(
            f"ORACLE_START_ROW case={case_name} cells={abs(requested_cells)} "
            f"mode={mode} perturbation={fraction:.0e} "
            f"contracted={result['terminal']['contracted_toward_analytic']} "
            f"distance={result['terminal']['relative_sup_distance_to_analytic']:.6e}",
            flush=True,
        )
    row["jacobian"] = _jacobian_probe(
        operator,
        analytic,
        span,
        requested_class,
        target_current,
        directions[1:],
    )
    row["wall_seconds"] = perf_counter() - started
    row["completed"] = True
    _write_json(part, row)
    if terminal_for_figure is None or figure_summary is None:
        raise RuntimeError("the 1e-2 terminal state was not retained")
    return row, terminal_for_figure, figure_summary


def _stability(mode_row: dict[str, Any]) -> dict[str, Any]:
    outcomes = {
        f"{item['requested_relative_perturbation']:.0e}": item["terminal"][
            "contracted_toward_analytic"
        ]
        for item in mode_row["perturbations"]
    }
    return {
        "analytic_flux_is_stable_fixed_point": all(outcomes.values()),
        "contraction_by_perturbation": outcomes,
    }


def _diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [row["modes"]["exact"] for row in rows]
    maximum_floor = max(
        row["analytic_map_floor"]["relative_sup_of_span"] for row in exact
    )
    discrepancies = [
        step["relative_jvp_discrepancy"]
        for row in exact
        for direction in row["jacobian"]["directions"]
        if direction["status"] == "measured"
        for step in direction["relative_steps"]
        if step["relative_step_of_flux_span"] == 1.0e-5
    ]
    maximum_jacobian_discrepancy = max(discrepancies, default=float("nan"))
    any_departure = any(
        not item["terminal"]["contracted_toward_analytic"]
        for row in exact
        for item in row["perturbations"]
    )
    if maximum_floor > 1.0e-8:
        classification = "allocation_floor"
        sentence = (
            "The exact-clip production map does not admit the analytic state at "
            "the measured floor, so the defect enters before the Newton Jacobian."
        )
    elif maximum_jacobian_discrepancy > 1.0e-3:
        classification = "jacobian"
        sentence = (
            "The analytic state is a map fixed point, but its exact tangent does "
            "not agree with the finite-difference residual action."
        )
    elif any_departure:
        classification = "globalisation"
        sentence = (
            "The analytic state is a map fixed point and the Jacobian is consistent, "
            "but at least one production trajectory leaves its analytic neighbourhood."
        )
    else:
        classification = "stable_fixed_point_no_defect_reproduced"
        sentence = (
            "The analytic state is a stable fixed point of the measured exact-clip "
            "iteration; no allocation, Jacobian, or globalisation defect reproduced."
        )
    return {
        "classification": classification,
        "maximum_exact_relative_map_floor_sup": maximum_floor,
        "maximum_exact_relative_jacobian_discrepancy_at_1e-5": (
            maximum_jacobian_discrepancy
        ),
        "any_exact_trajectory_left_analytic_neighbourhood": any_departure,
        "sentence": sentence,
    }


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# Analytic-start production Newton probe",
        "",
        receipt["diagnosis"]["sentence"],
        "",
        "Each map-floor pair is relative RMS / relative sup of the analytic flux span. "
        "Contraction columns list 1e-4, 1e-3, 1e-2, and 1e-1 starts in that order. "
        "The Jacobian value is the worst central-difference discrepancy at "
        "relative step 1e-5.",
        "",
        "| Row | Mode | Map floor rms / sup | Booked / analytic current | "
        "Contraction | Trips | Worst JVP discrepancy | Stable fixed point |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for row in receipt["rows"]:
        for mode in MODES:
            measured = row["modes"][mode]
            floor = measured["analytic_map_floor"]
            current = floor["booked_current"]["booked_over_analytic"]
            perturbations = measured["perturbations"]
            contractions = ", ".join(
                "yes" if item["terminal"]["contracted_toward_analytic"] else "no"
                for item in perturbations
            )
            trips = ", ".join(
                str(item["terminal"]["trip_count"]) for item in perturbations
            )
            discrepancies = [
                step["relative_jvp_discrepancy"]
                for direction in measured["jacobian"]["directions"]
                if direction["status"] == "measured"
                for step in direction["relative_steps"]
                if step["relative_step_of_flux_span"] == 1.0e-5
            ]
            stability = row["stability"][mode]["analytic_flux_is_stable_fixed_point"]
            lines.append(
                f"| {row['case']} {abs(row['requested_cells'])} | {mode} | "
                f"{floor['relative_rms_of_span']:.3e} / "
                f"{floor['relative_sup_of_span']:.3e} | {current:.8f} | "
                f"{contractions} | {trips} | {max(discrepancies):.3e} | "
                f"{'yes' if stability else 'no'} |"
            )
    lines.extend(
        (
            "",
            "The machine interaction arrays were constructed once per row and reused "
            "for both allocation modes and every perturbation. Each prefix state was "
            "recomputed from the same initial state with the corresponding production "
            "Newton-step budget, preserving the certificate solver rather than "
            "replacing it with an instrumented iteration.",
            "",
            f"Figure directory: `{DEFAULT_OUTPUT.parent}`.",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(output: Path, report_directory: Path) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    started = perf_counter()
    lane = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    original_mode = support_clip_mode()
    rows: list[dict[str, Any]] = []
    try:
        for case_name, requested_cells in ROWS:
            carrier_case, source_case, exact = certificate._case(case_name)
            machine = certificate._case_machine(
                case_name, carrier_case, exact, requested_cells
            )
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            analytic = certificate._exact_state(case_name, exact, coordinates)
            empty_operator = oracle_fixture.forward_operator(source_case, machine)
            exact_physical = oracle_fixture.exact_current_moments(
                source_case, empty_operator, analytic
            )
            exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
            exact_internal = oracle_fixture._internal_flux_image(
                empty_operator, exact_coefficients
            )
            operator = oracle_fixture.forward_operator(
                source_case, machine, analytic - exact_internal
            )
            target_current, _centroid, target_receipt = (
                certificate._closed_form_current_target(
                    case_name, source_case, operator, exact_physical
                )
            )
            requested_class = int(
                TopologyClass.DIVERTED
                if certificate._is_diverted_case(case_name)
                else TopologyClass.LIMITED
            )
            grid_count = len(machine.node)
            analytic_topology = _analytic_topology(case_name, exact)
            axis_reference = np.asarray(
                analytic_topology["axis_rz_m"], dtype=np.float64
            )
            analytic_read = _topology(operator, analytic)
            if analytic_read["axis_flux_wb"] is None:
                raise RuntimeError(
                    f"the analytic topology instrument could not read {case_name}"
                )
            span = abs(
                float(analytic_read["axis_flux_wb"])
                - float(analytic_read["boundary_flux_wb"])
            )
            if not np.isfinite(span) or span <= 0.0:
                raise RuntimeError(f"the analytic flux span is invalid for {case_name}")
            modes: dict[str, Any] = {}
            terminals: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            for mode in MODES:
                measured, terminal, figure_summary = _measure_mode(
                    case_name,
                    requested_cells,
                    mode,
                    operator,
                    analytic,
                    coordinates,
                    grid_count,
                    span,
                    requested_class,
                    target_current,
                    axis_reference,
                    output,
                )
                modes[mode] = measured
                terminals[mode] = (terminal, figure_summary)
            figure = output.parent / f"{_row_slug(case_name, requested_cells)}.png"
            _render_row(
                figure,
                case_name,
                requested_cells,
                coordinates,
                analytic,
                np.asarray(machine.wall_node, dtype=np.float64),
                certificate._boundary(case_name, exact),
                analytic_topology,
                terminals,
            )
            row = {
                "case": case_name,
                "requested_cells": requested_cells,
                "realised_cells": grid_count,
                "state_dimension": len(analytic),
                "characteristic_pitch_m": float(
                    np.sqrt(np.median(np.asarray(machine.area)))
                ),
                "analytic_flux_span_wb": span,
                "analytic_current_target_a": target_current,
                "analytic_current_target_receipt": target_receipt,
                "interaction_matrix_cache": machine.cache,
                "interaction_matrix_construction_count": 1,
                "modes": modes,
                "stability": {mode: _stability(modes[mode]) for mode in MODES},
                "figure": {
                    "filesystem_path": str(figure.relative_to(ROOT)),
                    "project_absolute_src": (
                        f"/nova/{figure.relative_to(ROOT / 'docs')}"
                    ),
                    "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
                    "caption": (
                        "Exact and whole-cell control terminal states from the 1e-2 "
                        "analytic perturbation; analytic contours and nulls blue, "
                        "terminal contours and nulls ochre, shared Wb levels, "
                        "wall shown."
                    ),
                },
            }
            rows.append(row)
            row_part = (
                output.parent
                / PART_DIRECTORY_NAME
                / f"{_row_slug(case_name, requested_cells)}.json"
            )
            _write_json(
                row_part,
                row,
            )
    finally:
        set_support_clip_mode(original_mode)
    receipt = {
        "schema": "nova.oracle-start-newton-probe",
        "version": 1,
        "source_revision": _source_revision(),
        "production_code_modified": False,
        "lane": {
            **lane,
            "persistent_compilation_cache": cache.receipt(),
            "wall_seconds": perf_counter() - started,
            "exit_marker": "ORACLE_START_NEWTON_PROBE_EXIT=0",
        },
        "design": {
            "rows": [
                {"case": case_name, "requested_cells": requested_cells}
                for case_name, requested_cells in ROWS
            ],
            "modes": {
                "exact": "signed-flux spline-chain exact clip",
                "chord": "production whole-cell booking control",
            },
            "perturbation_relative_sup_fractions": PERTURBATION_FRACTIONS,
            "smooth_random_seed": RANDOM_SEED,
            "newton_steps": recovery.NEWTON_STEPS,
            "gmres_iterations": recovery.KRYLOV_ITERATIONS,
            "warmup": 0,
            "finite_difference_relative_steps": FINITE_DIFFERENCE_STEPS,
            "random_jacobian_directions": RANDOM_DIRECTION_COUNT,
            "interaction_matrix_policy": (
                "one cached OracleMachine and one ForwardFluxOperator per row, reused "
                "across allocation modes and perturbations"
            ),
        },
        "rows": rows,
    }
    receipt["diagnosis"] = _diagnosis(rows)
    _write_json(output, receipt)
    _write_report(report_directory / "report.md", receipt)
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    receipt = run(arguments.output, arguments.report_directory)
    print(json.dumps(_strict(receipt["diagnosis"]), sort_keys=True), flush=True)
    print("ORACLE_START_NEWTON_PROBE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
