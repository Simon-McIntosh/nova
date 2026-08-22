"""Measure constrained cold starts and same-shot warm recovery on DIII-D.

The five validation frames and their score-independent exclusions are inherited
from the banked circuit-driven validation.  Every solve uses the public forward
profile seam with a declared total plasma current and the fixed-wiring circuit
currents.  A stalled target may consume only the first convergence-qualified
frame on a declared same-shot time-offset ladder; label metrics never choose
the neighbour or the reported terminal state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from benchmarks import diiid_current_pinned_forward as current_pinned
from benchmarks.diiid_circuit_driven_forward_validation import (
    FRAME_COUNT,
    LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS,
    SelectedFrame,
    _current_receipt,
    _score_state,
    _sha256,
    _strict_float,
    calibration_population,
    polarity_population,
    select_frames,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA,
    REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    _CURRENT_COLUMNS,
    _GEOMETRY_COLUMNS,
    _LABEL_COLUMNS,
    _read,
    build_profile,
)
from nova.imas.diiid_current import (
    DiiidCurrentAdapter,
    complete_profile_current_adapter,
    resolve_diiid_currents,
    shipped_current_at,
)
from nova.imas.diiid_description import (
    PF_ACTIVE_CIRCUIT,
    POLOIDAL_CONDUCTORS,
    dataset_machine_description,
    geometry_digest,
)
from nova.equilibrium.source import CurrentNormalisationError
from nova.jax.config import configure_dtypes


DEFAULT_OUTPUT = Path("docs/figures/current-constrained-forward-solve/cold-start")
RECEIPT_NAME = "constrained_cold_start_receipt.json"
FIGURE_NAME = "constrained_cold_start_summary.png"
RELATIVE_RESIDUAL_TOLERANCE = 1.0e-6
CURRENT_RELATIVE_ERROR_TOLERANCE = 1.0e-10
HOST_OUTER_ITERATIONS = 100
HOST_INNER_ITERATIONS = 40
NEIGHBOUR_FRAME_OFFSETS = (-1, 1, -2, 2, -4, 4, -8, 8, -16, 16, -32, 32)
ROUTE_NAMES = ("cold_start", "warm_neighbour")
_RESPONSE_CACHE: dict[tuple[str, bytes, bytes], tuple[Any, Any, dict[str, Any]]] = {}


@dataclass(frozen=True)
class PreparedFrame:
    """One frame with label sources and fixed-wiring inference inputs."""

    selected: SelectedFrame
    row: dict[str, Any]
    profile: Any
    seed: np.ndarray
    label: np.ndarray
    wall: np.ndarray
    reliable_surfaces: int
    wall_statement: str
    current: np.ndarray
    current_receipt: dict[str, Any]


@dataclass(frozen=True)
class SolveOutcome:
    """Public-seam terminal state and the fields needed for scoring."""

    state: np.ndarray
    residual: float
    iterations: int
    termination: str
    amplitude: float
    achieved_current_a: float
    residual_trajectory: tuple[float, ...]


def _columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_LABEL_COLUMNS,
                *_GEOMETRY_COLUMNS,
                *_CURRENT_COLUMNS,
                *current_pinned.PLASMA_CURRENT_COLUMNS,
            )
        )
    )


def prepare_frame(path: Path, frame: int) -> PreparedFrame:
    """Build one constrained fixed-wiring frame without consulting a score."""

    row = _read(path, _columns())
    row["_source_path"] = str(path)
    time_ms = float(row["efit_times"][frame])
    target_current_a = current_pinned._target_current(row, time_ms)
    selected = SelectedFrame(path, frame, time_ms, target_current_a)
    profile, seed, label, wall, reliable, wall_statement = build_profile(
        row,
        frame,
        REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    )
    current_row = {
        name: row[name]
        for name in (
            *_GEOMETRY_COLUMNS,
            *_CURRENT_COLUMNS,
            *current_pinned.PLASMA_CURRENT_COLUMNS,
            "efit_grid_R",
            "efit_grid_Z",
        )
    }
    current_row["_source_path"] = str(path)
    description = dataset_machine_description(
        current_row, source_row=str(path)
    ).physical
    shipped = shipped_current_at(
        current_row,
        description,
        POLOIDAL_CONDUCTORS,
        time_ms,
    )
    adapter = _fixed_wiring_adapter(profile, shipped, geometry_digest(row))
    current = np.asarray(adapter.resolution.current(()), dtype=float)
    if len(current) != 24 or adapter.resolution.unknown_names:
        raise RuntimeError("the fixed-wiring circuit did not prescribe 24 currents")
    return PreparedFrame(
        selected=selected,
        row=row,
        profile=adapter.profile,
        seed=np.asarray(seed, dtype=float),
        label=np.asarray(label, dtype=float),
        wall=np.asarray(wall, dtype=float),
        reliable_surfaces=reliable,
        wall_statement=wall_statement,
        current=current,
        current_receipt=_current_receipt(adapter, current),
    )


def _fixed_wiring_adapter(
    profile: Any, shipped: dict[str, float], geometry: str
) -> DiiidCurrentAdapter:
    """Reuse exact response columns for frames with identical machine geometry."""

    grid_coordinate = np.asarray(profile.operator.grid.coordinate)
    wall_coordinate = np.asarray(profile.operator.wall.coordinate)
    key = (geometry, grid_coordinate.tobytes(), wall_coordinate.tobytes())
    cached = _RESPONSE_CACHE.get(key)
    if cached is None:
        adapter = complete_profile_current_adapter(
            profile,
            shipped_names=POLOIDAL_CONDUCTORS,
            shipped_current_a=shipped,
            use_circuit=True,
        )
        _RESPONSE_CACHE[key] = (
            adapter.profile.operator.grid.source_target,
            adapter.profile.operator.wall.source_target,
            adapter.response_receipt,
        )
        return adapter
    grid_response, wall_response, response_receipt = cached
    np.testing.assert_allclose(
        np.asarray(profile.operator.grid.source_target),
        np.asarray(grid_response[:, : len(POLOIDAL_CONDUCTORS)]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(profile.operator.wall.source_target),
        np.asarray(wall_response[:, : len(POLOIDAL_CONDUCTORS)]),
        rtol=0.0,
        atol=0.0,
    )
    resolution = resolve_diiid_currents(
        POLOIDAL_CONDUCTORS,
        shipped,
        use_circuit=True,
    )
    grid = replace(profile.operator.grid, source_target=grid_response)
    wall = replace(profile.operator.wall, source_target=wall_response)
    operator = replace(
        profile.operator,
        grid=grid,
        wall=wall,
        external_current=jnp.asarray(resolution.current(())),
    )
    return DiiidCurrentAdapter(
        profile=replace(profile, operator=operator),
        resolution=resolution,
        response_receipt=response_receipt,
    )


def _solve_public_seam(frame: PreparedFrame, seed: np.ndarray) -> SolveOutcome:
    """Run the public constrained host-Krylov seam and retain its terminal."""

    flux = jnp.asarray(seed)
    current = jnp.asarray(frame.current)
    target = frame.selected.recorded_plasma_current_a
    try:
        initial_image = np.asarray(
            frame.profile.flux_map(current, target_current=target)(flux), dtype=float
        )
    except CurrentNormalisationError as error:
        return SolveOutcome(
            state=np.asarray(seed, dtype=float),
            residual=float("inf"),
            iterations=0,
            termination=f"initial declared-current guard: {error}",
            amplitude=error.amplitude,
            achieved_current_a=float("nan"),
            residual_trajectory=(),
        )
    absolute_tolerance = RELATIVE_RESIDUAL_TOLERANCE * max(
        float(np.max(np.abs(initial_image))), 1.0e-30
    )
    try:
        equilibrium = frame.profile.solve(
            flux,
            route="host_krylov",
            current=current,
            target_current=target,
            method="gmres",
            inner_maxiter=HOST_INNER_ITERATIONS,
            maxiter=HOST_OUTER_ITERATIONS,
            f_tol=absolute_tolerance,
            line_search="armijo",
        )
        state = np.asarray(equilibrium.flux, dtype=float)
        trace = tuple(
            float(value)
            for value in np.asarray(equilibrium.fixed_point.trace, dtype=float)
            if np.isfinite(value)
        )
        termination = "public host Krylov solver returned"
        iterations = len(trace)
    except scipy.optimize.NoConvergence as error:
        state = np.asarray(error.args[0], dtype=float)
        equilibrium = frame.profile.observe(
            jnp.asarray(state),
            current=current,
            target_current=target,
        )
        trace = ()
        termination = "public host Krylov outer iteration ceiling exhausted"
        iterations = HOST_OUTER_ITERATIONS
    except ValueError as error:
        if "Jacobian inversion yielded zero vector" not in str(error):
            raise
        state = np.asarray(seed, dtype=float)
        equilibrium = frame.profile.observe(
            jnp.asarray(state),
            current=current,
            target_current=target,
        )
        trace = ()
        termination = f"public host Krylov rejected its initial Jacobian: {error}"
        iterations = 0
    achieved_current = float(np.sum(np.asarray(equilibrium.cell_current)))
    return SolveOutcome(
        state=state,
        residual=float(equilibrium.fixed_point.residual),
        iterations=iterations,
        termination=termination,
        amplitude=float(equilibrium.normalisation.amplitude),
        achieved_current_a=achieved_current,
        residual_trajectory=trace,
    )


def _route_record(
    frame: PreparedFrame,
    outcome: SolveOutcome,
    *,
    route_id: str,
    seed_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score a public-seam terminal in the landed validation shape."""

    _masks, topology = frame.profile.operator.read(jnp.asarray(outcome.state))
    finite = bool(np.all(np.isfinite(outcome.state)))
    current_error = abs(
        outcome.achieved_current_a - frame.selected.recorded_plasma_current_a
    ) / abs(frame.selected.recorded_plasma_current_a)
    converged = bool(
        finite
        and bool(topology.diverted)
        and np.isfinite(outcome.residual)
        and outcome.residual <= RELATIVE_RESIDUAL_TOLERANCE
        and np.isfinite(current_error)
        and current_error <= CURRENT_RELATIVE_ERROR_TOLERANCE
    )
    metrics, fields, terminal = _score_state(
        frame.profile,
        outcome.state,
        frame.label,
        frame.row,
        frame.selected.frame,
        converged,
    )
    x_point = terminal["x_point_rz_m"]
    label_x_point = terminal["label_x_point_rz_m"]
    terminal_x_separation = (
        float(np.linalg.norm(np.asarray(x_point) - np.asarray(label_x_point)))
        if x_point is not None
        else None
    )
    terminal.update(
        {
            "target_plasma_current_a": frame.selected.recorded_plasma_current_a,
            "achieved_plasma_current_a": outcome.achieved_current_a,
            "current_relative_error": _strict_float(current_error),
            "profile_amplitude": _strict_float(outcome.amplitude),
            "x_point_separation_m": _strict_float(terminal_x_separation)
            if terminal_x_separation is not None
            else None,
        }
    )
    return (
        {
            "route_id": route_id,
            "entry_point": "nova.equilibrium.forward.ForwardProfile.solve",
            "route": "host_krylov_with_declared_scalar_current",
            "attempted": True,
            "seed": seed_receipt,
            "options": {
                "outer_iterations": HOST_OUTER_ITERATIONS,
                "inner_iterations": HOST_INNER_ITERATIONS,
                "line_search": "armijo",
            },
            "residual_tolerance": RELATIVE_RESIDUAL_TOLERANCE,
            "current_tolerance": CURRENT_RELATIVE_ERROR_TOLERANCE,
            "converged": converged,
            "fixed_point_relative_residual": _strict_float(outcome.residual),
            "iterations": outcome.iterations,
            "residual_trajectory": list(outcome.residual_trajectory),
            "requested_class": "diverted",
            "achieved_class": "diverted" if bool(topology.diverted) else "limited",
            "topology_consistent": bool(topology.diverted),
            "termination": outcome.termination,
            "terminal_state": terminal,
            "metrics": metrics,
        },
        fields,
    )


def _neighbour_candidates(frame: PreparedFrame) -> list[int]:
    count = len(frame.row["efit_times"])
    return [
        frame.selected.frame + offset
        for offset in NEIGHBOUR_FRAME_OFFSETS
        if 0 <= frame.selected.frame + offset < count
    ]


def _find_warm_source(
    frame: PreparedFrame, checked_frames: set[int] | None = None
) -> tuple[list[dict[str, Any]], tuple[PreparedFrame, SolveOutcome] | None]:
    """Return the nearest declared convergence-qualified same-shot source."""

    skipped = set() if checked_frames is None else checked_frames
    checks = []
    for candidate_frame in _neighbour_candidates(frame):
        if candidate_frame in skipped:
            continue
        candidate = prepare_frame(frame.selected.path, candidate_frame)
        candidate_outcome = _solve_public_seam(candidate, candidate.seed)
        candidate_record, _candidate_fields = _route_record(
            candidate,
            candidate_outcome,
            route_id="neighbour_qualification",
            seed_receipt={
                "route": "same-frame EFIT label map in Nova convention",
                "source_frame": candidate_frame,
                "source_time_ms": candidate.selected.time_ms,
            },
        )
        checks.append(
            {
                "frame": candidate_frame,
                "time_ms": candidate.selected.time_ms,
                "time_offset_ms": (candidate.selected.time_ms - frame.selected.time_ms),
                "converged": candidate_record["converged"],
                "residual": candidate_record["fixed_point_relative_residual"],
                "achieved_class": candidate_record["achieved_class"],
                "topology_consistent": candidate_record["topology_consistent"],
            }
        )
        if candidate_record["converged"]:
            return checks, (candidate, candidate_outcome)
    return checks, None


def _select_terminal(routes: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    converged_routes = [route for route in routes.values() if route["converged"]]
    best = (
        min(
            converged_routes,
            key=lambda route: route["metrics"]["fractional_flux_rms"],
        )["route_id"]
        if converged_routes
        else None
    )
    selected = best or min(
        routes,
        key=lambda name: (
            routes[name]["fixed_point_relative_residual"]
            if routes[name]["fixed_point_relative_residual"] is not None
            else float("inf")
        ),
    )
    return best, selected


def solve_frame(
    selected: SelectedFrame,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Measure cold entry, then a same-shot warm arm only after a stall."""

    frame = prepare_frame(selected.path, selected.frame)
    if frame.selected != selected:
        raise RuntimeError("the selected frame changed while preparing the solve")
    cold_outcome = _solve_public_seam(frame, frame.seed)
    cold, cold_fields = _route_record(
        frame,
        cold_outcome,
        route_id="cold_start",
        seed_receipt={
            "route": "same-frame EFIT label map in Nova convention",
            "source_frame": selected.frame,
            "source_time_ms": selected.time_ms,
        },
    )
    routes = {"cold_start": cold}
    fields = {"cold_start": cold_fields}
    neighbour_checks: list[dict[str, Any]] = []
    warm_source: tuple[PreparedFrame, SolveOutcome] | None = None
    if not cold["converged"]:
        neighbour_checks, warm_source = _find_warm_source(frame)
    if warm_source is not None:
        candidate, candidate_outcome = warm_source
        warm_outcome = _solve_public_seam(frame, candidate_outcome.state)
        warm, warm_fields = _route_record(
            frame,
            warm_outcome,
            route_id="warm_neighbour",
            seed_receipt={
                "route": "converged frame from the declared same-shot ladder",
                "source_frame": candidate.selected.frame,
                "source_time_ms": candidate.selected.time_ms,
                "time_offset_ms": candidate.selected.time_ms - selected.time_ms,
                "source_residual": candidate_outcome.residual,
            },
        )
        routes["warm_neighbour"] = warm
        fields["warm_neighbour"] = warm_fields
    best, selected_terminal = _select_terminal(routes)
    record = {
        "shot": selected.path.name,
        "frame": selected.frame,
        "time_ms": selected.time_ms,
        "source_parquet": str(selected.path),
        "source_parquet_sha256": _sha256(selected.path),
        "geometry_digest": geometry_digest(frame.row),
        "qualification": {
            "finite_diverted_label": True,
            "polarity_screened": True,
            "calibration_frame_member": False,
            "calibration_shot_member": False,
            "reliable_flux_function_surfaces": frame.reliable_surfaces,
        },
        "source_and_seed": {
            "profile_functions": "extracted from the target-frame EFIT label",
            "cold_branch_seed": "target-frame EFIT label map in Nova convention",
            "warm_branch_seed": "converged same-shot ladder frame when available",
            "pseudo_wall": frame.wall_statement,
            "pseudo_wall_expansion": REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
        },
        "target_plasma_current": {
            "value_a": selected.recorded_plasma_current_a,
            "authority": "same-frame shipped magnetics_plasma_current channel",
            "role": "declared current-elimination target, never a label fit",
        },
        "circuit_driven": {
            "current_receipt": frame.current_receipt,
            "routes": routes,
            "best_converged_route": best,
            "selected_terminal_route": selected_terminal,
        },
        "warm_neighbour_search": {
            "triggered": not cold["converged"],
            "candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
            "selection_rule": (
                "symmetric geometric time-offset ladder, earlier before later; first "
                "convergence-qualified candidate, without label-score ranking"
            ),
            "checks": neighbour_checks,
            "qualified_source_found": warm_source is not None,
        },
    }
    return record, fields


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the two seeding arms without treating skips as failures."""

    route_table = {}
    for route_name in ROUTE_NAMES:
        routes = [
            record["circuit_driven"]["routes"][route_name]
            for record in records
            if route_name in record["circuit_driven"]["routes"]
        ]
        converged = [route for route in routes if route["converged"]]
        residuals = [
            route["fixed_point_relative_residual"]
            for route in routes
            if route["fixed_point_relative_residual"] is not None
        ]
        route_table[route_name] = {
            "attempted_frames": len(routes),
            "converged_frames": len(converged),
            "failed_frames": len(routes) - len(converged),
            "minimum_terminal_residual": min(residuals) if residuals else None,
            "maximum_terminal_residual": max(residuals) if residuals else None,
        }
    selected_rows = []
    for record in records:
        route_name = record["circuit_driven"]["selected_terminal_route"]
        route = record["circuit_driven"]["routes"][route_name]
        selected_rows.append(
            {
                "shot": record["shot"],
                "frame": record["frame"],
                "seeding_route": route_name,
                "converged": route["converged"],
                "terminal_residual": route["fixed_point_relative_residual"],
                "achieved_class": route["achieved_class"],
                "topology_consistent": route["topology_consistent"],
                "x_point_separation_m": route["terminal_state"]["x_point_separation_m"],
                "fractional_flux_rms": route["metrics"]["fractional_flux_rms"],
            }
        )
    qualified = sum(row["converged"] for row in selected_rows)
    return {
        "frame_count": len(records),
        "shot_count": len({record["shot"] for record in records}),
        "routes": route_table,
        "selected_terminal_per_frame": selected_rows,
        "convergence_qualified_frames": qualified,
    }


def render_summary(records: list[dict[str, Any]], path: Path) -> None:
    """Plot residual and X-point outcomes for every attempted seeding route."""

    labels = [f"{record['shot'][10:18]}\nf{record['frame']}" for record in records]
    x = np.arange(len(records), dtype=float)
    width = 0.34
    figure, (residual_axis, x_axis) = plt.subplots(
        2, 1, figsize=(9.0, 6.8), constrained_layout=True
    )
    for index, route_name in enumerate(ROUTE_NAMES):
        residuals = []
        separations = []
        for record in records:
            route = record["circuit_driven"]["routes"].get(route_name)
            residuals.append(
                np.nan if route is None else route["fixed_point_relative_residual"]
            )
            separations.append(
                np.nan
                if route is None
                else route["terminal_state"]["x_point_separation_m"]
            )
        position = x + (index - 0.5) * width
        residual_axis.bar(
            position, residuals, width, label=route_name.replace("_", " ")
        )
        x_axis.bar(position, separations, width, label=route_name.replace("_", " "))
    residual_axis.axhline(
        RELATIVE_RESIDUAL_TOLERANCE,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="1e-6 criterion",
    )
    residual_axis.set_yscale("log")
    residual_axis.set_ylabel("terminal relative residual")
    residual_axis.set_xticks(x, labels)
    residual_axis.legend(fontsize=8, ncols=3)
    x_axis.set_ylabel("terminal X-point separation [m]")
    x_axis.set_xticks(x, labels)
    x_axis.set_xlabel("score-blind validation frame")
    figure.suptitle("Current-constrained cold entry and same-shot warm recovery")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(data: Path, output: Path, frame_count: int = FRAME_COUNT) -> dict[str, Any]:
    """Run the fixed five-frame constrained cold-start measurement."""

    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    calibration_frames, calibration_shots, calibration = calibration_population()
    affected = polarity_population()
    selected, _low_current = select_frames(
        list(data.glob("*.parquet")), calibration_shots, affected, frame_count
    )
    selected_pairs = {(item.path.name, item.frame) for item in selected}
    if selected_pairs & calibration_frames:
        raise RuntimeError("selected frame overlaps the calibration bank")
    records = [solve_frame(item)[0] for item in selected]
    aggregate = summarize(records)
    qualified = aggregate["convergence_qualified_frames"]
    figure_path = output / FIGURE_NAME
    render_summary(records, figure_path)
    receipt = {
        "measurement": (
            "cold-start and same-shot warm recovery through the public "
            "current-constrained DIII-D forward seam"
        ),
        "selection": {
            "rule": (
                "the exact five score-blind circuit-driven validation frames; "
                "no solve score changes the cohort"
            ),
            "selected_frames": [
                {
                    "shot": item.path.name,
                    "frame": item.frame,
                    "time_ms": item.time_ms,
                    "recorded_plasma_current_a": item.recorded_plasma_current_a,
                }
                for item in selected
            ],
            "selected_frame_count": len(selected),
            "selected_shot_count": len({item.path.name for item in selected}),
            "calibration_bank": {
                "frame_count": len(calibration_frames),
                "shot_count": len(calibration_shots),
                "selection_receipt_frames": calibration["selection"]["frames"],
                "exact_selected_pair_intersection": [],
            },
            "polarity_affected_shot_count": len(affected),
            "polarity_selected_intersection": [],
        },
        "arms": {
            "cold_start": "same-frame labelled branch seed through the public seam",
            "warm_neighbour": (
                "first convergence-qualified frame on the declared same-shot "
                "time-offset ladder, evaluated only when the cold target stalls"
            ),
            "shared": (
                "fixed-wiring 24-current circuit, target-frame prescribed source "
                "functions, declared plasma current and host Newton-Krylov route"
            ),
        },
        "current_path_audit": {
            "fixed_wiring_current_count": 24,
            "unknown_current_parameters": 0,
            "label_derived_current_reads": 0,
            "per_frame_current_fits": 0,
            "plasma_current_channel": "magnetics_plasma_current",
        },
        "solver": {
            "entry_point": "nova.equilibrium.forward.ForwardProfile.solve",
            "route": "host_krylov",
            "target_current_supplied": True,
            "relative_residual_tolerance": RELATIVE_RESIDUAL_TOLERANCE,
            "current_relative_error_tolerance": CURRENT_RELATIVE_ERROR_TOLERANCE,
            "required_topology": "diverted",
            "anderson_present": False,
            "neighbour_candidate_offsets": list(NEIGHBOUR_FRAME_OFFSETS),
        },
        "comparison": {
            "receipt_shape": (
                "circuit-driven validation frame, route, terminal_state and metrics"
            ),
            "flux_gauge": "one additive constant over the labelled LCFS interior",
            "label_representability_ceiling_fractional_rms": (
                LABEL_REPRESENTABILITY_CEILING_FRACTIONAL_RMS
            ),
            "terminal_x_point_scope": (
                "reported for every finite terminal; non-converged values are "
                "trajectory diagnostics, not equilibrium scores"
            ),
        },
        "aggregate": aggregate,
        "verdict": {
            "convergence_qualified_frames": qualified,
            "three_frame_convergence_floor_reached": qualified >= 3,
            "recovery_demonstrated": qualified >= 3,
            "statement": (
                "Recovery is demonstrated if and only if at least three of the "
                "fixed five frames terminate finite, diverted, current-consistent "
                "and at or below 1e-6 relative residual."
            ),
        },
        "pf_active_circuit": PF_ACTIVE_CIRCUIT.as_record(),
        "frames": records,
        "artifacts": {
            "receipt": str(output / RECEIPT_NAME),
            "summary_figure": str(figure_path),
        },
    }
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def resume_warm(output: Path) -> dict[str, Any]:
    """Extend an honestly banked cold receipt along the declared neighbour ladder."""

    configure_dtypes()
    receipt_path = output / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    records = receipt["frames"]
    for record in records:
        routes = record["circuit_driven"]["routes"]
        if routes["cold_start"]["converged"] or "warm_neighbour" in routes:
            continue
        selected = SelectedFrame(
            path=Path(record["source_parquet"]),
            frame=int(record["frame"]),
            time_ms=float(record["time_ms"]),
            recorded_plasma_current_a=float(record["target_plasma_current"]["value_a"]),
        )
        frame = prepare_frame(selected.path, selected.frame)
        if frame.selected != selected:
            raise RuntimeError("the banked target changed before warm recovery")
        search = record["warm_neighbour_search"]
        checked = {int(item["frame"]) for item in search["checks"]}
        checks, warm_source = _find_warm_source(frame, checked)
        search["checks"].extend(checks)
        search["candidate_offsets"] = list(NEIGHBOUR_FRAME_OFFSETS)
        search["selection_rule"] = (
            "symmetric geometric time-offset ladder, earlier before later; first "
            "convergence-qualified candidate, without label-score ranking"
        )
        search["qualified_source_found"] = warm_source is not None
        if warm_source is not None:
            candidate, candidate_outcome = warm_source
            warm_outcome = _solve_public_seam(frame, candidate_outcome.state)
            warm, _warm_fields = _route_record(
                frame,
                warm_outcome,
                route_id="warm_neighbour",
                seed_receipt={
                    "route": "converged same-shot time-sequence frame",
                    "source_frame": candidate.selected.frame,
                    "source_time_ms": candidate.selected.time_ms,
                    "time_offset_ms": (candidate.selected.time_ms - selected.time_ms),
                    "source_residual": candidate_outcome.residual,
                },
            )
            routes["warm_neighbour"] = warm
        best, selected_terminal = _select_terminal(routes)
        record["circuit_driven"]["best_converged_route"] = best
        record["circuit_driven"]["selected_terminal_route"] = selected_terminal
    aggregate = summarize(records)
    qualified = aggregate["convergence_qualified_frames"]
    figure_path = output / FIGURE_NAME
    render_summary(records, figure_path)
    receipt["aggregate"] = aggregate
    receipt["verdict"].update(
        {
            "convergence_qualified_frames": qualified,
            "three_frame_convergence_floor_reached": qualified >= 3,
            "recovery_demonstrated": qualified >= 3,
        }
    )
    receipt["solver"]["neighbour_candidate_offsets"] = list(NEIGHBOUR_FRAME_OFFSETS)
    receipt["measurement"] = (
        "cold-start and same-shot warm recovery through the public "
        "current-constrained DIII-D forward seam"
    )
    receipt["arms"]["warm_neighbour"] = (
        "first convergence-qualified frame on the declared same-shot time-offset "
        "ladder, evaluated only when the cold target stalls"
    )
    for record in records:
        record["source_and_seed"]["warm_branch_seed"] = (
            "converged same-shot ladder frame when available"
        )
    receipt["measurement_update"] = {
        "route": "resume only the unmeasured declared warm-neighbour candidates",
        "cold_rows_preserved": len(records),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--resume-warm", action="store_true")
    arguments = parser.parse_args()
    receipt = (
        resume_warm(arguments.output)
        if arguments.resume_warm
        else run(arguments.data, arguments.output, arguments.frames)
    )
    print(json.dumps(receipt["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
