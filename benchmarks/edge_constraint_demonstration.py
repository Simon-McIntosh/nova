#!/usr/bin/env python3
"""Measure synthetic outboard-edge commands on one converged MAST equilibrium.

The named quantity is the outboard intersection of the last closed flux
surface with the horizontal chord through the seed magnetic axis.  A single
boundary-referenced isoflux row is placed five, ten, and twenty millimetres
outward on that chord.  Each command starts from the same converged free state,
uses the constrained reduced route with a one-kiloampere per-trip cap, and is
persisted after every trip.  A separate bracketed sequence of unconstrained
free solves measures the circuit current that reaches the same edge radius.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import settled_mask_stall as settled
from nova.equilibrium import reduced_newton
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    IsofluxConstraint,
    sample_lattice_flux,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/constraint-augmented-newton-krylov/edge-constraint/receipt.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/edge-constraint/"
    "edge-contours.png"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-handoff/"
    "edge-constraint/report.md"
)
TARGET = (22086, 43)
COMMANDS_M = (0.005, 0.010, 0.020)
POSITION_TOLERANCE_M = 2.5e-4
CURRENT_STEP_CAP_A = 1_000.0
CURRENT_CEILING_A = 100_000.0
CURRENT_AGREEMENT_PERCENT = 10.0
NEWTON_STEPS = 24
ACTIVE_SET_TRIPS = 12
FREE_REFERENCE_REFINEMENTS = 4


def _source_revision() -> str:
    """Return the revision whose code produced the receipt."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _write(payload: dict[str, Any], output: Path) -> None:
    """Atomically replace a partially complete receipt."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _grid(profile, flux) -> jax.Array:
    """Return one state on the profile lattice."""
    lattice = profile.lattice
    return jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)


def outboard_midplane_edge_radius(
    profile, flux, *, height_m: float, requested_class=None
) -> float:
    """Return the outermost boundary crossing on one horizontal chord."""
    _masks, topology = profile.operator.read(jnp.asarray(flux), requested_class)
    level = float(np.asarray(topology.boundary_flux))
    grid = _grid(profile, flux)
    radius = np.asarray(profile.lattice.radius, dtype=float)
    dense_radius = np.linspace(radius[0], radius[-1], 8 * radius.size + 1)
    points = jnp.column_stack(
        (
            jnp.asarray(dense_radius),
            jnp.full(dense_radius.size, float(height_m), dtype=jnp.float64),
        )
    )
    sampled = np.asarray(
        jax.vmap(lambda point: sample_lattice_flux(profile.lattice, grid, point))(
            points
        ),
        dtype=float,
    )
    signed = float(profile.operator.polarity) * (sampled - level)
    crossings: list[tuple[float, float]] = []
    for lower, upper, left, right in zip(
        dense_radius[:-1], dense_radius[1:], signed[:-1], signed[1:], strict=True
    ):
        if left == 0.0:
            crossings.append((float(lower), float(lower)))
        elif left * right < 0.0 or right == 0.0:
            crossings.append((float(lower), float(upper)))
    if not crossings:
        raise ValueError(
            f"no boundary crossing found at the named chord height {height_m:.9f} m"
        )
    lower, upper = crossings[-1]
    if lower == upper:
        return lower
    for _ in range(52):
        middle = 0.5 * (lower + upper)
        value = float(
            np.asarray(
                sample_lattice_flux(
                    profile.lattice,
                    grid,
                    jnp.asarray([middle, height_m], dtype=jnp.float64),
                )
            )
        )
        middle_signed = float(profile.operator.polarity) * (value - level)
        lower_value = float(
            np.asarray(
                sample_lattice_flux(
                    profile.lattice,
                    grid,
                    jnp.asarray([lower, height_m], dtype=jnp.float64),
                )
            )
        )
        lower_signed = float(profile.operator.polarity) * (lower_value - level)
        if lower_signed == 0.0 or lower_signed * middle_signed <= 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def _edge_pair(
    profile,
    flux,
    *,
    point_rz_m: np.ndarray,
    flux_span_wb: float,
    requested_class,
    target_current,
    prescribed_current,
    program,
    circuits,
) -> tuple[ConstraintPair, Any]:
    """Return one matrix-led boundary-flux point row."""
    functional = IsofluxConstraint(point_count=1, reference="boundary")
    grid = _grid(profile, flux)
    radial_gradient = float(
        np.asarray(
            jax.grad(
                lambda radius: sample_lattice_flux(
                    profile.lattice,
                    grid,
                    jnp.asarray([radius, point_rz_m[1]], dtype=jnp.float64),
                )
            )(jnp.asarray(point_rz_m[0], dtype=jnp.float64))
        )
    )
    flux_tolerance = max(
        abs(radial_gradient) * POSITION_TOLERANCE_M,
        1.0e-10 * abs(flux_span_wb),
    )
    seeded = ConstraintPair(
        functional=functional,
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.zeros(1, dtype=jnp.float64),
            tolerance=jnp.asarray([flux_tolerance], dtype=jnp.float64),
            scale=jnp.asarray([abs(flux_span_wb)], dtype=jnp.float64),
            initial_unknown=jnp.zeros(1, dtype=jnp.float64),
            payload=jnp.asarray(point_rz_m, dtype=jnp.float64)[None, :],
        ),
    )
    (derived,), selection = reduced_newton.derive_reduced_constraint_pairs(
        profile,
        (seeded,),
        jnp.asarray(flux),
        requested_class=requested_class,
        target_current=target_current,
        prescribed_current=prescribed_current,
        program=program,
        circuits=circuits,
    )
    return derived, selection


def _carry_pair(pair: ConstraintPair, initial_unknown) -> ConstraintPair:
    """Carry the terminal unknown into the next independently persisted trip."""
    return replace(
        pair,
        binding=replace(pair.binding, initial_unknown=jnp.asarray(initial_unknown)),
    )


def _constraint_merit(
    profile, score_state, result, unknown, base_state, requested_class
) -> float:
    """Read the production augmented merit at one trip boundary."""
    program = result.program
    if program is None:
        raise ValueError("constrained solve did not return its reduced program")
    amplitudes = program.kernels["initial_gather"](jnp.asarray(score_state))
    reduced = jnp.concatenate((amplitudes, jnp.ravel(jnp.asarray(unknown))))
    shadow = jnp.ravel(
        jnp.asarray(
            profile.operator.residual_shadow_mask(
                jnp.asarray(base_state), requested_class=requested_class
            ),
            dtype=bool,
        )
    )
    scores = program.kernels["step_scores"](reduced, shadow, jnp.asarray(base_state))
    jax.block_until_ready(scores.merit)
    return float(np.asarray(scores.merit))


def _command_row(
    profile,
    free,
    pair,
    *,
    command_m: float,
    edge_radius_m: float,
    chord_height_m: float,
    requested_class,
    target_current,
    prescribed_current,
    persist: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], Any]:
    """Drive one edge target through separately recorded capped trips."""
    current_state = free.state
    current_unknown = jnp.asarray(pair.binding.initial_unknown)
    program = None
    merits: list[float] = []
    trips: list[dict[str, Any]] = []
    terminal_result = None
    refusal = None
    row = {
        "commanded_displacement_mm": 1.0e3 * command_m,
        "target_point_rz_m": [edge_radius_m + command_m, chord_height_m],
        "position_tolerance_mm": 1.0e3 * POSITION_TOLERANCE_M,
        "current_step_cap_a": CURRENT_STEP_CAP_A,
        "status": "running",
        "trips": trips,
    }
    persist(row)
    for trip_index in range(1, ACTIVE_SET_TRIPS + 1):
        start_radius = outboard_midplane_edge_radius(
            profile,
            current_state,
            height_m=chord_height_m,
            requested_class=requested_class,
        )
        trip_pair = _carry_pair(pair, current_unknown)
        result = reduced_newton.solve_constrained_reduced_newton(
            profile,
            jnp.asarray(current_state),
            constraint_pairs=(trip_pair,),
            requested_class=requested_class,
            target_current=target_current,
            prescribed_current=jnp.asarray(prescribed_current),
            tolerance=settled.FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            active_set_steps=1,
            constraint_current_step_cap=CURRENT_STEP_CAP_A,
            constraint_current_ceiling=CURRENT_CEILING_A,
            program=program,
        )
        if not merits:
            merits.append(
                _constraint_merit(
                    profile,
                    current_state,
                    result,
                    current_unknown,
                    current_state,
                    requested_class,
                )
            )
        terminal_result = result
        final_radius = outboard_midplane_edge_radius(
            profile,
            result.state,
            height_m=chord_height_m,
            requested_class=requested_class,
        )
        achieved_displacement = final_radius - edge_radius_m
        position_error = achieved_displacement - command_m
        record = result.constraints[0]
        previous_current = float(
            np.asarray(pair.unknown.physical_value(current_unknown))[0]
        )
        cumulative_current = float(np.asarray(record.physical_unknown)[0])
        applied_current = cumulative_current - previous_current
        direction = np.asarray(pair.unknown.direction, dtype=float)[:, 0]
        post_merit = _constraint_merit(
            profile,
            result.state,
            result,
            result.compensating_unknown,
            current_state,
            requested_class,
        )
        prior_merit = merits[-1]
        merits.append(post_merit)
        newton_step_count = int(sum(result.newton_steps_per_trip))
        overshot = bool(
            newton_step_count > 0
            and achieved_displacement > command_m + POSITION_TOLERANCE_M
        )
        trip = {
            "trip": trip_index,
            "newton_steps": newton_step_count,
            "accepted_factors": [
                entry.accepted_factor
                for entry in result.steps
                if entry.accepted_factor is not None
            ],
            "merit_before": prior_merit,
            "merit_after": post_merit,
            "merit_change": post_merit - prior_merit,
            "applied_compensating_current_a": applied_current,
            "largest_applied_compensating_current_a": abs(applied_current),
            "largest_per_circuit_application_a": float(
                np.max(np.abs(applied_current * direction))
            ),
            "cumulative_compensating_current_a": cumulative_current,
            "achieved_edge_radius_m": final_radius,
            "achieved_displacement_mm": 1.0e3 * achieved_displacement,
            "position_error_mm": 1.0e3 * position_error,
            "row_physical_residual_wb": float(np.asarray(record.physical_residual)[0]),
            "row_qualified": bool(np.asarray(record.qualified)[0]),
            "accepted_overshoot": overshot,
            "termination": result.termination_name,
            "terminal_fixed_point_residual": float(result.terminal_residual),
            "edge_motion_this_trip_mm": 1.0e3 * (final_radius - start_radius),
        }
        trips.append(trip)
        row.update(
            {
                "status": "running",
                "merit_sequence": merits,
                "terminal_position_error_mm": trip["position_error_mm"],
                "cumulative_compensating_current_a": cumulative_current,
            }
        )
        persist(row)
        within_position = abs(position_error) <= POSITION_TOLERANCE_M
        if (
            result.converged
            and bool(np.asarray(record.qualified)[0])
            and within_position
        ):
            row["status"] = "converged"
            break
        if result.termination_name == "sufficient_decrease_refused" or (
            newton_step_count == 0 and post_merit >= prior_merit
        ):
            refusal = {
                "trip": trip_index,
                "reason": "trip-stopped-lowering-the-augmented-merit",
                "merit_before": prior_merit,
                "merit_after": post_merit,
            }
            row["status"] = "refused"
            break
        current_state = result.state
        current_unknown = result.compensating_unknown
        program = result.program
    if terminal_result is None:
        raise RuntimeError("edge command did not execute a trip")
    if row["status"] == "running":
        refusal = {
            "trip": len(trips),
            "reason": "trip-budget-exhausted-before-the-edge-tolerance",
            "merit_before": merits[-2],
            "merit_after": merits[-1],
        }
        row["status"] = "refused"
    nonzero = [merits[0]] + [
        trip["merit_after"] for trip in trips if trip["newton_steps"] > 0
    ]
    row.update(
        {
            "trip_count": len(trips),
            "merit_sequence": merits,
            "nonzero_step_merit_sequence": nonzero,
            "merit_monotone_where_steps_are_nonzero": all(
                after < before for before, after in zip(nonzero, nonzero[1:])
            ),
            "largest_trip_application_a": max(
                abs(trip["applied_compensating_current_a"]) for trip in trips
            ),
            "current_cap_respected": all(
                abs(trip["applied_compensating_current_a"])
                <= CURRENT_STEP_CAP_A + 1.0e-9
                for trip in trips
            ),
            "accepted_overshoot": any(trip["accepted_overshoot"] for trip in trips),
            "total_compensating_current_a": trips[-1][
                "cumulative_compensating_current_a"
            ],
            "terminal_edge_radius_m": trips[-1]["achieved_edge_radius_m"],
            "terminal_displacement_mm": trips[-1]["achieved_displacement_mm"],
            "terminal_position_error_mm": trips[-1]["position_error_mm"],
            "terminal_fixed_point_residual": trips[-1]["terminal_fixed_point_residual"],
            "refusal": refusal,
        }
    )
    persist(row)
    return row, terminal_result


def _free_sample(
    profile,
    free,
    *,
    current_delta_a: float,
    direction: np.ndarray,
    prescribed_current: np.ndarray,
    requested_class,
    target_current,
    edge_radius_m: float,
    chord_height_m: float,
    target_displacement_m: float,
    program,
) -> tuple[dict[str, Any], Any]:
    """Measure one unconstrained solve at a declared circuit-current delta."""
    try:
        result = reduced_newton.solve_reduced_newton(
            profile.operator,
            jnp.asarray(free.state),
            requested_class=requested_class,
            target_current=target_current,
            prescribed_current=jnp.asarray(
                prescribed_current + current_delta_a * direction
            ),
            tolerance=settled.FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            active_set_steps=ACTIVE_SET_TRIPS,
            program=program,
        )
        radius = outboard_midplane_edge_radius(
            profile,
            result.state,
            height_m=chord_height_m,
            requested_class=requested_class,
        )
    except Exception as error:
        return (
            {
                "current_delta_a": current_delta_a,
                "converged": False,
                "termination": f"{type(error).__name__}: {error}",
                "edge_displacement_mm": None,
                "position_error_mm": None,
                "terminal_fixed_point_residual": None,
            },
            program,
        )
    displacement = radius - edge_radius_m
    return (
        {
            "current_delta_a": current_delta_a,
            "converged": bool(result.converged),
            "termination": result.termination_name,
            "trip_count": int(result.active_set_iterations),
            "edge_radius_m": radius,
            "edge_displacement_mm": 1.0e3 * displacement,
            "position_error_mm": 1.0e3 * (displacement - target_displacement_m),
            "terminal_fixed_point_residual": float(result.terminal_residual),
        },
        result.program,
    )


def _free_reference(
    profile,
    free,
    *,
    constrained_current_a: float,
    direction: np.ndarray,
    prescribed_current: np.ndarray,
    requested_class,
    target_current,
    edge_radius_m: float,
    chord_height_m: float,
    target_displacement_m: float,
    persist: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Bracket and directly measure the free current reaching one edge target."""
    samples: list[dict[str, Any]] = []
    program = free.program

    def sample_at(current_delta_a: float) -> None:
        nonlocal program
        if any(
            np.isclose(current_delta_a, sample["current_delta_a"], atol=1.0e-9)
            for sample in samples
        ):
            return
        sample, program = _free_sample(
            profile,
            free,
            current_delta_a=current_delta_a,
            direction=direction,
            prescribed_current=prescribed_current,
            requested_class=requested_class,
            target_current=target_current,
            edge_radius_m=edge_radius_m,
            chord_height_m=chord_height_m,
            target_displacement_m=target_displacement_m,
            program=program,
        )
        samples.append(sample)
        persist(
            {
                "method": "direct free solves bracketed and refined by secant",
                "samples": samples,
                "reached": False,
            }
        )

    def find_bracket() -> tuple[dict[str, Any], dict[str, Any]] | None:
        converged = sorted(
            (sample for sample in samples if sample["converged"]),
            key=lambda sample: sample["current_delta_a"],
        )
        return next(
            (
                (lower, upper)
                for lower, upper in zip(converged, converged[1:])
                if lower["position_error_mm"] * upper["position_error_mm"] <= 0.0
            ),
            None,
        )

    for factor in (0.0, 0.5, 1.0, 1.5):
        sample_at(factor * constrained_current_a)
    bracket = find_bracket()
    if bracket is None:
        for magnitude_a in (CURRENT_STEP_CAP_A, 2_000.0, 4_000.0, 8_000.0):
            sample_at(-magnitude_a)
            sample_at(magnitude_a)
            bracket = find_bracket()
            if bracket is not None:
                break
    if bracket is not None:
        lower, upper = bracket
        endpoint_reached = (
            min(abs(lower["position_error_mm"]), abs(upper["position_error_mm"]))
            <= 1.0e3 * POSITION_TOLERANCE_M
        )
        for _ in range(0 if endpoint_reached else FREE_REFERENCE_REFINEMENTS):
            denominator = upper["position_error_mm"] - lower["position_error_mm"]
            if denominator == 0.0:
                break
            estimate = (
                lower["current_delta_a"]
                - lower["position_error_mm"]
                * (upper["current_delta_a"] - lower["current_delta_a"])
                / denominator
            )
            sample, program = _free_sample(
                profile,
                free,
                current_delta_a=estimate,
                direction=direction,
                prescribed_current=prescribed_current,
                requested_class=requested_class,
                target_current=target_current,
                edge_radius_m=edge_radius_m,
                chord_height_m=chord_height_m,
                target_displacement_m=target_displacement_m,
                program=program,
            )
            samples.append(sample)
            persist(
                {
                    "method": "direct free solves bracketed and refined by secant",
                    "samples": samples,
                    "reached": False,
                }
            )
            if not sample["converged"]:
                break
            if abs(sample["position_error_mm"]) <= 1.0e3 * POSITION_TOLERANCE_M:
                break
            if lower["position_error_mm"] * sample["position_error_mm"] <= 0.0:
                upper = sample
            else:
                lower = sample
    qualifying = [
        sample
        for sample in samples
        if sample["converged"]
        and abs(sample["position_error_mm"]) <= 1.0e3 * POSITION_TOLERANCE_M
    ]
    reference = min(
        qualifying,
        key=lambda sample: abs(sample["position_error_mm"]),
        default=None,
    )
    result = {
        "method": (
            "direct free solves bracketed and refined by secant; the constrained "
            "current sets the probe scale but is not extrapolated"
        ),
        "position_tolerance_mm": 1.0e3 * POSITION_TOLERANCE_M,
        "refinement_limit": FREE_REFERENCE_REFINEMENTS,
        "samples": samples,
        "bracketed": bracket is not None,
        "reached": reference is not None,
        "current_delta_a": (
            None if reference is None else reference["current_delta_a"]
        ),
        "edge_displacement_mm": (
            None if reference is None else reference["edge_displacement_mm"]
        ),
        "position_error_mm": (
            None if reference is None else reference["position_error_mm"]
        ),
    }
    persist(result)
    return result


def _render(
    profile,
    free_state,
    terminal_states,
    commands,
    *,
    chord_height_m: float,
    requested_class,
    output: Path,
) -> None:
    """Draw free and terminal flux on shared levels with both null sets."""
    lattice = profile.lattice
    radius = np.asarray(lattice.radius, dtype=float)
    height = np.asarray(lattice.height, dtype=float)
    free_grid = np.asarray(_grid(profile, free_state), dtype=float).T
    terminal_grids = [
        np.asarray(_grid(profile, state), dtype=float).T for state in terminal_states
    ]
    levels = poloidal.contour_levels(
        np.concatenate([free_grid.ravel(), *[grid.ravel() for grid in terminal_grids]]),
        count=12,
    )
    wall = np.asarray(profile.operator.wall.coordinate, dtype=float)
    _masks, free_topology = profile.operator.read(
        jnp.asarray(free_state), requested_class
    )
    figure, axes = plt.subplots(1, len(commands), figsize=(4.6 * len(commands), 5.0))
    axes = np.atleast_1d(axes)
    free_style = DEFAULT_INK.variant(
        axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
    )
    solved_style = DEFAULT_INK.variant(
        axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
    )
    for axis, grid, command, state in zip(
        axes, terminal_grids, commands, terminal_states, strict=True
    ):
        poloidal.draw_flux_contours(
            axis, radius, height, free_grid, levels, color="#3366cc"
        )
        poloidal.draw_flux_contours(axis, radius, height, grid, levels, color="#cc7722")
        poloidal.draw_wall(axis, units=(wall,))
        poloidal.draw_nulls(
            axis,
            magnetic_axis=np.asarray(free_topology.axis, dtype=float),
            x_points=np.asarray(free_topology.x_point, dtype=float)[None, :],
            style=free_style,
            contain=(wall,),
        )
        _terminal_masks, terminal_topology = profile.operator.read(
            jnp.asarray(state), requested_class
        )
        poloidal.draw_nulls(
            axis,
            magnetic_axis=np.asarray(terminal_topology.axis, dtype=float),
            x_points=np.asarray(terminal_topology.x_point, dtype=float)[None, :],
            style=solved_style,
            contain=(wall,),
        )
        target = np.asarray(command["target_point_rz_m"], dtype=float)
        achieved = np.asarray(
            [command["terminal_edge_radius_m"], chord_height_m], dtype=float
        )
        axis.plot(
            target[0],
            target[1],
            marker="o",
            markerfacecolor="none",
            color="black",
            markersize=5,
        )
        axis.plot(achieved[0], achieved[1], marker="x", color="black", markersize=5)
        poloidal_axes(axis)
        axis.set_title(
            f"{command['commanded_displacement_mm']:.0f} mm command\n"
            f"error {command['terminal_position_error_mm']:+.3f} mm; "
            f"{command['trip_count']} trips",
            fontsize=9,
        )
    figure.suptitle(
        "MAST 22086/43 edge constraint: free blue, terminal ochre; shared Wb levels",
        y=0.98,
    )
    figure.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.87, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(
    payload: dict[str, Any], receipt: Path, figure: Path, report: Path
) -> None:
    """Write the compact human-readable reading beside the machine receipt."""
    lines = [
        "# Synthetic outboard-edge constraint receipt",
        "",
        f"Source revision: `{payload['source']['revision']}`. "
        f"MAST row: `{payload['identity']}`.",
        "",
        "The named quantity is the outboard LCFS intersection with the horizontal "
        "chord through the seed magnetic axis. "
        f"The position tolerance is "
        f"{payload['configuration']['position_tolerance_mm']:.3f} mm and each "
        "trip is capped at "
        f"{payload['configuration']['current_step_cap_a']:.0f} A.",
        "",
        "| Command | Status | Error [mm] | Trips | Largest trip [A] | Total [A] "
        "| Direct free [A] | Overshoot |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for command in payload["commands"]:
        reference = command.get("free_solve_reference", {})
        lines.append(
            (
                "| {command:.0f} | {status} | {error:+.3f} | {trips} | "
                "{largest:.3f} | {total:.3f} | {free} | {overshoot} |"
            ).format(
                command=command["commanded_displacement_mm"],
                status=command["status"],
                error=command["terminal_position_error_mm"],
                trips=command["trip_count"],
                largest=command["largest_trip_application_a"],
                total=command["total_compensating_current_a"],
                free=(
                    "absent"
                    if reference.get("current_delta_a") is None
                    else f"{reference['current_delta_a']:.3f}"
                ),
                overshoot=command["accepted_overshoot"],
            )
        )
    lines.extend(
        [
            "",
            f"Machine receipt: `{receipt}`.",
            f"Flux-contour figure: `{figure}`.",
            "",
            f"Overall verdict: `{payload['verdict']['status']}`.",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure(*, output: Path, figure: Path, report: Path) -> dict[str, Any]:
    """Run the complete three-command receipt on one reserved accelerator."""
    configure_dtypes()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("the edge receipt requires JAX x64 before arrays are built")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = settled._persisted_response_cache(
        settled.response_carrier.DEFAULT_CARRIER,
        settled.response_carrier.DEFAULT_RECEIPT,
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in settled.select_slices_by_shot(
            settled.DECOMPOSITION_BANK
        )
    }
    selected_row, qualification = selected[TARGET]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, response_cache
    )
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    prescribed_current = np.asarray(
        profile.operator.prescribed_current_field.current, dtype=float
    )
    circuits = sorted(
        int(item["stored_circuit"]) - 1 for item in policy["active_mapping"]
    )
    names = {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }
    free = reduced_newton.solve_reduced_newton(
        profile.operator,
        jnp.asarray(passive_case["state"]),
        requested_class=requested,
        target_current=target_current,
        prescribed_current=jnp.asarray(prescribed_current),
        tolerance=settled.FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        active_set_steps=ACTIVE_SET_TRIPS,
    )
    if not free.converged:
        raise RuntimeError(
            f"the MAST seed did not converge: {free.termination_name}, "
            f"residual {free.terminal_residual:.6e}"
        )
    _masks, topology = profile.operator.read(jnp.asarray(free.state), requested)
    chord_height = float(np.asarray(topology.axis)[1])
    edge_radius = outboard_midplane_edge_radius(
        profile,
        free.state,
        height_m=chord_height,
        requested_class=requested,
    )
    flux_span = abs(float(np.asarray(topology.flux_span)))
    payload: dict[str, Any] = {
        "receipt": "synthetic outboard-midplane edge displacement",
        "identity": f"{TARGET[0]}/{TARGET[1]}",
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            },
        },
        "configuration": {
            "quantity": (
                "outboard intersection of the last closed flux surface with "
                "the horizontal chord through the seed magnetic axis"
            ),
            "seed_edge_point_rz_m": [edge_radius, chord_height],
            "commanded_displacements_mm": [1.0e3 * value for value in COMMANDS_M],
            "position_tolerance_mm": 1.0e3 * POSITION_TOLERANCE_M,
            "current_step_cap_a": CURRENT_STEP_CAP_A,
            "current_ceiling_a": CURRENT_CEILING_A,
            "current_agreement_percent": CURRENT_AGREEMENT_PERCENT,
            "active_set_trip_budget": ACTIVE_SET_TRIPS,
            "newton_steps_per_trip": NEWTON_STEPS,
            "constraint": (
                "one IsofluxConstraint row referenced to the terminal boundary flux"
            ),
            "compensator": "matrix-led direction over the active MAST circuit mapping",
            "persistent_compilation_cache": cache.receipt(),
        },
        "evidence_inputs": {
            "response_carrier": carrier_evidence,
            "seed_terminal_residual": float(free.terminal_residual),
            "seed_trip_count": int(free.active_set_iterations),
            "seed_axis_rz_m": np.asarray(topology.axis, dtype=float).tolist(),
            "seed_x_point_rz_m": np.asarray(topology.x_point, dtype=float).tolist(),
            "seed_flux_span_wb": flux_span,
        },
        "commands": [],
        "figure": str(figure),
        "verdict": {"status": "running"},
    }
    _write(payload, output)
    terminal_states = []
    for command_m in COMMANDS_M:
        target_point = np.asarray([edge_radius + command_m, chord_height])
        pair, selection = _edge_pair(
            profile,
            free.state,
            point_rz_m=target_point,
            flux_span_wb=flux_span,
            requested_class=requested,
            target_current=target_current,
            prescribed_current=prescribed_current,
            program=free.program,
            circuits=circuits,
        )
        command_slot: dict[str, Any] = {}
        payload["commands"].append(command_slot)

        def persist_command(row: dict[str, Any]) -> None:
            command_slot.clear()
            command_slot.update(row)
            _write(payload, output)

        row, terminal = _command_row(
            profile,
            free,
            pair,
            command_m=command_m,
            edge_radius_m=edge_radius,
            chord_height_m=chord_height,
            requested_class=requested,
            target_current=target_current,
            prescribed_current=prescribed_current,
            persist=persist_command,
        )
        direction = np.asarray(pair.unknown.direction, dtype=float)[:, 0]

        def persist_reference(reference: dict[str, Any]) -> None:
            row["free_solve_reference"] = reference
            persist_command(row)

        reference = _free_reference(
            profile,
            free,
            constrained_current_a=row["total_compensating_current_a"],
            direction=direction,
            prescribed_current=prescribed_current,
            requested_class=requested,
            target_current=target_current,
            edge_radius_m=edge_radius,
            chord_height_m=chord_height,
            target_displacement_m=command_m,
            persist=persist_reference,
        )
        row["free_solve_reference"] = reference
        row["total_current_difference_percent"] = (
            None
            if reference["current_delta_a"] in (None, 0.0)
            else 100.0
            * (row["total_compensating_current_a"] - reference["current_delta_a"])
            / abs(reference["current_delta_a"])
        )
        row["selection"] = {
            "rule": selection.rule.name.lower(),
            "singular_values_row_scales_per_ampere": np.asarray(
                selection.singular_values, dtype=float
            ).tolist(),
            "direction_authority_row_scales_per_ampere": np.asarray(
                selection.direction_authority, dtype=float
            ).tolist(),
            "leading_circuits": [
                {"circuit": int(index), "family": names.get(int(index))}
                for index in selection.leading_circuits(0, count=5)
            ],
            "direction": [
                {
                    "circuit": int(index),
                    "family": names.get(int(index)),
                    "component": float(direction[index]),
                }
                for index in np.argsort(np.abs(direction))[::-1]
                if abs(float(direction[index])) > 1.0e-8
            ],
        }
        difference = row["total_current_difference_percent"]
        row["accepted"] = bool(
            row["status"] == "converged"
            and abs(row["terminal_position_error_mm"]) <= 1.0e3 * POSITION_TOLERANCE_M
            and row["merit_monotone_where_steps_are_nonzero"]
            and row["current_cap_respected"]
            and not row["accepted_overshoot"]
            and reference["reached"]
            and difference is not None
            and abs(difference) <= CURRENT_AGREEMENT_PERCENT
        )
        persist_command(row)
        terminal_states.append(terminal.state)
        _render(
            profile,
            free.state,
            terminal_states,
            payload["commands"],
            chord_height_m=chord_height,
            requested_class=requested,
            output=figure,
        )
    payload["verdict"] = {
        "status": (
            "pass"
            if all(command["accepted"] for command in payload["commands"])
            else "refusal"
        ),
        "commands_measured": len(payload["commands"]),
        "commands_accepted": sum(
            bool(command["accepted"]) for command in payload["commands"]
        ),
        "largest_abs_position_error_mm": max(
            abs(command["terminal_position_error_mm"])
            for command in payload["commands"]
        ),
        "largest_trip_application_a": max(
            command["largest_trip_application_a"] for command in payload["commands"]
        ),
        "any_accepted_overshoot": any(
            command["accepted_overshoot"] for command in payload["commands"]
        ),
        "refusals": [
            {
                "commanded_displacement_mm": command["commanded_displacement_mm"],
                **command["refusal"],
            }
            for command in payload["commands"]
            if command["refusal"] is not None
        ],
    }
    _write(payload, output)
    _write_report(payload, output, figure, report)
    return payload


def main() -> None:
    """Parse explicit destinations and execute the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    payload = measure(
        output=arguments.output.resolve(),
        figure=arguments.figure.resolve(),
        report=arguments.report.resolve(),
    )
    print(json.dumps(payload["verdict"], indent=2), flush=True)
    if payload["verdict"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
