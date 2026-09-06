"""Steer a bank equilibrium's shape through the bounding-box control rows.

One converged bank row is given the bounding-box rows read from its own
achieved boundary's turning points — boundary flux at the outer, upper,
inner and lower points, zero radial field at the outer and inner points,
zero vertical field at the upper and lower points, and the two-gradient
null row at the X-point — and is then re-solved twice from that equilibrium,
once with the upper turning point raised two centimetres and once with the
elongation raised five percent through the upper and lower points.  Each arm
reports the trips it took, the compensating current every driven circuit
carried, the achieved turning points against the commanded ones, and how far
the magnetic axis and the X-point moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import settled_mask_stall as settled
from nova.equilibrium.constraint import (
    BoundingBoxTarget,
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    FieldComponentConstraint,
    IsofluxConstraint,
    XPointConstraint,
    derive_circuit_compensators,
    sample_lattice_flux,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.reduced_newton import solve_constrained_reduced_newton
from nova.equilibrium.solve_request import resolve_forward_solve_policy
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = (
    ROOT / "docs/figures/constraint-augmented-newton-krylov/shape-control"
)
#: The bank row the shape rows are steered on.
ROW = (22086, 43)
#: Commanded moves: the upper turning point raised by two centimetres and the
#: elongation raised by five percent through the upper and lower points.
VERTICAL_STEP_M = 0.02
ELONGATION_FRACTION = 0.05
#: Budget the constrained arms are given. A commanded shape move opens rows
#: the public default budget was never sized for, so the arms state their own.
NEWTON_STEPS = 16
ACTIVE_SET_STEPS = 24
#: The outer active-set trip budget the steering measurement runs every arm
#: under. The Newton-Krylov settlement and stagnation stops are disabled for
#: the lifted arms so the bounded loop is the only limit beside a detected
#: active-set cycle.
STEERING_TRIP_BUDGET = 20
#: Upper-point commands the steering measurement scans, in metres.
UPPER_COMMANDS_M = (0.005, 0.010, 0.020)


def _strict_float(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _boundary_polygon(profile, flux, *, angles=181):
    """Ray-cast the achieved boundary contour outward from the magnetic axis."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    level = float(np.asarray(topology.boundary_flux))
    polarity = float(profile.operator.polarity)
    lattice = profile.lattice
    grid = jnp.reshape(jnp.asarray(flux)[: lattice.node_count], lattice.shape)
    reach = 0.5 * min(
        float(lattice.radius[-1] - lattice.radius[0]),
        float(lattice.height[-1] - lattice.height[0]),
    )
    theta = 2.0 * np.pi * np.arange(angles) / angles
    points = []
    for angle in theta:
        ray = np.asarray([np.cos(angle), np.sin(angle)])
        low, high = 0.0, reach
        for _step in range(48):
            middle = 0.5 * (low + high)
            value = float(
                np.asarray(
                    sample_lattice_flux(lattice, grid, jnp.asarray(axis + middle * ray))
                )
            )
            if polarity * (value - level) > 0.0:
                low = middle
            else:
                high = middle
        points.append(axis + 0.5 * (low + high) * ray)
    return np.asarray(points)


def _reference_point(profile, flux) -> np.ndarray:
    """Return the point whose flux the isoflux rows are measured against."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    saddle = np.asarray(topology.x_point, dtype=float)
    if np.all(np.isfinite(saddle)):
        return saddle
    return np.asarray(topology.wall_point, dtype=float)


def _circuit_count(profile) -> int:
    """Return the number of prescribed circuit columns the operator carries."""
    field = profile.operator.prescribed_current_field
    if field is None:
        raise RuntimeError("the persisted prescribed-current field is unavailable")
    return int(field.circuit_count)


def _row_scale(profile, span, kind) -> float:
    """Return the residual scale one bounding-box row kind is measured in."""
    if kind == "flux":
        return span
    if kind == "field":
        lattice = profile.lattice
        return span / (
            TOTAL_FLUX_FACTOR * float(lattice.radius[0]) * float(lattice.radial_step)
        )
    return span / float(profile.lattice.radial_step)


def _bounding_box_pairs(
    profile, target, *, span, ampere_scale=1.0e3
) -> tuple[ConstraintPair, ...]:
    """Assemble the isoflux, field and X-point pairs one target produces."""
    flux_points = np.asarray(target.flux_points, dtype=float)
    radial = np.asarray(target.radial_field_points, dtype=float)
    vertical = np.asarray(target.vertical_field_points, dtype=float)
    x_point = np.asarray(target.x_point, dtype=float)
    circuits = _circuit_count(profile)

    def pair(functional, points, kind):
        rows = functional.row_count
        scale = _row_scale(profile, span, kind)
        return ConstraintPair(
            functional=functional,
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, rows)).at[0].set(1.0),
                ampere_scale=jnp.full((rows,), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(rows),
                tolerance=jnp.full((rows,), 1.0e-6 * scale),
                scale=jnp.full((rows,), scale),
                initial_unknown=jnp.zeros(rows),
                payload=jnp.asarray(points, dtype=jnp.float64),
            ),
        )

    flux_payload = jnp.concatenate(
        (
            jnp.asarray(flux_points, dtype=jnp.float64),
            jnp.asarray(target.reference_point, dtype=jnp.float64)[None],
        )
    )
    pairs = [
        ConstraintPair(
            functional=IsofluxConstraint(
                point_count=flux_points.shape[0], reference="reference_point"
            ),
            unknown=CircuitCurrentUnknown(
                direction=jnp.zeros((circuits, flux_points.shape[0])).at[0].set(1.0),
                ampere_scale=jnp.full((flux_points.shape[0],), ampere_scale),
            ),
            binding=ConstraintBinding(
                target=jnp.zeros(flux_points.shape[0]),
                tolerance=jnp.full(
                    (flux_points.shape[0],), 1.0e-6 * _row_scale(profile, span, "flux")
                ),
                scale=jnp.full((flux_points.shape[0],), span),
                initial_unknown=jnp.zeros(flux_points.shape[0]),
                payload=flux_payload,
            ),
        ),
        pair(
            FieldComponentConstraint(
                components=("radial",) * radial.shape[0]
                + ("vertical",) * vertical.shape[0]
            ),
            np.concatenate((radial, vertical)),
            "field",
        ),
    ]
    if x_point.shape == (2,):
        pairs.append(pair(XPointConstraint(), x_point[None, :], "xpoint"))
    return tuple(pairs)


def _moved_target(target, *, upper_delta=0.0, lower_delta=0.0):
    """Return the same target with turning points moved by the stated deltas.

    The flux rows are ordered outer, upper, inner, lower and the vertical
    field rows upper, lower, so an upper command touches rows one and the
    first vertical-field row, a lower command rows three and the second.
    """
    flux = np.asarray(target.flux_points, dtype=float).copy()
    vertical = np.asarray(target.vertical_field_points, dtype=float).copy()
    flux[1] = flux[1] + np.asarray([0.0, upper_delta])
    flux[3] = flux[3] + np.asarray([0.0, lower_delta])
    vertical[0] = vertical[0] + np.asarray([0.0, upper_delta])
    vertical[1] = vertical[1] + np.asarray([0.0, lower_delta])
    return BoundingBoxTarget(
        flux_points=jnp.asarray(flux),
        radial_field_points=target.radial_field_points,
        vertical_field_points=jnp.asarray(vertical),
        x_point=target.x_point,
        reference_point=target.reference_point,
    )


def _achieved_turning_points(profile, flux):
    """Return the achieved boundary's own turning points as a target."""
    return BoundingBoxTarget.from_boundary(
        _boundary_polygon(profile, flux),
        x_point=None,
        reference_point=None,
    )


def _landmarks(profile, flux) -> dict[str, float | None]:
    """Return the axis and saddle positions one flux state reads."""
    _masks, topology = profile.operator.read(jnp.asarray(flux))
    axis = np.asarray(topology.axis, dtype=float)
    saddle = np.asarray(topology.x_point, dtype=float)
    return {
        "axis_radius_m": _strict_float(axis[0]),
        "axis_height_m": _strict_float(axis[1]),
        "x_point_radius_m": _strict_float(saddle[0]),
        "x_point_height_m": _strict_float(saddle[1]),
        "axis_flux_wb": _strict_float(topology.axis_flux),
        "boundary_flux_wb": _strict_float(topology.boundary_flux),
    }


def _turning_error(target, achieved):
    """Return each commanded turning-point row against the achieved extrema."""
    commanded = np.asarray(target.flux_points, dtype=float)
    reached = np.asarray(achieved.flux_points, dtype=float)
    return np.linalg.norm(reached - commanded, axis=1).tolist()


class _SolvedArmView(NamedTuple):
    """One route-normalised constrained solve result the record builder reads."""

    flux: Any
    constraints: tuple
    trips: int
    termination_reason: int
    converged: bool
    terminal_residual: float | None
    topology_consistent: bool | None = None
    prescribed_current: Any = None


def _solve_newton_krylov(
    profile,
    requested,
    target_current,
    *,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int | None = None,
    stop_on_active_set_stagnation: bool | None = None,
    stop_on_active_set_settlement: bool | None = None,
):
    """Return a solver that runs one arm through the branch Newton route."""

    def solve(derived, previous_flux):
        policy = resolve_forward_solve_policy(route="newton_krylov")
        branch_options = dict(policy.kernel_options())
        branch_options.update(
            newton_steps=newton_steps,
            active_set_steps=(
                STEERING_TRIP_BUDGET if active_set_steps is None else active_set_steps
            ),
        )
        if stop_on_active_set_stagnation is not None:
            branch_options["stop_on_active_set_stagnation"] = (
                stop_on_active_set_stagnation
            )
        if stop_on_active_set_settlement is not None:
            branch_options["stop_on_active_set_settlement"] = (
                stop_on_active_set_settlement
            )
        branch = profile.solve_branch(
            jnp.asarray(previous_flux),
            requested,
            target_current=target_current,
            constraint_pairs=derived,
            route="newton_krylov",
            **branch_options,
        )
        branch.equilibrium.flux.block_until_ready()
        equilibrium = branch.equilibrium
        return _SolvedArmView(
            flux=equilibrium.flux,
            constraints=equilibrium.constraints,
            trips=int(np.asarray(equilibrium.fixed_point.active_set_iterations)),
            termination_reason=int(
                np.asarray(equilibrium.fixed_point.termination_reason)
            ),
            converged=bool(np.asarray(branch.converged)),
            terminal_residual=_strict_float(branch.residual),
            topology_consistent=bool(np.asarray(branch.topology_consistent)),
        )

    return solve


def _solve_reduced_newton(
    profile,
    requested,
    target_current,
    *,
    newton_steps: int = NEWTON_STEPS,
    active_set_steps: int | None = None,
):
    """Return a solver that runs one arm through the constrained reduced route."""

    def solve(derived, previous_flux):
        result = solve_constrained_reduced_newton(
            profile,
            jnp.asarray(previous_flux),
            constraint_pairs=derived,
            requested_class=requested,
            target_current=target_current,
            newton_steps=newton_steps,
            active_set_steps=(
                STEERING_TRIP_BUDGET if active_set_steps is None else active_set_steps
            ),
        )
        return _SolvedArmView(
            flux=result.state,
            constraints=result.constraints,
            trips=int(np.asarray(result.active_set_iterations)),
            termination_reason=int(np.asarray(result.termination_reason)),
            converged=bool(np.asarray(result.converged)),
            terminal_residual=_strict_float(result.terminal_residual),
            prescribed_current=result.prescribed_current,
        )

    return solve


def _per_circuit_compensation(derived, constraints):
    """Return the per-circuit compensation every driven direction carried.

    The derived direction matrix keeps the full prescribed circuit row, so the
    compensated vector spans every circuit and only the drivable columns are
    nonzero.
    """
    total = np.zeros(np.asarray(derived[0].unknown.direction).shape[0])
    for pair, record in zip(derived, constraints, strict=True):
        direction = np.asarray(pair.unknown.direction)
        delta = direction @ np.asarray(record.physical_unknown)
        total = total + delta
    return total


def _movement_record(previous_points, commanded_points, achieved_points):
    """Return the previous-commanded-achieved triple for the outer rows."""
    previous = np.asarray(previous_points, dtype=float)
    commanded = np.asarray(commanded_points, dtype=float)
    achieved = np.asarray(achieved_points, dtype=float)
    movement = []
    for name, index in (("outer", 0), ("upper", 1), ("inner", 2), ("lower", 3)):
        displacement = commanded[index] - previous[index]
        realised = achieved[index] - previous[index]
        achieved_delta = achieved[index] - commanded[index]
        delta_norm = float(np.dot(displacement, displacement))
        movement.append(
            {
                "point": name,
                "previous_rz_m": [float(value) for value in previous[index]],
                "commanded_rz_m": [float(value) for value in commanded[index]],
                "achieved_rz_m": [float(value) for value in achieved[index]],
                "commanded_minus_previous": [float(value) for value in displacement],
                "achieved_minus_previous": [float(value) for value in realised],
                "achieved_minus_commanded": [float(value) for value in achieved_delta],
                "fraction_of_command": (
                    float(np.dot(realised, displacement) / (delta_norm + 1.0e-300))
                ),
            }
        )
    return movement


def _currents_payload(compensation, baseline, names, circuits, *, floor=1.0e-6):
    """Return the top compensating and applied per-circuit currents."""
    ordered = np.argsort(np.abs(compensation))[::-1]
    prominence = float(np.max(np.abs(compensation)))
    rows = []
    for index in ordered:
        if prominence > 0 and abs(float(compensation[index])) > floor * prominence:
            rows.append(
                {
                    "circuit": int(index),
                    "family": names.get(int(index)),
                    "compensating_a": float(compensation[index]),
                    "applied_total_a": float(
                        compensation[index] + float(baseline[index])
                    ),
                }
            )
    return rows


def _arm(
    name,
    profile,
    previous_flux,
    target,
    *,
    span,
    target_current,
    requested,
    circuits,
    names,
    solve=None,
    previous_points=None,
):
    """Solve one commanded shape move warm-started from the previous state."""
    pairs = _bounding_box_pairs(profile, target, span=span)
    derived, selection = derive_circuit_compensators(
        profile,
        pairs,
        jnp.asarray(previous_flux),
        requested_class=requested,
        target_current=target_current,
        circuits=circuits,
    )
    solver = (
        _solve_newton_krylov(profile, requested, target_current)
        if solve is None
        else solve
    )
    view = solver(derived, jnp.asarray(previous_flux))
    achieved = _achieved_turning_points(profile, view.flux)
    compensated = []
    for pair, record in zip(derived, view.constraints, strict=True):
        direction = np.asarray(pair.unknown.direction)
        delta = direction @ np.asarray(record.physical_unknown)
        compensated.append(
            {
                "row_kind": pair.functional.__class__.__name__,
                "current_a": [
                    {
                        "circuit": int(index),
                        "family": names.get(int(index)),
                        "current_a": float(delta[index]),
                    }
                    for index in np.argsort(np.abs(delta))[::-1][:6]
                    if abs(float(delta[index])) > 1.0e-6 * float(np.max(np.abs(delta)))
                ],
                "current_norm_a": float(np.linalg.norm(delta)),
            }
        )
    if previous_points is None:
        previous_points = np.asarray(
            _achieved_turning_points(profile, previous_flux).flux_points, dtype=float
        )
    baseline_current = np.asarray(
        profile.operator.prescribed_current_field.current, dtype=float
    )
    compensation = _per_circuit_compensation(derived, view.constraints)
    return {
        "arm": name,
        "commanded_turning_points": {
            "outer_m": [float(value) for value in np.asarray(target.flux_points)[0]],
            "upper_m": [float(value) for value in np.asarray(target.flux_points)[1]],
            "inner_m": [float(value) for value in np.asarray(target.flux_points)[2]],
            "lower_m": [float(value) for value in np.asarray(target.flux_points)[3]],
        },
        "selection": {
            "rule": selection.rule.name.lower(),
            "competing_rows": bool(selection.competing),
            "singular_values_row_scales_per_ampere": [
                float(value) for value in np.asarray(selection.singular_values)
            ],
            "leading_circuits_per_row": [
                [
                    {"circuit": int(index), "family": names.get(int(index))}
                    for index in selection.leading_circuits(row, count=3)
                ]
                for row in range(sum(int(pair.functional.row_count) for pair in pairs))
            ],
        },
        "qualified": bool(np.asarray(view.converged)),
        "topology_consistent": (
            None
            if view.topology_consistent is None
            else bool(np.asarray(view.topology_consistent))
        ),
        "terminal_residual": view.terminal_residual,
        "trips": view.trips,
        "termination": settled._termination_name(view.termination_reason),
        "rows_qualified": bool(
            np.all(
                [
                    bool(np.all(np.asarray(record.qualified)))
                    for record in view.constraints
                ]
            )
        ),
        "rows": [
            {
                "row_kind": pair.functional.__class__.__name__,
                "qualified": bool(np.all(np.asarray(record.qualified))),
                "max_relative_error": float(
                    np.max(np.abs(np.asarray(record.scaled_residual)))
                ),
                "physical_residual": [
                    float(value) for value in np.asarray(record.physical_residual)
                ],
            }
            for pair, record in zip(derived, view.constraints, strict=True)
        ],
        "compensated_circuits": compensated,
        "compensating_current_norm_a": float(
            np.linalg.norm(np.asarray([item["current_norm_a"] for item in compensated]))
        ),
        "currents_per_circuit": _currents_payload(
            compensation, baseline_current, names, circuits
        ),
        "movement_rz_m": _movement_record(
            np.asarray(previous_points),
            np.asarray(target.flux_points),
            np.asarray(achieved.flux_points),
        ),
        "achieved_turning_points": {
            "outer_m": [float(value) for value in np.asarray(achieved.flux_points)[0]],
            "upper_m": [float(value) for value in np.asarray(achieved.flux_points)[1]],
            "inner_m": [float(value) for value in np.asarray(achieved.flux_points)[2]],
            "lower_m": [float(value) for value in np.asarray(achieved.flux_points)[3]],
        },
        "turning_point_error_m": _turning_error(target, achieved),
        "landmarks": _landmarks(profile, view.flux),
        "achieved_boundary_rz_m": _boundary_polygon(profile, view.flux).tolist(),
    }


def _render(receipt, output: Path):
    """Overlay the commanded, achieved and previous boundary for both arms."""
    arms = receipt["arms"]
    figure, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 5.6))
    axes = np.atleast_1d(axes)
    previous = np.asarray(receipt["converged"]["boundary_rz_m"])
    anchor = np.asarray(receipt["converged"]["target"]["flux_points_rz_m"])
    for axis, arm in zip(axes, arms, strict=True):
        commanded = np.asarray(list(arm["commanded_turning_points"].values())).reshape(
            -1, 2
        )
        achieved = np.asarray(arm["achieved_boundary_rz_m"])
        axis.plot(
            previous[:, 0],
            previous[:, 1],
            color="0.55",
            linewidth=1.2,
            label="previous boundary",
        )
        axis.plot(
            achieved[:, 0],
            achieved[:, 1],
            color="tab:blue",
            linewidth=1.6,
            label="achieved boundary",
        )
        axis.plot(
            commanded[:, 0],
            commanded[:, 1],
            "o",
            color="tab:orange",
            markersize=6,
            label="commanded turning points",
        )
        axis.plot(
            anchor[:, 0],
            anchor[:, 1],
            "x",
            color="0.35",
            markersize=6,
            label="previous turning points",
        )
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.grid(alpha=0.2)
        axis.set_title(
            f"{arm['arm']}\n{arm['trips']} trips, "
            f"{arm['compensating_current_norm_a'] / 1.0e3:.2f} kA compensator"
        )
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        f"Bounding-box shape steering on {receipt['identity']}: "
        "commanded turning points against the achieved boundary",
        y=0.98,
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.86, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _steering_arm_specs(target, *, elongation_delta_m):
    """Return the upper-point scan and elongation arm commands."""
    return {
        "upper-5": (
            "upper turning point plus 5 mm",
            _moved_target(target, upper_delta=UPPER_COMMANDS_M[0]),
            "upper-scan",
            "5 mm",
            int(1.0e3 * UPPER_COMMANDS_M[0]),
        ),
        "upper-10": (
            "upper turning point plus 10 mm",
            _moved_target(target, upper_delta=UPPER_COMMANDS_M[1]),
            "upper-scan",
            "10 mm",
            int(1.0e3 * UPPER_COMMANDS_M[1]),
        ),
        "upper-20": (
            "upper turning point plus 20 mm",
            _moved_target(target, upper_delta=UPPER_COMMANDS_M[2]),
            "upper-scan",
            "20 mm",
            int(1.0e3 * UPPER_COMMANDS_M[2]),
        ),
        "elongation": (
            "elongation plus 5 percent through the upper and lower points",
            _moved_target(
                target, upper_delta=elongation_delta_m, lower_delta=-elongation_delta_m
            ),
            "elongation",
            "5 percent",
            int(1.0e3 * elongation_delta_m),
        ),
    }


def measure_steering_authority(
    *,
    directory: Path,
    cache_root: Path | None = None,
    routes: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
):
    """Measure how far commanded shape moves let the boundary travel.

    The two arm families run on two routes: the branch Newton-Krylov route
    with the active-set settlement and stagnation stops disabled under a
    twenty-trip budget, and the constrained reduced route under the same
    budget.  The upper-point arm runs at five, ten and twenty millimetres so
    the achieved motion and the compensating current can be read against the
    command; the elongation arm moves the upper and lower points through five
    percent of the boundary's own vertical span.  Every arm is persisted as it
    lands so a job split across several bounded invocations loses nothing.
    """
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
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
    selected_row, qualification = selected[ROW]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, response_cache
    )
    names = {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }
    circuits = sorted(names)
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    seed = jnp.asarray(passive_case["state"])
    span = float(passive_case["span_wb"])
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    identity = f"{ROW[0]}/{ROW[1]}"
    print(f"STEERING-AUTHORITY {identity} unconstrained solve", flush=True)
    base = profile.solve_branch(seed, requested, target_current=target_current)
    base.equilibrium.flux.block_until_ready()
    base_flux = base.equilibrium.flux
    boundary = _boundary_polygon(profile, base_flux)
    x_point = np.asarray(
        profile.operator.read(jnp.asarray(base_flux))[1].x_point, dtype=float
    )
    reference = _reference_point(profile, base_flux)
    target = BoundingBoxTarget.from_boundary(
        boundary, x_point=x_point, reference_point=reference
    )
    vertical_reference = float(np.asarray(target.flux_points)[1][1]) - float(
        np.asarray(target.flux_points)[3][1]
    )
    source = {
        "revision": _source_revision(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }
    configuration = {
        "constraint_policy": "imposed",
        "row_set": "boundary flux at the four bounding-box points, radial "
        "field at the outer and inner points, vertical field at the upper "
        "and lower points, two-gradient null at the X-point",
        "isoflux_reference": "reference_point at the read saddle",
        "compensating_direction": "derive_circuit_compensators restricted to "
        "the machine active mapping; the singular-distribution rule takes "
        "over from dominant authority when the rows compete",
        "drivable_circuits": [
            {"circuit": int(index), "family": names[index]} for index in circuits
        ],
        "prescribed_circuit_count": _circuit_count(profile),
        "row_tolerance_fraction": 1.0e-6,
        "flux_span_wb": span,
        "active_set_trip_budget": STEERING_TRIP_BUDGET,
        "inner_newton_steps": NEWTON_STEPS,
        "route_configuration": {
            "newton_krylov_lifted": {
                "route": "newton_krylov",
                "stop_on_active_set_stagnation": False,
                "stop_on_active_set_settlement": False,
            },
            "reduced": {
                "route": "solve_constrained_reduced_newton",
                "trip_boundary": "fused",
            },
        },
        "persistent_compilation_cache": {
            "directory": str(cache.directory),
            "version": cache.version_key,
        },
    }
    converged = {
        "qualified": bool(np.asarray(base.converged)),
        "topology_consistent": bool(np.asarray(base.topology_consistent)),
        "terminal_residual": _strict_float(base.residual),
        "trips": int(np.asarray(base.equilibrium.fixed_point.active_set_iterations)),
        "target": {
            "flux_points_rz_m": np.asarray(target.flux_points).tolist(),
            "x_point_rz_m": np.asarray(target.x_point).tolist(),
            "reference_point_rz_m": np.asarray(target.reference_point).tolist(),
        },
        "landmarks": _landmarks(profile, base_flux),
        "boundary_rz_m": boundary.tolist(),
    }
    baseline_current = np.asarray(
        profile.operator.prescribed_current_field.current, dtype=float
    )
    receipt = {
        "receipt": "bounding-box shape rows steered on one bank equilibrium",
        "identity": identity,
        "source": source,
        "configuration": configuration,
        "inputs": {"carrier_evidence": carrier_evidence},
        "converged": converged,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "converged.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    elongation_delta_m = 0.5 * ELONGATION_FRACTION * vertical_reference
    specar = _steering_arm_specs(target, elongation_delta_m=elongation_delta_m)
    selected_routes = (
        list(("newton_krylov_lifted", "reduced")) if routes is None else list(routes)
    )
    selected_arms = list(specar) if arms is None else list(arms)
    solvers = {}
    for route_name in selected_routes:
        if route_name == "newton_krylov_lifted":
            solvers[route_name] = _solve_newton_krylov(
                profile,
                requested,
                target_current,
                stop_on_active_set_stagnation=False,
                stop_on_active_set_settlement=False,
            )
        elif route_name == "reduced":
            solvers[route_name] = _solve_reduced_newton(
                profile, requested, target_current
            )
        else:
            raise ValueError(f"unknown steering route {route_name!r}")
    previous_points = np.asarray(target.flux_points, dtype=float)
    for route_name, solver in solvers.items():
        for arm_name in selected_arms:
            label, moved, family, command_label, millimetres = specar[arm_name]
            slug = f"{route_name}-{arm_name}"
            print(f"STEERING-AUTHORITY {identity} {slug}", flush=True)
            arm = _arm(
                label,
                profile,
                base_flux,
                moved,
                span=span,
                target_current=target_current,
                requested=requested,
                circuits=circuits,
                names=names,
                solve=solver,
                previous_points=previous_points,
            )
            payload = {
                "receipt": "steering authority arm record",
                "identity": identity,
                "route": route_name,
                "arm_family": family,
                "command_label": command_label,
                "command_millimetres": millimetres,
                "baseline_current_per_circuit_a": [
                    {
                        "circuit": int(index),
                        "family": names.get(int(index)),
                        "current_a": float(baseline_current[index]),
                    }
                    for index in circuits
                ],
                "arm": arm,
            }
            (directory / f"steering-{slug}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            print(
                "STEERING-AUTHORITY-ARM "
                + json.dumps(
                    {
                        "route": route_name,
                        "arm": arm_name,
                        "command_label": command_label,
                        "trips": arm["trips"],
                        "termination": arm["termination"],
                        "qualified": arm["qualified"],
                        "terminal_residual": arm["terminal_residual"],
                        "rows_qualified": arm["rows_qualified"],
                        "compensating_current_norm_a": arm[
                            "compensating_current_norm_a"
                        ],
                        "movement_rz_m": arm["movement_rz_m"],
                        "landmarks": arm["landmarks"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print("STEERING-AUTHORITY-ARMS-DONE", flush=True)
    return directory


def assemble_steering_authority(*, directory: Path):
    """Merge the per-route arm records into the shared steering receipt."""
    baseline = json.loads((directory / "converged.json").read_text(encoding="utf-8"))
    records = []
    for path in sorted(directory.glob("steering-*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise RuntimeError(f"no steering arm records under {directory}")
    first = records[0]
    receipt = {
        "receipt": "steering authority: how far commanded shape moves let the "
        "boundary travel, per route",
        "identity": first["identity"],
        "source": baseline["source"],
        "configuration": baseline["configuration"],
        "converged": baseline["converged"],
        "previous_turning_points_rz_m": baseline["converged"]["target"][
            "flux_points_rz_m"
        ],
        "arms": records,
    }
    (directory / "steering-authority.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    _render_steering_authority(receipt, directory / "steering-authority.png")
    print("STEERING-AUTHORITY-ASSEMBLED", flush=True)
    return receipt


def _render_steering_authority(receipt, output: Path):
    """Overlay previous, commanded and achieved boundaries per route and arm."""
    records = receipt["arms"]
    routes = sorted({record["route"] for record in records})
    families = sorted({record["arm_family"] for record in records})
    figure, axes = plt.subplots(
        len(routes), len(families), figsize=(6.4 * len(families), 6.2 * len(routes))
    )
    axes = np.atleast_2d(axes)
    previous = np.asarray(receipt["converged"]["boundary_rz_m"])
    anchor = np.asarray(receipt["converged"]["target"]["flux_points_rz_m"])
    for row, route in enumerate(routes):
        for column, family in enumerate(families):
            axis = axes[row][column]
            panel = [
                record
                for record in records
                if record["route"] == route and record["arm_family"] == family
            ]
            axis.plot(
                previous[:, 0],
                previous[:, 1],
                color="0.60",
                linewidth=1.2,
                label="previous boundary",
            )
            for record in panel:
                achieved = np.asarray(record["arm"]["achieved_boundary_rz_m"])
                axis.plot(
                    achieved[:, 0],
                    achieved[:, 1],
                    linewidth=1.5,
                    label=f"achieved, {record['command_label']}",
                )
                commanded = np.asarray(
                    list(record["arm"]["commanded_turning_points"].values())
                ).reshape(-1, 2)
                axis.plot(
                    commanded[:, 0],
                    commanded[:, 1],
                    "o",
                    markersize=5,
                    label=f"commanded {record['command_label']}",
                )
            axis.plot(
                anchor[:, 0],
                anchor[:, 1],
                "x",
                color="0.30",
                markersize=6,
                label="previous turning points",
            )
            axis.set_aspect("equal")
            axis.set_xlabel("R [m]")
            axis.set_ylabel("Z [m]")
            axis.grid(alpha=0.2)
            axis.set_title(f"{route}: {family}")
            axis.legend(frameon=False, fontsize=7, loc="lower right")
    figure.suptitle(
        f"Steering authority on {receipt['identity']}: "
        "commanded shape moves against the achieved boundary, per route",
        y=0.98,
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.90, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure(*, directory: Path, cache_root: Path | None = None):
    """Steer the bank row's shape through two commanded moves."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
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
    selected_row, qualification = selected[ROW]
    case, context = settled._mast_case_from_selection(
        settled.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, policy = settled._passive_inclusive_case(
        case, context, response_cache
    )
    names = {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }
    circuits = sorted(names)
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    seed = jnp.asarray(passive_case["state"])
    span = float(passive_case["span_wb"])
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    identity = f"{ROW[0]}/{ROW[1]}"
    print(f"SHAPE-CONTROL {identity} unconstrained solve", flush=True)
    base = profile.solve_branch(seed, requested, target_current=target_current)
    base.equilibrium.flux.block_until_ready()
    base_flux = base.equilibrium.flux
    boundary = _boundary_polygon(profile, base_flux)
    x_point = np.asarray(
        profile.operator.read(jnp.asarray(base_flux))[1].x_point, dtype=float
    )
    reference = _reference_point(profile, base_flux)
    target = BoundingBoxTarget.from_boundary(
        boundary, x_point=x_point, reference_point=reference
    )
    vertical_reference = float(np.asarray(target.flux_points)[1][1]) - float(
        np.asarray(target.flux_points)[3][1]
    )
    receipt = {
        "receipt": "bounding-box shape rows steered on one bank equilibrium",
        "identity": identity,
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "route": "ForwardProfile.solve_branch public defaults",
            "constraint_policy": "imposed",
            "row_set": "boundary flux at the four bounding-box points, radial "
            "field at the outer and inner points, vertical field at the upper "
            "and lower points, two-gradient null at the X-point",
            "isoflux_reference": "reference_point at the read saddle",
            "compensating_direction": "derive_circuit_compensators restricted to "
            "the machine active mapping; the singular-distribution rule takes "
            "over from dominant authority when the rows compete",
            "drivable_circuits": [
                {"circuit": int(index), "family": names[index]} for index in circuits
            ],
            "prescribed_circuit_count": _circuit_count(profile),
            "row_tolerance_fraction": 1.0e-6,
            "flux_span_wb": span,
            "persistent_compilation_cache": {
                "directory": str(cache.directory),
                "version": cache.version_key,
            },
        },
        "inputs": {"carrier_evidence": carrier_evidence},
        "converged": {
            "qualified": bool(np.asarray(base.converged)),
            "topology_consistent": bool(np.asarray(base.topology_consistent)),
            "terminal_residual": _strict_float(base.residual),
            "trips": int(
                np.asarray(base.equilibrium.fixed_point.active_set_iterations)
            ),
            "target": {
                "flux_points_rz_m": np.asarray(target.flux_points).tolist(),
                "x_point_rz_m": np.asarray(target.x_point).tolist(),
                "reference_point_rz_m": np.asarray(target.reference_point).tolist(),
            },
            "landmarks": _landmarks(profile, base_flux),
            "boundary_rz_m": boundary.tolist(),
        },
        "arms": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "converged.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    elongation_delta_m = 0.5 * ELONGATION_FRACTION * vertical_reference
    commands = (
        (
            "upper point plus 2 cm",
            "vertical-shift",
            _moved_target(target, upper_delta=VERTICAL_STEP_M),
        ),
        (
            "elongation plus 5 percent through the upper and lower points",
            "elongation-raise",
            _moved_target(
                target,
                upper_delta=elongation_delta_m,
                lower_delta=-elongation_delta_m,
            ),
        ),
    )
    for label, slug, moved in commands:
        print(f"SHAPE-CONTROL {identity} {slug}", flush=True)
        arm = _arm(
            label,
            profile,
            base_flux,
            moved,
            span=span,
            target_current=target_current,
            requested=requested,
            circuits=circuits,
            names=names,
        )
        receipt["arms"].append(arm)
        (directory / f"{slug}.json").write_text(
            json.dumps({**receipt, "arms": [arm]}, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "SHAPE-CONTROL-ARM "
            + json.dumps(
                {
                    key: value
                    for key, value in arm.items()
                    if key not in ("achieved_boundary_rz_m",)
                },
                sort_keys=True,
            ),
            flush=True,
        )
    (directory / "two-arms.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    _render(receipt, directory / "shape-steering.png")
    print("SHAPE-CONTROL-DONE", flush=True)
    return receipt


def main(argv=None):
    """Run the shape-steering receipt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("receipt", "steering", "assemble"),
        default="receipt",
        help="receipt: the committed two-arm receipt; steering: the lifted and "
        "reduced route arms; assemble: merge the landed arm records and render",
    )
    parser.add_argument(
        "--routes",
        default=None,
        help="comma-separated steering routes: newton_krylov_lifted,reduced",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="comma-separated steering arms: upper-5,upper-10,upper-20,elongation",
    )
    arguments = parser.parse_args(argv)
    if arguments.mode == "receipt":
        measure(directory=arguments.directory, cache_root=arguments.cache_root)
    elif arguments.mode == "steering":
        routes = None if arguments.routes is None else arguments.routes.split(",")
        arms = None if arguments.arms is None else arguments.arms.split(",")
        measure_steering_authority(
            directory=arguments.directory,
            cache_root=arguments.cache_root,
            routes=routes,
            arms=arms,
        )
    else:
        assemble_steering_authority(directory=arguments.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
