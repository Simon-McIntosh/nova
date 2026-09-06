"""Measure inverse-forward shape control through the playable solver on MAST."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.shape_inverse import (
    achieved_target,
    shape_response_matrix,
    shape_row_target,
    shape_values,
    solve_shape_inverse,
)
from nova.equilibrium.topology import NoQualifiedAxisError
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

from apps.playable.production import ForwardMachine, ProductionSolver

ROOT = Path(__file__).resolve().parents[1]
TARGET = (22086, 43)
DEFAULT_DIRECTORY = ROOT / "docs/figures/playable-forward-solve/shape-inverse"
NEGATIVE_CONTROL = "all-prescribed-negative-control.json"
CONSISTENCY_DIAGNOSTIC = "seed-consistency-diagnostic.json"


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _points(target) -> np.ndarray:
    """Return one target's principal turning points as a host array."""
    return np.asarray(target.flux_points, dtype=float)


def _target_with_points(target, points: np.ndarray):
    """Return a target whose field rows follow its moved turning points."""
    return replace(
        target,
        flux_points=points,
        radial_field_points=points[[0, 2]],
        vertical_field_points=points[[1, 3]],
    )


def _upper_point_target(target):
    """Raise only the upper turning point by two centimetres."""
    points = _points(target).copy()
    points[1, 1] += 0.02
    return _target_with_points(target, points)


def _elongation_target(target):
    """Increase the upper-to-lower height by five percent about its centre."""
    points = _points(target).copy()
    centre = 0.5 * (points[1, 1] + points[3, 1])
    points[[1, 3], 1] = centre + 1.05 * (points[[1, 3], 1] - centre)
    return _target_with_points(target, points)


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Persist one complete arm as soon as it lands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _circuit_names(policy: dict[str, Any]) -> dict[int, str]:
    """Return active-family names keyed by zero-based response column."""
    return {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }


def _circuit_label(index: int, names: dict[int, str]) -> str:
    """Return a stable circuit label, including an active family when known."""
    family = names.get(index)
    return f"circuit_{index:02d}" if family is None else f"circuit_{index:02d}_{family}"


def _null_modes(inverse, names: dict[int, str]) -> list[dict[str, Any]]:
    """Describe every circuit combination with no authority on the active rows."""
    response = inverse.response[:, inverse.free_circuits]
    zero_singular_modes = inverse.singular_values.size - inverse.numerical_rank
    modes = []
    for index, weights in enumerate(inverse.right_null_space):
        largest_weight = float(np.max(np.abs(weights)))
        dominant = [
            {
                "circuit": int(circuit),
                "family": names[int(circuit)],
                "weight": float(weight),
            }
            for circuit, weight in zip(inverse.free_circuits, weights, strict=True)
            if abs(float(weight)) >= 0.25 * largest_weight
        ]
        modes.append(
            {
                "index": index,
                "origin": (
                    "zero_singular_value"
                    if index < zero_singular_modes
                    else "rectangular_right_null_space"
                ),
                "response_l2": float(np.linalg.norm(response @ weights)),
                "dominant_combination": dominant,
                "circuit_weights": {
                    _circuit_label(int(circuit), names): float(weight)
                    for circuit, weight in zip(
                        inverse.free_circuits, weights, strict=True
                    )
                },
            }
        )
    return modes


def _row_names(target) -> list[str]:
    """Name the absolute flux and field rows in their assembled order."""
    names = [
        "psi_outer",
        "psi_upper",
        "psi_inner",
        "psi_lower",
        "br_outer",
        "br_inner",
        "bz_upper",
        "bz_lower",
    ]
    if target.x_point is not None and np.shape(target.x_point) == (2,):
        names.extend(("br_x_point", "bz_x_point"))
    return names


def _current_comparison(
    inverse,
    seed: np.ndarray,
    circuit_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Report solved total currents and changes beside the carrier seed."""
    rows = []
    for circuit, total, delta in zip(
        inverse.free_circuits,
        inverse.currents[inverse.free_circuits],
        inverse.delta,
        strict=True,
    ):
        seed_current = float(seed[circuit])
        absolute_change = abs(float(delta))
        rows.append(
            {
                "circuit": int(circuit),
                "family": circuit_names[int(circuit)],
                "seed_current_a": seed_current,
                "solved_total_current_a": float(total),
                "current_change_a": float(delta),
                "absolute_change_a": absolute_change,
                "absolute_change_ka": absolute_change / 1000.0,
                "change_fraction_of_seed": (
                    absolute_change / abs(seed_current) if seed_current != 0.0 else None
                ),
            }
        )
    return rows


def _linear_closure(
    inverse, row_floors: np.ndarray | None = None
) -> list[dict[str, Any]]:
    """Report every final linear row against the assembled right-hand side."""
    names = _row_names_from_kinds(inverse.row_kinds)
    closure = inverse.linear_prediction - inverse.right_hand_side
    if row_floors is None:
        row_floors = np.zeros_like(closure)
    return [
        {
            "row": name,
            "unit": "Wb" if kind == "flux" else "T",
            "linear_prediction": float(prediction),
            "right_hand_side": float(target),
            "residual": float(residual),
            "consistency_floor": float(floor),
            "closure_fraction": float(
                max(0.0, 1.0 - abs(residual) / max(abs(target), floor, 1.0e-15))
            ),
            "closes_at_least_eighty_percent": bool(
                abs(residual) <= max(0.2 * abs(target), floor)
            ),
        }
        for name, kind, prediction, target, residual, floor in zip(
            names,
            inverse.row_kinds,
            inverse.linear_prediction,
            inverse.right_hand_side,
            closure,
            row_floors,
            strict=True,
        )
    ]


def _row_names_from_kinds(kinds: tuple[str, ...]) -> list[str]:
    """Name a standard eight- or ten-row shape block."""
    names = [
        "psi_outer",
        "psi_upper",
        "psi_inner",
        "psi_lower",
        "br_outer",
        "br_inner",
        "bz_upper",
        "bz_lower",
        "br_x_point",
        "bz_x_point",
    ]
    if len(names) != len(kinds):
        names = [f"{kind}_{index}" for index, kind in enumerate(kinds)]
    return names[: len(kinds)]


def _seed_consistency_diagnostic(
    directory: Path,
    machine: ForwardMachine,
    prime,
    previous_target,
    circuit_names: dict[int, str],
) -> dict[str, Any]:
    """Persist the seed rows, null inverse and admitted command diagnostic."""
    path = directory / CONSISTENCY_DIAGNOSTIC
    profile = machine.profile
    seed = np.asarray(profile.operator.prescribed_current_field.current, dtype=float)
    free = np.asarray(machine.drivable_circuits, dtype=int)
    values = shape_values(profile, previous_target, prime.flux)
    targets = shape_row_target(profile, previous_target, prime.flux)
    response = shape_response_matrix(profile, previous_target, prime.flux)
    active_image = response[:, free] @ seed[free]
    fixed_and_plasma = values - active_image
    assembled = fixed_and_plasma + active_image
    residual = assembled - targets
    kinds = ("flux",) * 4 + ("field",) * (targets.size - 4)
    characteristic_scale = np.maximum.reduce(
        (np.abs(targets), np.abs(active_image), np.abs(fixed_and_plasma))
    )
    relative_residual = np.abs(residual) / characteristic_scale
    relative_tolerance = 1.0e-2
    tolerances = relative_tolerance * characteristic_scale
    rows = [
        {
            "row": name,
            "unit": "Wb" if kind == "flux" else "T",
            "assembled_value": float(value),
            "assembled_target": float(target),
            "residual": float(error),
            "active_circuit_contribution": float(active),
            "fixed_circuit_and_plasma_contribution": float(base),
            "characteristic_row_scale": float(scale),
            "relative_residual": float(relative_error),
            "absolute_tolerance": float(tolerance),
            "relative_tolerance": relative_tolerance,
            "passes": bool(relative_error <= relative_tolerance),
        }
        for (
            name,
            kind,
            value,
            target,
            error,
            active,
            base,
            scale,
            relative_error,
            tolerance,
        ) in zip(
            _row_names(previous_target),
            kinds,
            assembled,
            targets,
            residual,
            active_image,
            fixed_and_plasma,
            characteristic_scale,
            relative_residual,
            tolerances,
            strict=True,
        )
    ]
    payload: dict[str, Any] = {
        "source_commit": _source_revision(),
        "machine": machine.identity,
        "policy": {
            "active_circuit_count": int(free.size),
            "field_weight": 50.0,
            "picard_rounds": 3,
            "h200_forward_arms_admitted": False,
        },
        "previous_turning_points_m": _points(previous_target).tolist(),
        "check_1_seed_rows": {
            "description": (
                "Absolute shape rows evaluated at the seed total currents and "
                "the previous equilibrium turning points."
            ),
            "rows": rows,
            "maximum_absolute_flux_residual_wb": float(np.max(np.abs(residual[:4]))),
            "maximum_absolute_field_residual_t": float(np.max(np.abs(residual[4:]))),
            "maximum_relative_residual": float(np.max(relative_residual)),
            "consistency_floor": {
                "relative_to": (
                    "largest of the target, active-circuit image and "
                    "fixed-circuit-plus-plasma image"
                ),
                "maximum_relative_residual": float(np.max(relative_residual)),
                "admission_tolerance": relative_tolerance,
                "interpretation": (
                    "Boundary-extraction precision, not an inverse-system "
                    "assembly error."
                ),
            },
            "passes": bool(np.all(relative_residual <= relative_tolerance)),
        },
        "check_2_null_inverse": {"status": "not_run"},
        "check_3_command_gamma_sweep": {"status": "not_run"},
    }
    _write(path, payload)

    null_inverse = solve_shape_inverse(
        profile,
        previous_target,
        prime.flux,
        prescribed_current=seed,
        free_circuits=free,
    )
    current_rows = _current_comparison(null_inverse, seed, circuit_names)
    fractions = [
        row["change_fraction_of_seed"]
        for row in current_rows
        if row["change_fraction_of_seed"] is not None
    ]
    zero_seed_changes = [
        row["absolute_change_a"]
        for row in current_rows
        if row["change_fraction_of_seed"] is None
    ]
    current_passes = all(fraction <= 0.05 for fraction in fractions) and all(
        change <= 1.0 for change in zero_seed_changes
    )
    boundary = np.asarray(null_inverse.picard_boundary_flux)
    boundary_excursion = float(np.max(np.abs(boundary - boundary[0])))
    boundary_passes = boundary_excursion <= 1.0e-6
    payload["check_2_null_inverse"] = {
        "status": "complete",
        "total_current_by_circuit": current_rows,
        "total_current_by_round_a": null_inverse.picard_currents.tolist(),
        "boundary_flux_target_wb": float(null_inverse.target[0]),
        "boundary_flux_by_round_wb": boundary.tolist(),
        "maximum_boundary_flux_excursion_wb": boundary_excursion,
        "current_change_l2_a": float(np.linalg.norm(null_inverse.delta)),
        "maximum_absolute_current_change_a": float(np.max(np.abs(null_inverse.delta))),
        "maximum_change_fraction_of_nonzero_seed": float(max(fractions)),
        "current_within_five_percent": bool(current_passes),
        "boundary_stable_within_1e-6_wb": bool(boundary_passes),
        "passes": bool(current_passes and boundary_passes),
        "linear_row_closure": _linear_closure(null_inverse),
    }
    _write(path, payload)

    gamma_factors = (1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9)
    maximum_change_a = 20_000.0
    minimum_row_closure = 0.8
    command_results = []
    commands = (
        ("upper-point-plus-20mm", _upper_point_target(previous_target)),
        ("elongation-plus-5pct", _elongation_target(previous_target)),
    )
    for command_name, command in commands:
        sweep = []
        selected_gamma = None
        for gamma_factor in gamma_factors:
            entry: dict[str, Any] = {"gamma_factor_per_ampere": gamma_factor}
            try:
                inverse = solve_shape_inverse(
                    profile,
                    command,
                    prime.flux,
                    prescribed_current=seed,
                    free_circuits=free,
                    gamma=gamma_factor,
                )
            except NoQualifiedAxisError as error:
                entry.update(
                    {
                        "status": "axis_lost_during_picard",
                        "error": str(error),
                        "placement_result": None,
                        "admitted": False,
                    }
                )
            else:
                closure = inverse.linear_prediction - inverse.right_hand_side
                weighted_closure = closure.copy()
                weighted_closure[4:] *= np.sqrt(inverse.field_weight)
                row_closure = _linear_closure(inverse, tolerances)
                max_change = float(np.max(np.abs(inverse.delta)))
                all_rows_close = all(
                    row["closes_at_least_eighty_percent"] for row in row_closure
                )
                admitted = max_change <= maximum_change_a and all_rows_close
                if admitted and selected_gamma is None:
                    selected_gamma = gamma_factor
                entry.update(
                    {
                        "status": "complete",
                        "admitted": admitted,
                        "placement_result": {
                            "tikhonov_gamma": inverse.gamma,
                            "total_current_by_circuit": _current_comparison(
                                inverse, seed, circuit_names
                            ),
                            "current_change_l2_a": float(np.linalg.norm(inverse.delta)),
                            "maximum_absolute_current_change_a": max_change,
                            "maximum_absolute_current_change_ka": max_change / 1000.0,
                            "linear_row_closure": row_closure,
                            "every_row_closes_at_least_eighty_percent": (
                                all_rows_close
                            ),
                            "linear_row_closure_l2_mixed": float(
                                np.linalg.norm(closure)
                            ),
                            "weighted_linear_row_closure_l2_mixed": float(
                                np.linalg.norm(weighted_closure)
                            ),
                            "boundary_flux_by_round_wb": (
                                inverse.picard_boundary_flux.tolist()
                            ),
                        },
                    }
                )
            sweep.append(entry)
            partial = {
                "command": command_name,
                "commanded_turning_points_m": _points(command).tolist(),
                "selected_gamma_factor_per_ampere": selected_gamma,
                "gamma_sweep": sweep,
            }
            payload["check_3_command_gamma_sweep"] = {
                "status": "running",
                "commands": command_results + [partial],
            }
            _write(path, payload)
        command_results.append(partial)
    admission_policy = {
        "maximum_absolute_current_change_a": maximum_change_a,
        "minimum_linear_row_closure_fraction": minimum_row_closure,
        "row_residual_allowance": (
            "twenty percent of the delta-row target or the seed consistency "
            "floor, whichever is larger"
        ),
        "selection": "smallest admitted gamma per command",
    }
    all_commands_admitted = all(
        command["selected_gamma_factor_per_ampere"] is not None
        for command in command_results
    )
    payload["check_3_command_gamma_sweep"] = {
        "status": "complete",
        "admission_policy": admission_policy,
        "commands": command_results,
        "all_commands_admitted": all_commands_admitted,
    }
    payload["policy"]["h200_forward_arms_admitted"] = all_commands_admitted
    payload["outcome"] = "cpu_consistency_table_complete"
    _write(path, payload)
    return payload


def _command_diagnostic(
    name: str,
    machine: ForwardMachine,
    previous,
    target,
    circuit_names: dict[int, str],
) -> dict[str, Any]:
    """Return one seed-anchored current command without solving it forward."""
    seed = np.asarray(machine.profile.operator.prescribed_current_field.current)
    inverse = solve_shape_inverse(
        machine.profile,
        target,
        previous.flux,
        prescribed_current=seed,
        free_circuits=machine.drivable_circuits,
    )
    fractions = np.abs(inverse.delta) / np.abs(seed[inverse.free_circuits])
    block = inverse.response[:, inverse.free_circuits]
    flux_rows = sum(kind == "flux" for kind in inverse.row_kinds)
    return {
        "command": name,
        "previous_turning_points_m": _points(
            achieved_target(machine.profile, previous.flux)
        ).tolist(),
        "commanded_turning_points_m": _points(target).tolist(),
        "formulation": {
            "unknown": "active-circuit current change about the fixed seed",
            "field_weight": inverse.field_weight,
            "tikhonov_gamma": inverse.gamma,
            "placement_picard_rounds": int(inverse.picard_currents.shape[0] - 1),
            "nonlinear_solve_inside_inverse": False,
            "current_step_guard": inverse.current_step_fraction,
        },
        "previous_variant_current_change_l2_a": 1146131.0,
        "seed_current_by_family_a": {
            _circuit_label(int(circuit), circuit_names): float(seed[circuit])
            for circuit in inverse.free_circuits
        },
        "current_change_by_family_a": {
            _circuit_label(int(circuit), circuit_names): float(delta)
            for circuit, delta in zip(inverse.free_circuits, inverse.delta, strict=True)
        },
        "change_fraction_of_seed_by_family": {
            _circuit_label(int(circuit), circuit_names): float(fraction)
            for circuit, fraction in zip(inverse.free_circuits, fractions, strict=True)
        },
        "current_change_l2_a": float(np.linalg.norm(inverse.delta)),
        "largest_change_fraction_of_seed": float(np.max(fractions)),
        "linear_row_prediction": inverse.linear_prediction.tolist(),
        "linear_row_right_hand_side": inverse.right_hand_side.tolist(),
        "picard_total_current_by_round_a": inverse.picard_currents.tolist(),
        "picard_boundary_flux_by_round_wb": inverse.picard_boundary_flux.tolist(),
        "weighted_response_singular_values_mixed": inverse.singular_values.tolist(),
        "flux_response_singular_values_wb_per_a": np.linalg.svd(
            block[:flux_rows], compute_uv=False
        ).tolist(),
        "field_response_singular_values_t_per_a": np.linalg.svd(
            block[flux_rows:], compute_uv=False
        ).tolist(),
        "row_units": ["Wb" if kind == "flux" else "T" for kind in inverse.row_kinds],
        "response_numerical_rank": inverse.numerical_rank,
        "null_modes": _null_modes(inverse, circuit_names),
    }


def _arm_receipt(
    name: str,
    machine: ForwardMachine,
    previous,
    target,
    null_points: np.ndarray,
    circuit_names: dict[int, str],
    gamma_factor: float,
) -> tuple[dict[str, Any], object]:
    """Run one arm through ``ProductionSolver`` and return its receipt."""
    solver = ProductionSolver(machine, inverse_gamma=gamma_factor)
    prior = achieved_target(machine.profile, previous.flux)
    equilibrium, _program = solver.solve_target(previous, target)
    achieved = achieved_target(machine.profile, equilibrium.flux)
    rounds = [
        {
            "index": index,
            "coil_current_a": item.inverse.currents.tolist(),
            "coil_delta_a": item.inverse.delta.tolist(),
            "uncapped_coil_delta_a": item.inverse.uncapped_delta.tolist(),
            "free_circuit_current_by_family_a": {
                _circuit_label(int(circuit), circuit_names): float(
                    item.inverse.currents[circuit]
                )
                for circuit in item.inverse.free_circuits
            },
            "current_change_by_family_a": {
                _circuit_label(int(circuit), circuit_names): float(delta)
                for circuit, delta in zip(
                    item.inverse.free_circuits, item.inverse.delta, strict=True
                )
            },
            "current_change_fraction_of_seed_by_family": {
                _circuit_label(int(circuit), circuit_names): float(
                    abs(delta) / abs(solver.reference_current[circuit])
                )
                for circuit, delta in zip(
                    item.inverse.free_circuits, item.inverse.delta, strict=True
                )
            },
            "current_change_l2_a": float(np.linalg.norm(item.inverse.delta)),
            "uncapped_current_change_l2_a": float(
                np.linalg.norm(item.inverse.uncapped_delta)
            ),
            "current_step_fraction": item.inverse.current_step_fraction,
            "current_step_limited": item.inverse.current_step_limited,
            "linear_row_prediction": item.inverse.linear_prediction.tolist(),
            "uncapped_linear_row_prediction": (
                item.inverse.response[:, item.inverse.free_circuits]
                @ item.inverse.uncapped_delta
            ).tolist(),
            "linear_row_right_hand_side": item.inverse.right_hand_side.tolist(),
            "least_squares_residual": item.inverse.least_squares_residual,
            "uncapped_least_squares_residual": (
                item.inverse.uncapped_least_squares_residual
            ),
            "response_singular_values": item.inverse.singular_values.tolist(),
            "response_numerical_rank": item.inverse.numerical_rank,
            "response_conditioning_span": float(
                item.inverse.singular_values[0]
                / item.inverse.singular_values[item.inverse.numerical_rank - 1]
            ),
            "null_modes": _null_modes(item.inverse, circuit_names),
            "placement_picard_total_currents_a": (
                item.inverse.picard_currents.tolist()
            ),
            "placement_picard_boundary_flux_wb": (
                item.inverse.picard_boundary_flux.tolist()
            ),
            "commanded_turning_points_m": _points(target).tolist(),
            "achieved_turning_points_m": item.achieved_turning_points.tolist(),
            "turning_point_error_m": item.turning_point_error,
            "trips": item.trips,
            "wall_s": item.wall,
        }
        for index, item in enumerate(solver.last_rounds, start=1)
    ]
    payload = {
        "arm": name,
        "previous_turning_points_m": _points(prior).tolist(),
        "commanded_turning_points_m": _points(target).tolist(),
        "achieved_turning_points_m": _points(achieved).tolist(),
        "null_turning_points_m": null_points.tolist(),
        "relative_turning_point_motion_m": (_points(achieved) - null_points).tolist(),
        "coil_current_by_circuit_a": {
            _circuit_label(index, circuit_names): float(current)
            for index, current in enumerate(solver.prescribed_current)
        },
        "rounds": rounds,
        "round_count": len(rounds),
        "current_step_policy": {
            "reference": "fixed carrier seed current per circuit",
            "gamma_factor_per_ampere": gamma_factor,
            "maximum_fraction_per_round": solver.current_step_fraction,
            "placement_picard_rounds": 3,
            "nonlinear_forward_solves": 1,
        },
        "total_wall_s": float(sum(item["wall_s"] for item in rounds)),
        "total_trips": int(sum(item["trips"] for item in rounds)),
        "converged": bool(np.asarray(equilibrium.fixed_point.converged)),
        "final_turning_point_error_m": rounds[-1]["turning_point_error_m"],
    }
    return payload, achieved


def _null_receipt(
    machine: ForwardMachine,
    previous,
    circuit_names: dict[int, str],
) -> tuple[dict[str, Any], object]:
    """Re-solve unchanged currents and return the physical motion baseline."""
    solver = ProductionSolver(machine)
    started = perf_counter()
    equilibrium, trips, _program = solver._forward(
        machine.profile, previous.flux, solver.prescribed_current
    )
    wall = perf_counter() - started
    prior = achieved_target(machine.profile, previous.flux)
    achieved = achieved_target(machine.profile, equilibrium.flux)
    payload = {
        "arm": "null-resolve",
        "previous_turning_points_m": _points(prior).tolist(),
        "achieved_turning_points_m": _points(achieved).tolist(),
        "turning_point_drift_m": (_points(achieved) - _points(prior)).tolist(),
        "coil_current_by_circuit_a": {
            _circuit_label(index, circuit_names): float(current)
            for index, current in enumerate(solver.prescribed_current)
        },
        "trips": int(trips),
        "wall_s": float(wall),
        "converged": bool(np.asarray(equilibrium.fixed_point.converged)),
    }
    return payload, equilibrium


def _draw(arms: list[dict[str, Any]], path: Path) -> None:
    """Plot commanded and achieved shape beside the circuit-current response."""
    figure, axes = plt.subplots(len(arms), 2, figsize=(11, 4.5 * len(arms)))
    axes = np.atleast_2d(axes)
    for row, arm in enumerate(arms):
        previous = np.asarray(arm["previous_turning_points_m"])
        commanded = np.asarray(arm["commanded_turning_points_m"])
        achieved = np.asarray(arm["achieved_turning_points_m"])
        shape_axis = axes[row, 0]
        shape_axis.scatter(previous[:, 0], previous[:, 1], marker="o", label="previous")
        shape_axis.scatter(
            commanded[:, 0], commanded[:, 1], marker="x", label="commanded"
        )
        shape_axis.scatter(achieved[:, 0], achieved[:, 1], marker="+", label="achieved")
        for before, command, after in zip(previous, commanded, achieved, strict=True):
            shape_axis.plot(
                [before[0], command[0]], [before[1], command[1]], color="0.75"
            )
            shape_axis.plot(
                [command[0], after[0]],
                [command[1], after[1]],
                color="tab:red",
                linestyle=":",
            )
        shape_axis.set_aspect("equal")
        shape_axis.set_title(
            f"{arm['arm']}: error {1000 * arm['final_turning_point_error_m']:.2f} mm"
        )
        shape_axis.set_xlabel("R / m")
        shape_axis.set_ylabel("Z / m")
        shape_axis.legend()

        currents = arm["coil_current_by_circuit_a"]
        current_axis = axes[row, 1]
        current_axis.bar(range(len(currents)), list(currents.values()))
        current_axis.set_title(
            f"{arm['round_count']} rounds, {arm['total_trips']} trips, "
            f"{arm['total_wall_s']:.3f} s"
        )
        current_axis.set_xlabel("circuit index")
        current_axis.set_ylabel("current / A")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def measure(
    directory: Path = DEFAULT_DIRECTORY,
    *,
    diagnose_inverse: bool = False,
    diagnose_consistency: bool = False,
) -> dict[str, Any]:
    """Run both commanded-shape arms on MAST 22086/43."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    selected_row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    circuit_names = _circuit_names(policy)
    machine = ForwardMachine(
        profile=profile,
        seed=jnp.asarray(passive_case["state"]),
        wall=np.asarray(profile.operator.wall.coordinate),
        identity="mast-22086/43",
        # The response carrier stores the active PF circuits first, followed
        # by passive and vessel currents that are state, not shape actuators.
        drivable_circuits=tuple(range(int(policy["active_circuit_count"]))),
    )
    prime_solver = ProductionSolver(machine)
    prime, _trips, _program = prime_solver._forward(
        profile, machine.seed, prime_solver.prescribed_current
    )
    previous_target = achieved_target(profile, prime.flux)
    if diagnose_consistency:
        payload = _seed_consistency_diagnostic(
            directory, machine, prime, previous_target, circuit_names
        )
        print("SEED-CONSISTENCY " + json.dumps(payload), flush=True)
        return payload
    if diagnose_inverse:
        diagnostics = [
            _command_diagnostic(name, machine, prime, target, circuit_names)
            for name, target in (
                ("upper-point-plus-20mm", _upper_point_target(previous_target)),
                ("elongation-plus-5pct", _elongation_target(previous_target)),
            )
        ]
        payload = {
            "source_commit": _source_revision(),
            "machine": machine.identity,
            "commands": diagnostics,
        }
        _write(directory / "pulse-design-inverse-diagnostic.json", payload)
        print("INVERSE-DIAGNOSTIC " + json.dumps(payload), flush=True)
        return payload
    null_arm, null_equilibrium = _null_receipt(machine, prime, circuit_names)
    null_points = _points(achieved_target(profile, null_equilibrium.flux))
    consistency_path = directory / CONSISTENCY_DIAGNOSTIC
    consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    gamma_by_command = {
        item["command"]: item["selected_gamma_factor_per_ampere"]
        for item in consistency["check_3_command_gamma_sweep"]["commands"]
    }
    if any(value is None for value in gamma_by_command.values()):
        raise ValueError("every H200 command must have an admitted CPU gamma")
    definitions = (
        (
            "upper-point-plus-20mm",
            _upper_point_target(previous_target),
            float(gamma_by_command["upper-point-plus-20mm"]),
        ),
        (
            "elongation-plus-5pct",
            _elongation_target(previous_target),
            float(gamma_by_command["elongation-plus-5pct"]),
        ),
    )
    runtime = {
        "source_commit": _source_revision(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "scheduler": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        },
        "carrier": carrier_evidence,
        "policy": policy,
        "inverse_gamma_factor_by_command": gamma_by_command,
        "active_circuits": [
            {
                "index": index,
                "family": circuit_names[index],
            }
            for index in sorted(circuit_names)
        ],
        "compilation_cache": cache.receipt(),
        "prime": {
            "converged": bool(np.asarray(prime.fixed_point.converged)),
            "trips": int(prime.fixed_point.active_set_iterations),
        },
    }
    arms = []
    null_arm["runtime"] = runtime
    _write(directory / "null-resolve.json", null_arm)
    for name, target, gamma_factor in definitions:
        arm, _achieved = _arm_receipt(
            name,
            machine,
            prime,
            target,
            null_points,
            circuit_names,
            gamma_factor,
        )
        arm["runtime"] = runtime
        arms.append(arm)
        _write(directory / f"{name}.json", arm)
        print(
            f"ARM-DONE {name} error_mm="
            f"{1000 * arm['final_turning_point_error_m']:.6g} "
            f"rounds={arm['round_count']} trips={arm['total_trips']} "
            f"wall_s={arm['total_wall_s']:.6g}",
            flush=True,
        )
    negative_control_path = directory / NEGATIVE_CONTROL
    if not negative_control_path.exists():
        raise FileNotFoundError(
            "the all-prescribed negative control must be retained beside the receipt"
        )
    negative_control = json.loads(negative_control_path.read_text(encoding="utf-8"))
    receipt = {
        "runtime": runtime,
        "null_arm": null_arm,
        "arms": arms,
        "active_response_block": arms[0]["rounds"][0],
        "all_prescribed_negative_control": {
            "interpretation": (
                "Driving passive and vessel circuits as shape actuators is an "
                "unphysical negative control. Its large errors are retained as "
                "evidence rather than used to judge active-circuit authority."
            ),
            "source_path": NEGATIVE_CONTROL,
            "receipt": negative_control,
        },
    }
    _write(directory / "shape-inverse-receipt.json", receipt)
    _draw(arms, directory / "shape-inverse-receipt.png")
    return receipt


def main() -> None:
    """Run the receipt from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--diagnose-inverse", action="store_true")
    parser.add_argument("--diagnose-consistency", action="store_true")
    arguments = parser.parse_args()
    measure(
        arguments.directory,
        diagnose_inverse=arguments.diagnose_inverse,
        diagnose_consistency=arguments.diagnose_consistency,
    )


if __name__ == "__main__":
    main()
