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
from nova.equilibrium.shape_inverse import achieved_target, solve_shape_inverse
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


def _upper_command_diagnostic(
    machine: ForwardMachine,
    previous,
    target,
    circuit_names: dict[int, str],
) -> dict[str, Any]:
    """Return the uncapped active-current command without solving it forward."""
    seed = np.asarray(machine.profile.operator.prescribed_current_field.current)
    inverse = solve_shape_inverse(
        machine.profile,
        target,
        previous.flux,
        prescribed_current=seed,
        free_circuits=machine.drivable_circuits,
    )
    fractions = np.abs(inverse.delta) / np.abs(seed[inverse.free_circuits])
    return {
        "command": "upper-point-plus-20mm",
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
        "linear_row_prediction": (
            inverse.response[:, inverse.free_circuits] @ inverse.delta
        ).tolist(),
        "linear_row_command": (inverse.target - inverse.observed).tolist(),
        "response_singular_values": inverse.singular_values.tolist(),
        "response_numerical_rank": inverse.numerical_rank,
        "response_rank_threshold": inverse.rank_threshold,
        "null_modes": _null_modes(inverse, circuit_names),
    }


def _arm_receipt(
    name: str,
    machine: ForwardMachine,
    previous,
    target,
    null_points: np.ndarray,
    circuit_names: dict[int, str],
) -> tuple[dict[str, Any], object]:
    """Run one arm through ``ProductionSolver`` and return its receipt."""
    solver = ProductionSolver(machine)
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
            "linear_row_prediction": (
                item.inverse.response[:, item.inverse.free_circuits]
                @ item.inverse.delta
            ).tolist(),
            "uncapped_linear_row_prediction": (
                item.inverse.response[:, item.inverse.free_circuits]
                @ item.inverse.uncapped_delta
            ).tolist(),
            "linear_row_command": (
                item.inverse.target - item.inverse.observed
            ).tolist(),
            "least_squares_residual": item.inverse.least_squares_residual,
            "uncapped_least_squares_residual": (
                item.inverse.uncapped_least_squares_residual
            ),
            "response_singular_values": item.inverse.singular_values.tolist(),
            "response_numerical_rank": item.inverse.numerical_rank,
            "response_rank_threshold": item.inverse.rank_threshold,
            "response_conditioning_span": float(
                item.inverse.singular_values[0]
                / item.inverse.singular_values[item.inverse.numerical_rank - 1]
            ),
            "null_modes": _null_modes(item.inverse, circuit_names),
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
            "maximum_fraction_per_round": solver.current_step_fraction,
            "maximum_rounds": solver.max_inverse_rounds,
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
    directory: Path = DEFAULT_DIRECTORY, *, diagnose_upper: bool = False
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
    if diagnose_upper:
        diagnostic = _upper_command_diagnostic(
            machine,
            prime,
            _upper_point_target(previous_target),
            circuit_names,
        )
        _write(directory / "upper-point-plus-20mm-diagnostic.json", diagnostic)
        print("UPPER-COMMAND-DIAGNOSTIC " + json.dumps(diagnostic), flush=True)
        return {"diagnostic": diagnostic}
    null_arm, null_equilibrium = _null_receipt(machine, prime, circuit_names)
    null_points = _points(achieved_target(profile, null_equilibrium.flux))
    definitions = (
        ("upper-point-plus-20mm", _upper_point_target(previous_target)),
        ("elongation-plus-5pct", _elongation_target(previous_target)),
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
    for name, target in definitions:
        arm, _achieved = _arm_receipt(
            name, machine, prime, target, null_points, circuit_names
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
    parser.add_argument("--diagnose-upper", action="store_true")
    arguments = parser.parse_args()
    measure(arguments.directory, diagnose_upper=arguments.diagnose_upper)


if __name__ == "__main__":
    main()
