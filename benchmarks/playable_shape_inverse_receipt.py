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
from nova.equilibrium.shape_inverse import achieved_target
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


def _arm_receipt(
    name: str,
    machine: ForwardMachine,
    previous,
    target,
    null_points: np.ndarray,
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
            "current_change_l2_a": float(np.linalg.norm(item.inverse.delta)),
            "linear_row_prediction": (
                item.inverse.response[:, item.inverse.free_circuits]
                @ item.inverse.delta
            ).tolist(),
            "linear_row_command": (
                item.inverse.target - item.inverse.observed
            ).tolist(),
            "least_squares_residual": item.inverse.least_squares_residual,
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
            f"circuit_{index:02d}": float(current)
            for index, current in enumerate(solver.prescribed_current)
        },
        "rounds": rounds,
        "round_count": len(rounds),
        "total_wall_s": float(sum(item["wall_s"] for item in rounds)),
        "total_trips": int(sum(item["trips"] for item in rounds)),
        "converged": bool(np.asarray(equilibrium.fixed_point.converged)),
        "final_turning_point_error_m": rounds[-1]["turning_point_error_m"],
    }
    return payload, achieved


def _null_receipt(machine: ForwardMachine, previous) -> tuple[dict[str, Any], object]:
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
            f"circuit_{index:02d}": float(current)
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


def measure(directory: Path = DEFAULT_DIRECTORY) -> dict[str, Any]:
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
    machine = ForwardMachine(
        profile=profile,
        seed=jnp.asarray(passive_case["state"]),
        wall=np.asarray(profile.operator.wall.coordinate),
        identity="mast-22086/43",
    )
    prime_solver = ProductionSolver(machine)
    prime, _program = prime_solver._forward(
        profile, machine.seed, prime_solver.prescribed_current
    )
    previous_target = achieved_target(profile, prime.flux)
    null_arm, null_equilibrium = _null_receipt(machine, prime)
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
        "policy": str(policy),
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
        arm, _achieved = _arm_receipt(name, machine, prime, target, null_points)
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
    receipt = {"runtime": runtime, "null_arm": null_arm, "arms": arms}
    _write(directory / "shape-inverse-receipt.json", receipt)
    _draw(arms, directory / "shape-inverse-receipt.png")
    return receipt


def main() -> None:
    """Run the receipt from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    arguments = parser.parse_args()
    measure(arguments.directory)


if __name__ == "__main__":
    main()
