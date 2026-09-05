"""Measure a steered keyframe on the constrained reduced-Newton route.

One bank row is solved free on the reduced-amplitude plain-Newton route, and
its converged equilibrium is then steered: the vertical current-centroid row
is commanded one centimetre away and re-solved from that equilibrium, and then
a second centimetre away and re-solved from the equilibrium the first move
left behind, unknowns included.  Each move reports the active-set trips, the
Newton steps, the keyframe wall, the compensating current every circuit
carries on the derived direction, and the centroid it achieved against the one
it was commanded.

The free arm is measured in the same job so the keyframe wall is read against
this machine's own unconstrained warm trip rather than against a stored
figure, and every move is written to the receipt as it lands, so a job that
runs out of wall clock still delivers the moves it measured.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium import reduced_newton
from nova.equilibrium.constraint import (
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
    compensator_rule_name,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = (22086, 43)
DEFAULT_OUTPUT = (
    ROOT
    / "docs/figures/playable-forward-solve/constrained-route/steered-keyframes.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/playable-forward-solve/constrained-route/steered-keyframes.png"
)
#: Newton budget per trip.  The prototype holds one Jacobian for a whole trip,
#: so its inner iteration is a chord method and contracts linearly where an
#: exact-tangent Newton contracts quadratically.
NEWTON_STEPS = 24
#: Commanded vertical displacements of the current centroid [m], each solved
#: from the equilibrium the previous one left behind.
MOVES = (1.0e-2, 2.0e-2)
#: The unconstrained warm trip the millisecond route banked on the four bank
#: rows, in seconds.  The keyframe wall is stated beside it rather than
#: against it: a constrained trip carries the same topology read plus the
#: rows, so the interval is the reference an added cost is read from.
BANKED_WARM_TRIP_S = (46.0e-3, 58.0e-3)


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_float(value: Any) -> float | None:
    """Return one finite host float, or None where the value is not finite."""
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def _centroid(profile, flux, target_current) -> float:
    """Return the vertical current centroid [m] of one flux state."""
    return float(
        np.asarray(
            profile.current_moment_observation(
                jnp.asarray(flux),
                support=MomentIntegralSupport.ALL_DOMAIN,
                target_current=target_current,
            ).centroid_z
        )
    )


def _circuit_names(policy) -> dict[int, str]:
    """Return the zero-based circuit index of every named active family."""
    return {
        int(item["stored_circuit"]) - 1: str(item["family"])
        for item in policy["active_mapping"]
    }


def _centroid_pair(profile, flux, *, target, unknown, target_current, requested, names):
    """Return the centroid row with a matrix-led compensating direction.

    The seed pair carries a multiplier rather than a circuit, so the
    derivation supplies both the direction and the current amplitude that
    moves the row by one declared scale.  ``unknown`` is the normalised
    compensation the previous move settled on, which is what makes a warm
    start warm in the unknowns as well as in the flux.
    """
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seeded = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    (derived,), selection = profile.derived_constraint_pairs(
        (seeded,),
        jnp.asarray(flux),
        requested_class=requested,
        target_current=target_current,
        circuits=sorted(names),
    )
    rescaled = ConstraintPair(
        functional=derived.functional,
        unknown=derived.unknown,
        binding=ConstraintBinding(
            target=derived.binding.target,
            tolerance=derived.binding.tolerance,
            scale=derived.binding.scale,
            initial_unknown=jnp.asarray(
                np.atleast_1d(
                    np.asarray(0.0 if unknown is None else unknown, dtype=float)
                )
            ),
        ),
    )
    return rescaled, selection


def _walls(result) -> dict[str, Any]:
    """Return the per-trip wall breakdown one solve measured."""
    warm = result.trip_wall_per_trip[1:]
    return {
        "trip_wall_per_trip_s": result.trip_wall_per_trip,
        "jacobian_wall_per_trip_s": result.jacobian_wall_per_trip,
        "newton_wall_per_trip_s": result.newton_wall_per_trip,
        "boundary_wall_per_trip_s": result.boundary_wall_per_trip,
        "first_trip_wall_s": (
            result.trip_wall_per_trip[0] if result.trip_wall_per_trip else None
        ),
        "median_warm_trip_wall_s": float(np.median(warm)) if warm else None,
    }


def _free_arm(operator, seed, target_current, requested) -> dict[str, Any]:
    """Solve the row free on the reduced route and time every trip."""
    started = time.perf_counter()
    result = reduced_newton.solve_reduced_newton(
        operator,
        seed,
        requested_class=requested,
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        stream=True,
    )
    wall = time.perf_counter() - started
    summary = {
        "route": "reduced_newton.solve_reduced_newton",
        "keyframe_wall_s": wall,
        "terminal_residual": result.terminal_residual,
        "active_set_trips": result.active_set_iterations,
        "newton_steps_per_trip": result.newton_steps_per_trip,
        "newton_step_count": int(sum(result.newton_steps_per_trip)),
        "converged": result.converged,
        "termination": result.termination_name,
        "reduced_dimension": result.reduced_dimension,
        "off_support_leakage": result.off_support_leakage,
        "banked_warm_trip_interval_s": list(BANKED_WARM_TRIP_S),
    }
    summary.update(_walls(result))
    return result, summary


def _move_arm(
    profile, state, unknown, *, target, target_current, requested, names
) -> dict[str, Any]:
    """Command one centroid target and re-solve from the state supplied."""
    pair, selection = _centroid_pair(
        profile,
        state,
        target=target,
        unknown=unknown,
        target_current=target_current,
        requested=requested,
        names=names,
    )
    started = time.perf_counter()
    result = reduced_newton.solve_constrained_reduced_newton(
        profile,
        state,
        constraint_pairs=(pair,),
        requested_class=requested,
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        stream=True,
    )
    wall = time.perf_counter() - started
    record = result.constraints[0]
    direction = np.asarray(pair.unknown.direction)
    delta = direction @ np.asarray(record.physical_unknown)
    achieved = _centroid(profile, result.state, target_current)
    peak = float(np.max(np.abs(delta))) if delta.size else 0.0
    summary = {
        "route": "reduced_newton.solve_constrained_reduced_newton",
        "commanded_centroid_m": float(target),
        "achieved_centroid_m": achieved,
        "centroid_error_m": achieved - float(target),
        "row_error_m": _strict_float(record.physical_residual[0]),
        "row_qualified": bool(np.asarray(record.qualified)[0]),
        "keyframe_wall_s": wall,
        "terminal_residual": result.terminal_residual,
        "active_set_trips": result.active_set_iterations,
        "newton_steps_per_trip": result.newton_steps_per_trip,
        "newton_step_count": int(sum(result.newton_steps_per_trip)),
        "converged": result.converged,
        "termination": result.termination_name,
        "compensator_rule": compensator_rule_name(record.compensator_rule),
        "normalised_unknown": _strict_float(record.normalized_unknown[0]),
        "compensating_current_a": _strict_float(record.physical_unknown[0]),
        "compensating_current_norm_a": float(np.linalg.norm(delta)),
        "compensating_current_per_circuit_a": [
            {
                "circuit": int(index),
                "family": names.get(int(index)),
                "current_a": float(delta[index]),
                "direction_component": float(direction[index, 0]),
            }
            for index in np.argsort(np.abs(delta))[::-1][:8]
            if peak > 0.0 and abs(float(delta[index])) > 1.0e-9 * peak
        ],
        "direction_authority_row_scales_per_ampere": [
            float(value) for value in np.asarray(selection.direction_authority)
        ],
        "ampere_scale_a": float(np.asarray(pair.unknown.ampere_scale)[0]),
        "warm_started_unknown": None if unknown is None else float(unknown),
        "banked_warm_trip_interval_s": list(BANKED_WARM_TRIP_S),
    }
    summary.update(_walls(result))
    return result, summary


def _draw(receipt: dict[str, Any], output: Path) -> None:
    """Draw the keyframe wall, the trips and the achieved centroid per move."""
    moves = receipt["moves"]
    free = receipt["free"]
    if not moves:
        return
    labels = [
        "free",
        *[f"+{1.0e2 * move['commanded_move_m']:.0f} cm" for move in moves],
    ]
    walls = [free["keyframe_wall_s"], *[move["keyframe_wall_s"] for move in moves]]
    trips = [free["active_set_trips"], *[move["active_set_trips"] for move in moves]]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    position = np.arange(len(labels))
    axes[0].bar(position, [1.0e3 * value for value in walls], color="#3b6ea5")
    axes[0].axhspan(
        1.0e3 * BANKED_WARM_TRIP_S[0],
        1.0e3 * BANKED_WARM_TRIP_S[1],
        color="0.75",
        alpha=0.5,
        label="banked unconstrained warm trip",
    )
    axes[0].set_ylabel("keyframe wall [ms]")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(position, trips, color="#3b6ea5")
    axes[1].set_ylabel("active-set trips")
    axes[2].plot(
        [move["commanded_centroid_m"] for move in moves],
        [move["achieved_centroid_m"] for move in moves],
        "o-",
        color="#a53b3b",
    )
    axes[2].plot(
        [move["commanded_centroid_m"] for move in moves],
        [move["commanded_centroid_m"] for move in moves],
        "k--",
        linewidth=0.8,
        label="commanded",
    )
    axes[2].set_xlabel("commanded centroid [m]")
    axes[2].set_ylabel("achieved centroid [m]")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in (axes[0], axes[1]):
        axis.set_xticks(position, labels)
        axis.grid(axis="y", alpha=0.2)
    axes[2].grid(alpha=0.2)
    figure.suptitle(
        f"Steered keyframes on the constrained reduced route — {receipt['identity']}",
        y=0.97,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.84, wspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write(receipt: dict[str, Any], output: Path) -> None:
    """Persist the receipt so far, creating its directory once."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def measure(*, output: Path, figure: Path, cache_root: Path | None = None):
    """Solve the row free, then steer it one commanded centimetre at a time."""
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
        if cache_root is None
        else cache_root
    )
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    identity = f"{TARGET[0]}/{TARGET[1]}"
    receipt: dict[str, Any] = {
        "artifact": "steered keyframes on the constrained reduced-Newton route",
        "identity": identity,
        "source_commit": _source_revision(),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
            "scheduler": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            },
        },
        "evidence_inputs": {
            "response_carrier": carrier_evidence,
            "persistent_compilation_cache": cache.receipt(),
        },
        "measurement_contract": {
            "seed": "persisted passive-inclusive bank seed",
            "free_route": "reduced_newton.solve_reduced_newton",
            "constrained_route": (
                "reduced_newton.solve_constrained_reduced_newton: the "
                "compensating unknowns extend the amplitude vector and the "
                "authored rows extend the amplitude residual"
            ),
            "warm_start": (
                "each move re-solves from the equilibrium and the normalised "
                "compensation the previous move settled on"
            ),
            "commanded_moves_m": list(MOVES),
            "convergence_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "banked_warm_trip_interval_s": list(BANKED_WARM_TRIP_S),
        },
        "free": None,
        "moves": [],
    }
    _write(receipt, output)

    selected_row, qualification = selected[TARGET]
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
    seed = jnp.asarray(passive_case["state"])
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    names = _circuit_names(policy)

    print(f"CONSTRAINED-KEYFRAME free {identity}", flush=True)
    free_result, free_summary = _free_arm(
        profile.operator, seed, target_current, requested
    )
    free_centroid = _centroid(profile, free_result.state, target_current)
    free_summary["vertical_centroid_m"] = free_centroid
    receipt["free"] = free_summary
    _write(receipt, output)
    print(
        "CONSTRAINED-KEYFRAME-DONE " + json.dumps(free_summary, sort_keys=True),
        flush=True,
    )

    state = free_result.state
    unknown = None
    for move in MOVES:
        target = free_centroid + move
        print(f"CONSTRAINED-KEYFRAME move {move:+.3f} {identity}", flush=True)
        result, summary = _move_arm(
            profile,
            state,
            unknown,
            target=target,
            target_current=target_current,
            requested=requested,
            names=names,
        )
        summary["commanded_move_m"] = float(move)
        summary["free_centroid_m"] = free_centroid
        receipt["moves"].append(summary)
        _write(receipt, output)
        _draw(receipt, figure)
        print(
            "CONSTRAINED-KEYFRAME-DONE " + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        state = result.state
        unknown = float(np.asarray(result.compensating_unknown)[0])

    receipt["verdict"] = {
        "moves_measured": len(receipt["moves"]),
        "every_move_converged": all(move["converged"] for move in receipt["moves"]),
        "max_abs_centroid_error_m": max(
            (abs(move["centroid_error_m"]) for move in receipt["moves"]), default=None
        ),
        "max_keyframe_wall_s": max(
            (move["keyframe_wall_s"] for move in receipt["moves"]), default=None
        ),
        "free_keyframe_wall_s": free_summary["keyframe_wall_s"],
    }
    _write(receipt, output)
    _draw(receipt, figure)
    return receipt


def main() -> None:
    """Parse the caller's operands and run the measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--cache-root", type=Path, default=None)
    arguments = parser.parse_args()
    measure(
        output=arguments.output,
        figure=arguments.figure,
        cache_root=arguments.cache_root,
    )


if __name__ == "__main__":
    main()
