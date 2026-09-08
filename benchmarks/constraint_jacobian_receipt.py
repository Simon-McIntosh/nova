#!/usr/bin/env python3
"""Measure bounded constrained-Newton globalisation on selected MAST rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zarr

from benchmarks import compensator_jacobian_response as comparison
from benchmarks.efit_forward_parity_slice import FIXED_POINT_CRITERION
from benchmarks.forward_labeller_throughput import NEWTON_STEPS, _circuit_names
from nova.equilibrium import reduced_newton
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintMultiplier,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.observation import MomentIntegralSupport
from scripts.labeller_batch import shard


ROOT = Path(__file__).resolve().parents[1]
SHOT = 27079
BANKED_ROWS = (16, 96)
EARLY_ROWS = (15, 16, 17)
DIRECTIONS = ("p6_upper", "p6_upper_minus_p6_lower")
CURRENT_STEP_CAP_A = 1_000.0
CENTROID_TOLERANCE_M = 1.0e-3
ACTIVE_SET_TRIPS = 8
BANKED_RECEIPT = (
    ROOT / "docs/figures/playable-forward-solve/compensator-jacobian/"
    "compensator-jacobian-response.json"
)


def _revision() -> str:
    """Return the source revision used for one receipt."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _centroid(prepared, flux, target_current, requested_value) -> float:
    """Return the labeller's achieved vertical current centroid in metres."""
    return float(
        shard._centroid_coordinates(
            prepared,
            flux,
            target_current,
            requested_class=requested_value,
        )[-1]
    )


def _seeded_pair(profile, target: float) -> ConstraintPair:
    """Return one unselected centroid row at a physical target."""
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    return ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([CENTROID_TOLERANCE_M]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )


def _explicit_pair(profile, target: float, direction: np.ndarray) -> ConstraintPair:
    """Return one centroid row driven along a declared circuit direction."""
    seeded = _seeded_pair(profile, target)
    return ConstraintPair(
        functional=seeded.functional,
        unknown=CircuitCurrentUnknown(
            direction=jnp.asarray(direction[:, None]),
            ampere_scale=jnp.asarray([CURRENT_STEP_CAP_A]),
        ),
        binding=seeded.binding,
    )


def _row_state(prepared, group, row: int) -> dict[str, Any]:
    """Build one row's seed, arguments, and converged free equilibrium."""
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    seed = np.asarray(shard._slices_seed(group, row, full_r, full_z), dtype=float)
    inputs = shard._slice_inputs(group, row)
    if inputs is None or not np.all(np.isfinite(seed)):
        raise ValueError(f"row {row} has no finite labeller seed and inputs")
    requested_value = shard._requested_class(group, row)
    requested = jnp.asarray(requested_value, dtype=jnp.int8)
    target_current = abs(float(inputs["reference_plasma_current"]))
    current = np.asarray(inputs["current"], dtype=np.float64)
    free = reduced_newton.solve_reduced_newton(
        prepared.profile.operator,
        jnp.asarray(seed),
        requested_class=requested,
        target_current=target_current,
        prescribed_current=jnp.asarray(current),
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
    )
    if not free.converged:
        raise ValueError(f"row {row} free equilibrium did not converge")
    return {
        "inputs": inputs,
        "requested": requested,
        "requested_value": requested_value,
        "target_current": target_current,
        "current": current,
        "free": free,
        "free_centroid_z_m": _centroid(
            prepared, free.state, target_current, requested_value
        ),
    }


def _solve_row(
    prepared, state, pair: ConstraintPair, *, target: float
) -> dict[str, Any]:
    """Run one bounded constraint row and return its complete scalar verdict."""
    free = state["free"]
    initial = state["free_centroid_z_m"]
    try:
        result = reduced_newton.solve_constrained_reduced_newton(
            prepared.profile,
            free.state,
            constraint_pairs=(pair,),
            requested_class=state["requested"],
            target_current=state["target_current"],
            prescribed_current=jnp.asarray(state["current"]),
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            active_set_steps=ACTIVE_SET_TRIPS,
            constraint_current_step_cap=CURRENT_STEP_CAP_A,
        )
    except Exception as error:
        return {
            "converged": False,
            "termination_reason": f"{type(error).__name__}: {error}",
            "initial_centroid_z_m": initial,
            "target_centroid_z_m": target,
            "final_centroid_z_m": None,
            "centroid_error_m": None,
            "overshot": None,
            "trip_count": None,
            "newton_steps": None,
            "newton_steps_per_trip": None,
            "jacobian_builds_per_trip": None,
            "compensating_current_a": None,
            "constraint_residual_m": None,
        }
    final = _centroid(
        prepared,
        result.state,
        state["target_current"],
        state["requested_value"],
    )
    command = target - initial
    overshot = bool(command != 0.0 and (final - target) * command > 0.0)
    record = result.constraints[0]
    return {
        "converged": bool(result.converged),
        "termination_reason": result.termination_name,
        "initial_centroid_z_m": initial,
        "target_centroid_z_m": target,
        "final_centroid_z_m": final,
        "centroid_error_m": final - target,
        "overshot": overshot,
        "trip_count": int(result.active_set_iterations),
        "newton_steps": int(sum(result.newton_steps_per_trip)),
        "newton_steps_per_trip": result.newton_steps_per_trip,
        "jacobian_builds_per_trip": result.jacobian_builds_per_trip,
        "compensating_current_a": float(np.asarray(record.physical_unknown)[0]),
        "constraint_residual_m": float(np.asarray(record.physical_residual)[0]),
    }


def _banked_cases(prepared, group, active_names, banked) -> list[dict[str, Any]]:
    """Measure the two declared P6 directions on both banked rows."""
    cases = []
    cached: dict[int, dict[str, Any]] = {}
    for row in BANKED_ROWS:
        cached[row] = _row_state(prepared, group, row)
        for circuit in DIRECTIONS:
            reference = banked[(row, circuit)]
            target = float(reference["perturbed_solves"]["plus"]["centroid_z_m"])
            direction = comparison._direction(
                active_names, circuit, cached[row]["current"].size
            )
            outcome = _solve_row(
                prepared,
                cached[row],
                _explicit_pair(prepared.profile, target, direction),
                target=target,
            )
            current_ok = (
                outcome["compensating_current_a"] is not None
                and 0.9 * CURRENT_STEP_CAP_A
                <= abs(outcome["compensating_current_a"])
                <= 1.1 * CURRENT_STEP_CAP_A
            )
            case_passed = bool(
                outcome["converged"]
                and outcome["trip_count"] is not None
                and outcome["trip_count"] <= ACTIVE_SET_TRIPS
                and outcome["centroid_error_m"] is not None
                and abs(outcome["centroid_error_m"]) <= CENTROID_TOLERANCE_M
                and (current_ok if row == 96 else True)
            )
            cases.append(
                {
                    "row": row,
                    "target_source": "banked plus 1 kA free solve",
                    "circuit": circuit,
                    "accepted": case_passed,
                    "flat_top_current_within_ten_percent": (
                        current_ok if row == 96 else None
                    ),
                    **outcome,
                }
            )
    return cases


def _early_cases(prepared, group, active_names) -> list[dict[str, Any]]:
    """Measure EFIT-centroid commands without accepting target overshoot."""
    cases = []
    for row in EARLY_ROWS:
        state = _row_state(prepared, group, row)
        target = float(state["inputs"]["target_centroid_z"])
        seeded = _seeded_pair(prepared.profile, target)
        (pair,), selection = reduced_newton.derive_reduced_constraint_pairs(
            prepared.profile,
            (seeded,),
            state["free"].state,
            requested_class=state["requested"],
            target_current=state["target_current"],
            prescribed_current=jnp.asarray(state["current"]),
            circuits=sorted(active_names),
            program=state["free"].program,
        )
        outcome = _solve_row(prepared, state, pair, target=target)
        has_reason = bool(outcome["termination_reason"])
        case_passed = bool(
            outcome["overshot"] is False and (outcome["converged"] or has_reason)
        )
        cases.append(
            {
                "row": row,
                "target_source": "EFIT current centroid",
                "circuit": "implicit-response selection",
                "selected_rule": str(selection.rule.name.lower()),
                "accepted": case_passed,
                **outcome,
            }
        )
    return cases


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    """Write the compact per-case table beside the JSON receipt."""
    fields = (
        "row",
        "target_source",
        "circuit",
        "converged",
        "termination_reason",
        "initial_centroid_z_m",
        "target_centroid_z_m",
        "final_centroid_z_m",
        "centroid_error_m",
        "overshot",
        "trip_count",
        "newton_steps",
        "compensating_current_a",
        "accepted",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)


def _write_figure(path: Path, cases: list[dict[str, Any]]) -> None:
    """Plot target error and compensating current for every completed case."""
    labels = [f"{case['row']}\n{case['circuit']}" for case in cases]
    errors = [
        np.nan if case["centroid_error_m"] is None else 1.0e3 * case["centroid_error_m"]
        for case in cases
    ]
    currents = [
        np.nan
        if case["compensating_current_a"] is None
        else case["compensating_current_a"] / 1.0e3
        for case in cases
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(labels, errors, color="#6f4aa8")
    axes[0].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].axhline(-1.0, color="black", lw=0.8, ls="--")
    axes[0].set_ylabel("centroid error [mm]")
    axes[1].bar(labels, currents, color="#4d7f78")
    axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[1].axhline(-1.0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("compensation [kA]")
    axes[1].tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _format(value: Any, digits: int = 6) -> str:
    """Format a scalar for one compact Markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _write_report(path: Path, receipt: Path, figure: Path, cases) -> None:
    """Write one outcome table per measured row."""
    lines = [
        "# Constrained Newton globalisation on MAST 27079",
        "",
        "The unknown remains a physical circuit current and every command is a "
        "diagnostic placement row, not shape control. Bounded candidates are "
        "accepted only when the augmented merit decreases, and every accepted "
        "candidate is re-linearised before another direction is formed.",
        "",
        f"Receipt: `{receipt}`. Figure: `{figure}`.",
    ]
    for row in sorted({case["row"] for case in cases}):
        lines.extend(
            [
                "",
                f"## Row {row}",
                "",
                "| target | direction | converged / reason | trips | Newton steps | "
                "final error [mm] | compensation [A] | overshot | accepted |",
                "|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for case in (item for item in cases if item["row"] == row):
            outcome = "converged" if case["converged"] else case["termination_reason"]
            error_mm = (
                None
                if case["centroid_error_m"] is None
                else 1.0e3 * case["centroid_error_m"]
            )
            lines.append(
                "| {target} | {direction} | {outcome} | {trips} | {steps} | "
                "{error} | {current} | {overshot} | {accepted} |".format(
                    target=case["target_source"],
                    direction=case["circuit"],
                    outcome=outcome,
                    trips=_format(case["trip_count"]),
                    steps=_format(case["newton_steps"]),
                    error=_format(error_mm),
                    current=_format(case["compensating_current_a"]),
                    overshot=_format(case["overshot"]),
                    accepted=_format(case["accepted"]),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure(output: Path, report: Path) -> dict[str, Any]:
    """Write the bounded globalisation receipt, table, figure, and report."""
    output.mkdir(parents=True, exist_ok=True)
    prepared = shard.prepare_labeller()
    prepared = shard._prepared_with_polarity(prepared, shard._shot_polarity(SHOT))
    group = zarr.open_group(str(shard.SHOT_STORE / f"{SHOT}.zarr"), mode="r")["efm"]
    active_names = _circuit_names(prepared.policy_evidence)
    banked_payload = json.loads(BANKED_RECEIPT.read_text(encoding="utf-8"))
    banked = {
        (int(item["row"]), str(item["circuit"])): item
        for item in banked_payload["rows_detail"]
    }
    cases = _banked_cases(prepared, group, active_names, banked)
    cases.extend(_early_cases(prepared, group, active_names))
    receipt = output / "constraint-globalisation.json"
    table = output / "constraint-globalisation.csv"
    figure = output / "constraint-globalisation.svg"
    payload = {
        "schema": "constraint-globalisation",
        "source_revision": _revision(),
        "shot": SHOT,
        "current_step_cap_a": CURRENT_STEP_CAP_A,
        "centroid_tolerance_m": CENTROID_TOLERANCE_M,
        "active_set_trip_limit": ACTIVE_SET_TRIPS,
        "banked_receipt": str(BANKED_RECEIPT.relative_to(ROOT)),
        "passed": all(case["accepted"] for case in cases),
        "cases": cases,
    }
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(table, cases)
    _write_figure(figure, cases)
    _write_report(report, receipt, figure, cases)
    return payload


def main() -> None:
    """Run the receipt with explicitly chosen destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args.output.resolve(), args.report.resolve())
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
