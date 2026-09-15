#!/usr/bin/env python3
"""Measure a directly referenced multi-cap centroid command on MAST 27079."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zarr

from benchmarks import compensator_jacobian_response as comparison
from benchmarks import constraint_jacobian_receipt as globalisation
from nova.equilibrium import reduced_newton
from nova.jax.config import configure_dtypes
from nova.media.ink import DEFAULT_INK, trace_axes
from scripts.labeller_batch import shard

configure_dtypes()


ROOT = Path(__file__).resolve().parents[1]
SHOT = 27079
ROW = 96
DIRECT_FREE_CURRENTS_A = (3_000.0, 4_000.0)
TARGET_REFERENCE_CURRENT_A = 3_000.0
CURRENT_STEP_CAP_A = 1_000.0
CENTROID_TOLERANCE_M = 1.0e-3
TRIP_LIMIT = 12
HISTORICAL_RECEIPT = (
    ROOT / "docs/figures/playable-forward-solve/constraint-globalisation/"
    "constraint-globalisation.json"
)


def _revision() -> str:
    """Return the exact revision that supplied the benchmark code."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace the durable receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _historical_case() -> dict[str, Any]:
    """Read the recorded large command that previously needed five caps."""
    payload = json.loads(HISTORICAL_RECEIPT.read_text(encoding="utf-8"))
    candidates = [
        case
        for case in payload["cases"]
        if case.get("trip_records") and not case.get("diagnostic", False)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "historical receipt must contain exactly one primary multi-trip case"
        )
    case = candidates[0]
    return {
        "source_revision": payload["source_revision"],
        "target_centroid_z_m": float(case["target_centroid_z_m"]),
        "free_current_reference_a": float(case["expected_free_solve_current_a"]),
        "recorded_capped_application_count": int(case["capped_application_count"]),
        "recorded_compensating_current_a": float(case["compensating_current_a"]),
    }


def _base_payload() -> dict[str, Any]:
    """Return the receipt written before the first solve starts."""
    return {
        "schema": "constraint-multi-trip-target",
        "status": "running",
        "source_revision": _revision(),
        "shot": SHOT,
        "row": ROW,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platform": jax.default_backend(),
        "current_step_cap_a": CURRENT_STEP_CAP_A,
        "centroid_tolerance_m": CENTROID_TOLERANCE_M,
        "trip_limit": TRIP_LIMIT,
        "target_selection": (
            "the centroid reached by the directly solved plus 3000 A free response"
        ),
        "historical_input": _historical_case(),
        "free_response_samples": [],
        "instrument_check": {},
        "measurements": {},
        "acceptance": {},
    }


def _centroid(prepared, state, flux) -> float:
    """Read the achieved vertical current centroid from one solved state."""
    return globalisation._centroid(
        prepared,
        flux,
        state["target_current"],
        state["requested_value"],
    )


def _free_responses(
    prepared,
    state: dict[str, Any],
    direction: np.ndarray,
    payload: dict[str, Any],
    persist: Callable[[], None],
) -> dict[float, dict[str, Any]]:
    """Measure and persist the direct 3 kA and 4 kA free responses."""
    samples: dict[float, dict[str, Any]] = {}
    program = state["free"].program
    initial_centroid = float(state["free_centroid_z_m"])
    for current in DIRECT_FREE_CURRENTS_A:
        sample, program = globalisation._free_response_sample(
            prepared,
            state,
            direction,
            initial_centroid,
            current,
            program,
        )
        sample["centroid_displacement_m"] = (
            None
            if sample["centroid_z_m"] is None
            else float(sample["centroid_z_m"] - initial_centroid)
        )
        samples[current] = sample
        payload["free_response_samples"].append(sample)
        persist()
    centroids = [samples[current]["centroid_z_m"] for current in DIRECT_FREE_CURRENTS_A]
    instrument_check = {
        "both_free_solves_converged": all(
            samples[current]["converged"] for current in DIRECT_FREE_CURRENTS_A
        ),
        "both_displacements_nonzero": all(
            samples[current]["centroid_displacement_m"] != 0.0
            for current in DIRECT_FREE_CURRENTS_A
        ),
        "responses_are_distinct": centroids[0] != centroids[1],
        "centroid_difference_m": float(centroids[1] - centroids[0]),
    }
    payload["instrument_check"] = instrument_check
    persist()
    if not all(instrument_check.values()):
        raise RuntimeError(f"free-response instrument check failed: {instrument_check}")
    return samples


def _run_capped_iteration(
    name: str,
    prepared,
    state: dict[str, Any],
    direction: np.ndarray,
    *,
    target_centroid_z_m: float,
    free_current_reference_a: float,
    payload: dict[str, Any],
    persist: Callable[[], None],
) -> dict[str, Any]:
    """Run one constrained command and persist its receipt after every trip."""
    pair = globalisation._explicit_pair(
        prepared.profile,
        target_centroid_z_m,
        direction,
    )
    current_state = state["free"].state
    current_unknown = jnp.asarray(pair.binding.initial_unknown)
    initial_centroid = float(state["free_centroid_z_m"])
    measurement: dict[str, Any] = {
        "status": "running",
        "initial_centroid_z_m": initial_centroid,
        "target_centroid_z_m": target_centroid_z_m,
        "target_displacement_m": target_centroid_z_m - initial_centroid,
        "free_current_reference_a": free_current_reference_a,
        "trip_records": [],
        "merit_sequence": [],
    }
    payload["measurements"][name] = measurement
    persist()
    program = None
    for trip_index in range(TRIP_LIMIT):
        trip_start_centroid = _centroid(prepared, state, current_state)
        trip_pair = globalisation._carry_pair(pair, current_unknown)
        result = reduced_newton.solve_constrained_reduced_newton(
            prepared.profile,
            current_state,
            constraint_pairs=(trip_pair,),
            requested_class=state["requested"],
            target_current=state["target_current"],
            prescribed_current=jnp.asarray(state["current"]),
            tolerance=globalisation.FIXED_POINT_CRITERION,
            newton_steps=globalisation.NEWTON_STEPS,
            active_set_steps=1,
            constraint_current_step_cap=CURRENT_STEP_CAP_A,
            program=program,
        )
        if not measurement["merit_sequence"]:
            initial_merit = globalisation._constraint_merit(
                prepared,
                current_state,
                result,
                current_unknown,
                current_state,
                state["requested"],
            )
            measurement["merit_sequence"].append(initial_merit)
        final_centroid = _centroid(prepared, state, result.state)
        record = result.constraints[0]
        physical_unknown = float(np.asarray(record.physical_unknown)[0])
        previous_physical_unknown = float(
            np.asarray(pair.unknown.physical_value(current_unknown))[0]
        )
        applied_current = physical_unknown - previous_physical_unknown
        command = target_centroid_z_m - trip_start_centroid
        overshot = bool(
            command != 0.0 and (final_centroid - target_centroid_z_m) * command > 0.0
        )
        post_merit = globalisation._constraint_merit(
            prepared,
            result.state,
            result,
            result.compensating_unknown,
            current_state,
            state["requested"],
        )
        previous_merit = float(measurement["merit_sequence"][-1])
        trip_record = {
            "trip": trip_index + 1,
            "applied_current_change_a": applied_current,
            "cumulative_compensating_current_a": physical_unknown,
            "augmented_constraint_merit": post_merit,
            "augmented_constraint_merit_change": post_merit - previous_merit,
            "achieved_centroid_z_m": final_centroid,
            "centroid_error_m": final_centroid - target_centroid_z_m,
            "overshot_target": overshot,
            "converged": bool(result.converged),
            "termination_reason": result.termination_name,
            "newton_steps": int(sum(result.newton_steps_per_trip)),
        }
        measurement["trip_records"].append(trip_record)
        measurement["merit_sequence"].append(post_merit)
        measurement["last_recorded_termination_reason"] = result.termination_name
        persist()
        current_state = result.state
        current_unknown = result.compensating_unknown
        program = result.program
        if result.converged or result.termination_name == "sufficient_decrease_refused":
            break

    records = measurement["trip_records"]
    if not records:
        raise RuntimeError(f"{name} produced no trip records")
    final = records[-1]
    applied = [float(item["applied_current_change_a"]) for item in records]
    application_count = sum(value != 0.0 for value in applied)
    merit_sequence = [float(value) for value in measurement["merit_sequence"]]
    applied_trip_merits = [merit_sequence[0]] + [
        float(item["augmented_constraint_merit"])
        for item in records
        if item["applied_current_change_a"] != 0.0
    ]
    measurement.update(
        {
            "status": "complete",
            "trip_count": len(records),
            "capped_application_count": application_count,
            "converged": bool(final["converged"]),
            "termination_reason": final["termination_reason"],
            "final_centroid_z_m": final["achieved_centroid_z_m"],
            "centroid_error_m": final["centroid_error_m"],
            "target_within_one_mm": (
                abs(final["centroid_error_m"]) <= CENTROID_TOLERANCE_M
            ),
            "compensating_current_a": final["cumulative_compensating_current_a"],
            "current_difference_percent": 100.0
            * (
                abs(final["cumulative_compensating_current_a"])
                - abs(free_current_reference_a)
            )
            / abs(free_current_reference_a),
            "current_within_ten_percent": (
                abs(
                    abs(final["cumulative_compensating_current_a"])
                    - abs(free_current_reference_a)
                )
                <= 0.1 * abs(free_current_reference_a)
            ),
            "applied_current_cap_respected": all(
                abs(value) <= CURRENT_STEP_CAP_A for value in applied
            ),
            "accepted_step_overshot_target": any(
                item["overshot_target"] and item["applied_current_change_a"] != 0.0
                for item in records
            ),
            "merit_monotone_all_trips": all(
                after <= before
                for before, after in zip(merit_sequence, merit_sequence[1:])
            ),
            "applied_trip_merit_sequence": applied_trip_merits,
            "merit_strictly_decreased_across_applied_trips": all(
                after < before
                for before, after in zip(applied_trip_merits, applied_trip_merits[1:])
            ),
        }
    )
    persist()
    return measurement


def _write_figure(path: Path, measurement: dict[str, Any]) -> None:
    """Plot augmented merit and absolute centroid error against trip."""
    records = measurement["trip_records"]
    trips = [int(item["trip"]) for item in records]
    merits = [float(item["augmented_constraint_merit"]) for item in records]
    errors = [1.0e3 * abs(float(item["centroid_error_m"])) for item in records]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    for axis in axes:
        trace_axes(axis)
    axes[0].plot(
        trips,
        merits,
        color=DEFAULT_INK.flux_color,
        marker="o",
        linewidth=DEFAULT_INK.trace_linewidth,
        markersize=DEFAULT_INK.trace_markersize,
    )
    axes[0].set_ylabel("augmented constraint merit")
    axes[1].semilogy(
        trips,
        errors,
        color=DEFAULT_INK.separatrix_color,
        marker="o",
        linewidth=DEFAULT_INK.trace_linewidth,
        markersize=DEFAULT_INK.trace_markersize,
    )
    axes[1].axhline(
        1.0,
        color=DEFAULT_INK.contour_color,
        linewidth=DEFAULT_INK.contour_linewidth,
    )
    axes[1].set_ylabel("absolute centroid error [mm]")
    axes[1].set_xlabel("recorded trip")
    axes[1].set_xticks(trips)
    figure.suptitle("MAST 27079 row 96: bounded centroid command")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", facecolor=DEFAULT_INK.figure_facecolor)
    plt.close(figure)


def _format(value: Any, digits: int = 9) -> str:
    """Format one report scalar without discarding a recorded qualifier."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _report_text(payload: dict[str, Any], receipt: Path, figure: Path) -> str:
    """Render the quantitative report and per-trip table."""
    reproduction = payload["measurements"]["historical_target_reproduction"]
    target = payload["measurements"]["direct_three_kiloampere_target"]
    free_rows = []
    for sample in payload["free_response_samples"]:
        free_rows.append(
            (
                "| {current} | {converged} | {trips} | {centroid} | {displacement} |"
            ).format(
                current=_format(sample["current_delta_a"]),
                converged=_format(sample["converged"]),
                trips=_format(sample["trip_count"]),
                centroid=_format(sample["centroid_z_m"]),
                displacement=_format(sample["centroid_displacement_m"]),
            )
        )
    trip_rows = []
    for trip in target["trip_records"]:
        trip_rows.append(
            "| {trip} | {current} | {cumulative} | {merit} | {centroid} | "
            "{error} | {reason} |".format(
                trip=trip["trip"],
                current=_format(trip["applied_current_change_a"]),
                cumulative=_format(trip["cumulative_compensating_current_a"]),
                merit=_format(trip["augmented_constraint_merit"]),
                centroid=_format(trip["achieved_centroid_z_m"]),
                error=_format(1.0e3 * trip["centroid_error_m"]),
                reason=trip["termination_reason"],
            )
        )
    acceptance = payload["acceptance"]
    convergence_line = (
        f"- Converged to within 1 mm: {_format(acceptance['converged_within_one_mm'])}"
    )
    all_merit_line = (
        "- Raw merit decreased monotonically over every recorded boundary "
        "(retained, non-gating): "
        f"{_format(acceptance['merit_monotone_all_trips'])}"
    )
    applied_merit_line = (
        "- Merit strictly decreased over every trip applying current: "
        f"{_format(acceptance['merit_strictly_decreased_across_applied_trips'])}"
    )
    application_count_line = (
        "- Three to four capped applications: "
        f"{_format(acceptance['three_to_four_capped_applications'])}"
    )
    cap_line = (
        "- No application exceeded 1000 A: "
        f"{_format(acceptance['applied_current_cap_respected'])}"
    )
    overshoot_line = (
        "- No accepted current step overshot the target: "
        f"{_format(acceptance['no_accepted_step_overshot'])}"
    )
    nonprogress_increases = [
        trip
        for trip in target["trip_records"]
        if trip["augmented_constraint_merit_change"] > 0.0
        and trip["applied_current_change_a"] == 0.0
        and trip["newton_steps"] == 0
    ]
    if nonprogress_increases:
        item = nonprogress_increases[0]
        null_trip_line = (
            f"- Raw merit first rose on closure trip {item['trip']} by "
            f"{_format(item['augmented_constraint_merit_change'])}; that trip "
            "applied 0 A and took 0 Newton steps."
        )
    else:
        null_trip_line = "- No zero-work closure trip increased the raw merit."
    return "\n".join(
        [
            "# Multi-cap constrained centroid command",
            "",
            (
                "Current-base reproduction: the earlier large command still used "
                f"{reproduction['capped_application_count']} nonzero capped "
                f"applications and {reproduction['trip_count']} recorded trips, "
                f"ending at {_format(reproduction['compensating_current_a'])} A. "
                f"Its recorded termination reason was "
                f"`{reproduction['termination_reason']}`."
            ),
            "",
            (
                "The replacement command is not a linear extrapolation. Its target is "
                "the centroid reached by the directly solved +3000 A free response; "
                "the +4000 A response is measured beside it as an instrument and "
                "nonlinearity check."
            ),
            "",
            "## Direct free responses",
            "",
            "| current change [A] | converged | trips | centroid Z [m] | "
            "displacement [m] |",
            "|---:|---|---:|---:|---:|",
            *free_rows,
            "",
            "## Constrained trip receipt",
            "",
            "| trip | applied current [A] | cumulative current [A] | merit | "
            "centroid Z [m] | error [mm] | recorded reason |",
            "|---:|---:|---:|---:|---:|---:|---|",
            *trip_rows,
            "",
            "## Verdict",
            "",
            (
                f"Overall: **{'PASS' if payload['passed'] else 'FAIL'}**. The "
                f"iteration recorded {target['trip_count']} trips and "
                f"{target['capped_application_count']} nonzero capped applications. "
                f"Its final centroid error was "
                f"{_format(1.0e3 * target['centroid_error_m'])} mm and the recorded "
                f"termination reason was `{target['termination_reason']}`."
            ),
            "",
            convergence_line,
            all_merit_line,
            applied_merit_line,
            null_trip_line,
            application_count_line,
            cap_line,
            overshoot_line,
            (
                "- Total current within 10 percent of the directly measured free "
                f"value: {_format(acceptance['current_within_ten_percent'])} "
                f"({_format(target['compensating_current_a'])} A against "
                f"{_format(target['free_current_reference_a'])} A; "
                f"{_format(target['current_difference_percent'])} percent)."
            ),
            "",
            (
                "The termination reason above is reported exactly as returned. Any "
                "termination-order change belongs to the separately owned solver node."
            ),
            "",
            f"Receipt: `{receipt}`. Figure: `{figure}`.",
            "",
        ]
    )


def measure(receipt: Path, figure: Path, reports: tuple[Path, ...]) -> dict[str, Any]:
    """Run the reproduction, direct probes, and measured multi-cap target."""
    payload = _base_payload()

    def persist() -> None:
        _write_json(receipt, payload)

    persist()
    prepared = shard.prepare_labeller()
    prepared = shard._prepared_with_polarity(prepared, shard._shot_polarity(SHOT))
    group = zarr.open_group(str(shard.SHOT_STORE / f"{SHOT}.zarr"), mode="r")["efm"]
    state = globalisation._row_state(prepared, group, ROW)
    active_names = globalisation._circuit_names(prepared.policy_evidence)
    direction = comparison._direction(
        active_names,
        "p6_upper",
        state["current"].size,
    )
    historical = payload["historical_input"]
    reproduction = _run_capped_iteration(
        "historical_target_reproduction",
        prepared,
        state,
        direction,
        target_centroid_z_m=historical["target_centroid_z_m"],
        free_current_reference_a=historical["free_current_reference_a"],
        payload=payload,
        persist=persist,
    )
    free = _free_responses(prepared, state, direction, payload, persist)
    selected = free[TARGET_REFERENCE_CURRENT_A]
    if not selected["converged"] or selected["centroid_z_m"] is None:
        raise RuntimeError(
            "the directly measured plus 3000 A free target is unavailable"
        )
    target = _run_capped_iteration(
        "direct_three_kiloampere_target",
        prepared,
        state,
        direction,
        target_centroid_z_m=float(selected["centroid_z_m"]),
        free_current_reference_a=TARGET_REFERENCE_CURRENT_A,
        payload=payload,
        persist=persist,
    )
    acceptance = {
        "historical_five_cap_behaviour_reproduced": (
            reproduction["capped_application_count"]
            == historical["recorded_capped_application_count"]
            == 5
        ),
        "free_response_instrument_check_passed": all(
            payload["instrument_check"].values()
        ),
        "converged_within_one_mm": bool(
            target["converged"] and target["target_within_one_mm"]
        ),
        "merit_monotone_all_trips": target["merit_monotone_all_trips"],
        "merit_strictly_decreased_across_applied_trips": target[
            "merit_strictly_decreased_across_applied_trips"
        ],
        "three_to_four_capped_applications": (
            3 <= target["capped_application_count"] <= 4
        ),
        "applied_current_cap_respected": target["applied_current_cap_respected"],
        "no_accepted_step_overshot": not target["accepted_step_overshot_target"],
        "current_within_ten_percent": target["current_within_ten_percent"],
    }
    gating_keys = (
        "historical_five_cap_behaviour_reproduced",
        "free_response_instrument_check_passed",
        "converged_within_one_mm",
        "merit_strictly_decreased_across_applied_trips",
        "three_to_four_capped_applications",
        "applied_current_cap_respected",
        "no_accepted_step_overshot",
        "current_within_ten_percent",
    )
    payload["acceptance"] = acceptance
    payload["gating_acceptance_keys"] = list(gating_keys)
    payload["merit_interpretation"] = (
        "strict decrease is required across trips that apply current; zero-current, "
        "zero-Newton closure re-evaluations remain in the raw sequence but do not "
        "measure globalisation progress"
    )
    payload["passed"] = all(acceptance[key] for key in gating_keys)
    payload["status"] = "complete"
    persist()
    _write_figure(figure, target)
    report = _report_text(payload, receipt, figure)
    for report_path in reports:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    return payload


def main() -> None:
    """Run the benchmark with explicit durable destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    args = parser.parse_args()
    payload = measure(
        args.receipt.resolve(),
        args.figure.resolve(),
        tuple(path.resolve() for path in args.report),
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
