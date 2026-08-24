"""Explain the coefficient carrier's single admitted advance.

The diagnostic replays the plasma-only carrier on the same closed-form coarse
frame and sequential factor ladder as the banked carrier measurement.  Every
offered candidate records the independent finiteness, topology, and residual
checks that decide admission.  At each attempted advance it also compares the
production exact-value Newton--Krylov direction with that direction projected
through the carrier and measures the dense coefficient residual Jacobian.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import carrier_arms as carrier_measurement
from nova.equilibrium import fixed_point as fixed_point_solver
from nova.equilibrium.coefficient_carrier import (
    CoefficientCarrier,
    relative_exact_residual,
)
from nova.equilibrium.fixed_point import KrylovActionQualification
from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "docs/figures/coefficient-space-newton/carrier-advance.json"
ROOT_BANK = HERE / "scripts/oracle_rebaseline/root-coarse.npz"
ROOT_RECEIPT = HERE / "scripts/oracle_rebaseline/results.json"
CARRIER_RECEIPT = (
    HERE / "docs/figures/coefficient-space-newton/plasma-only-carrier.json"
)
FIXTURE_MODULE = HERE / "scripts/analytic_oracle_fixtures/measure.py"
KNOTS_PER_AXIS = 6
ATTEMPTED_ADVANCES = 4
GMRES_ITERATIONS = 30
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
BANKED_PROJECTED_KRYLOV_CONDITION = 4087.40844825275


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    """Return the source identity used by the measurement."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _angle_and_norm(reference, compared) -> dict[str, float]:
    """Measure directional agreement between two exact-space steps."""

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(compared, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denominator = left_norm * right_norm
    cosine = (
        float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
        if denominator > 0.0
        else float("nan")
    )
    return {
        "angle_degrees": float(np.degrees(np.arccos(cosine))),
        "cosine_similarity": cosine,
        "reference_l2_norm": left_norm,
        "compared_l2_norm": right_norm,
        "compared_to_reference_norm_ratio": right_norm / left_norm,
    }


def _candidate_record(
    operator, coefficients, output, residual, current_residual, factor
):
    """Record every independent admission check and one exclusive outcome."""

    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    output_values = np.asarray(output, dtype=np.float64)
    residual_value = float(residual)
    coefficient_finite = bool(np.all(np.isfinite(coefficient_values)))
    output_and_residual_finite = bool(
        np.all(np.isfinite(output_values)) and np.isfinite(residual_value)
    )
    _masks, topology = operator.read(jnp.asarray(output))
    axis = np.asarray(topology.axis, dtype=np.float64)
    axis_finite = bool(np.all(np.isfinite(axis)))
    remains_limited = not bool(topology.diverted)
    improves_residual = residual_value < current_residual

    if not coefficient_finite:
        reason = "NONFINITE_COEFFICIENT_CANDIDATE"
    elif not output_and_residual_finite:
        reason = "NONFINITE_EXACT_OUTPUT_OR_RESIDUAL"
    elif not axis_finite:
        reason = "NONFINITE_TOPOLOGY_AXIS"
    elif not remains_limited:
        reason = "LEFT_LIMITED_TOPOLOGY_CLASS"
    elif not improves_residual:
        reason = "EXACT_RESIDUAL_NOT_IMPROVED"
    else:
        reason = "ADMITTED"

    return {
        "factor": factor,
        "outcome": reason,
        "exact_relative_residual": residual_value,
        "checks": {
            "coefficient_candidate_finite": coefficient_finite,
            "exact_output_and_residual_finite": output_and_residual_finite,
            "topology_axis_finite": axis_finite,
            "remains_limited": remains_limited,
            "improves_exact_residual": improves_residual,
        },
        "topology": {
            "class": boundary_mode(topology).value,
            "axis_m": axis.tolist() if axis_finite else None,
        },
    }


def _step_probe_record(operator, state, output, residual, current_residual, factor):
    """Record whether one non-promoting exact-space diagnostic trial is useful."""

    state_values = np.asarray(state, dtype=np.float64)
    output_values = np.asarray(output, dtype=np.float64)
    residual_value = float(residual)
    _masks, topology = operator.read(jnp.asarray(output))
    axis = np.asarray(topology.axis, dtype=np.float64)
    return {
        "factor": factor,
        "exact_relative_residual": residual_value,
        "improves_exact_residual": residual_value < current_residual,
        "state_finite": bool(np.all(np.isfinite(state_values))),
        "output_and_residual_finite": bool(
            np.all(np.isfinite(output_values)) and np.isfinite(residual_value)
        ),
        "topology_class": boundary_mode(topology).value,
        "topology_axis_finite": bool(np.all(np.isfinite(axis))),
    }


def _measure_attempt(exact_map, operator, carrier, external, coefficients, index):
    """Measure one dense carrier step and its sequential admission ladder."""

    known_external = jnp.asarray(external)

    def coefficient_residual(value):
        exact_output = exact_map(known_external + carrier.expand(value))
        return carrier.project(exact_output - known_external) - value

    def evaluated(value):
        exact_state = known_external + carrier.expand(value)
        exact_output = exact_map(exact_state)
        return (
            exact_state,
            exact_output,
            relative_exact_residual(exact_output, exact_state),
        )

    exact_state, exact_output, current_residual = evaluated(coefficients)
    current_residual.block_until_ready()
    residual_vector = coefficient_residual(coefficients)
    jacobian = jax.jacfwd(coefficient_residual)(coefficients)
    jacobian.block_until_ready()
    coefficient_step = jnp.linalg.solve(jacobian, -residual_vector)
    coefficient_step.block_until_ready()

    jacobian_host = np.asarray(jacobian, dtype=np.float64)
    residual_host = np.asarray(residual_vector, dtype=np.float64)
    coefficient_step_host = np.asarray(coefficient_step, dtype=np.float64)
    coefficient_linear_remainder = jacobian_host @ coefficient_step_host + residual_host
    coefficient_linear_residual = float(
        np.linalg.norm(coefficient_linear_remainder)
        / max(np.linalg.norm(residual_host), np.finfo(np.float64).tiny)
    )

    mapped, tangent = jax.linearize(exact_map, exact_state)
    exact_residual_vector = mapped - exact_state
    qualified_step = fixed_point_solver._qualified_krylov_step(
        lambda vector: vector - tangent(vector),
        exact_residual_vector,
        relative_exact_residual(mapped, exact_state),
        gmres_iterations=GMRES_ITERATIONS,
        condition_ratio_limit=np.e,
        preceding_condition_baseline=jnp.asarray(np.nan, dtype=exact_state.dtype),
    )
    exact_newton_step = qualified_step.unconditioned_step
    exact_newton_step.block_until_ready()
    projected_exact_step = carrier.expand(carrier.project(exact_newton_step))
    projected_exact_step.block_until_ready()
    projected_exact_coefficient_step = carrier.project(exact_newton_step)
    expanded_coefficient_step = carrier.expand(coefficient_step)
    expanded_coefficient_step.block_until_ready()

    exact_step_probe = []
    projected_exact_step_probe = []
    for factor in FACTORS:
        exact_probe_state = exact_state + factor * exact_newton_step
        exact_probe_output = exact_map(exact_probe_state)
        exact_probe_residual = relative_exact_residual(
            exact_probe_output, exact_probe_state
        )
        exact_probe_residual.block_until_ready()
        exact_step_probe.append(
            _step_probe_record(
                operator,
                exact_probe_state,
                exact_probe_output,
                exact_probe_residual,
                float(current_residual),
                factor,
            )
        )

        projected_probe_coefficients = (
            coefficients + factor * projected_exact_coefficient_step
        )
        projected_probe_state, projected_probe_output, projected_probe_residual = (
            evaluated(projected_probe_coefficients)
        )
        projected_probe_residual.block_until_ready()
        projected_exact_step_probe.append(
            _step_probe_record(
                operator,
                projected_probe_state,
                projected_probe_output,
                projected_probe_residual,
                float(current_residual),
                factor,
            )
        )

    candidates = []
    chosen = None
    for factor in FACTORS:
        candidate = coefficients + factor * coefficient_step
        candidate_state, candidate_output, candidate_residual = evaluated(candidate)
        candidate_residual.block_until_ready()
        record = _candidate_record(
            operator,
            candidate,
            candidate_output,
            candidate_residual,
            float(current_residual),
            factor,
        )
        candidates.append(record)
        if record["outcome"] == "ADMITTED":
            chosen = (
                candidate,
                candidate_state,
                candidate_output,
                candidate_residual,
                factor,
            )
            break

    attempt = {
        "attempt": index + 1,
        "current_exact_relative_residual": float(current_residual),
        "dense_coefficient_jacobian": {
            "shape": list(jacobian_host.shape),
            "condition_number_2_norm": float(np.linalg.cond(jacobian_host)),
            "condition_number_over_banked_projected_krylov": float(
                np.linalg.cond(jacobian_host) / BANKED_PROJECTED_KRYLOV_CONDITION
            ),
            "achieved_relative_linear_residual_2_norm": (coefficient_linear_residual),
        },
        "exact_newton_krylov_step": {
            "gmres_iterations": GMRES_ITERATIONS,
            "qualification": KrylovActionQualification(
                int(qualified_step.qualification)
            ).name,
            "projected_krylov_condition_on_diagnostic_state": float(
                qualified_step.projected_condition
            ),
        },
        "exact_step_after_carrier_projection": _angle_and_norm(
            exact_newton_step, projected_exact_step
        ),
        "dense_coefficient_step_against_projected_exact_step": _angle_and_norm(
            projected_exact_step, expanded_coefficient_step
        ),
        "step_utility_probe": {
            "promotion_affected": False,
            "exact_newton_step": exact_step_probe,
            "same_step_after_carrier_projection": projected_exact_step_probe,
        },
        "candidate_ladder": {
            "selection": "sequential first admitted candidate",
            "records": candidates,
            "admitted_candidate_count": sum(
                record["outcome"] == "ADMITTED" for record in candidates
            ),
            "rejected_candidate_count": sum(
                record["outcome"] != "ADMITTED" for record in candidates
            ),
            "accepted_factor": chosen[-1] if chosen is not None else None,
        },
    }
    return attempt, chosen


def _verdict(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign the observed stall while preserving all three discriminators."""

    stalled = attempts[-1]
    records = stalled["candidate_ladder"]["records"]
    refusal_reasons = {record["outcome"] for record in records}
    topology_refused = any(
        reason in {"NONFINITE_TOPOLOGY_AXIS", "LEFT_LIMITED_TOPOLOGY_CLASS"}
        for reason in refusal_reasons
    )
    projection = stalled["exact_step_after_carrier_projection"]
    coefficient_mismatch = stalled[
        "dense_coefficient_step_against_projected_exact_step"
    ]
    dense_condition = stalled["dense_coefficient_jacobian"]["condition_number_2_norm"]
    exact_step_useful = any(
        record["improves_exact_residual"]
        for record in stalled["step_utility_probe"]["exact_newton_step"]
    )
    projected_step_useful = any(
        record["improves_exact_residual"]
        for record in stalled["step_utility_probe"][
            "same_step_after_carrier_projection"
        ]
    )

    if topology_refused:
        classification = "ADMISSION"
        basis = (
            "At least one rejected candidate failed the unchanged topology predicate."
        )
    elif exact_step_useful and not projected_step_useful:
        classification = "PROJECTION"
        basis = (
            "The exact Newton direction has a residual-improving factor but the same "
            "direction after carrier projection does not."
        )
    elif (
        projected_step_useful
        and coefficient_mismatch["compared_to_reference_norm_ratio"] <= 0.1
    ):
        classification = "PROJECTION"
        basis = (
            "The projected exact Newton direction remains useful, but forming and "
            "solving the projected residual Jacobian collapses its exact-space norm "
            "by at least tenfold."
        )
    elif dense_condition > BANKED_PROJECTED_KRYLOV_CONDITION:
        classification = "CONDITIONING"
        basis = (
            "Topology admits every offered candidate and projection does not destroy "
            "the exact direction, while the dense coefficient Jacobian is worse "
            "conditioned than the banked projected Krylov comparator."
        )
    else:
        classification = "THREE_MECHANISMS_EXCLUDED"
        basis = (
            "Topology admits every offered candidate, carrier projection preserves a "
            "forward direction with more than ten per cent norm, and the dense "
            "coefficient Jacobian is better conditioned than the banked comparator."
        )

    return {
        "classification": classification,
        "basis": basis,
        "admission_discriminator": {
            "topology_refused_any_stalled_candidate": topology_refused,
            "exclusive_refusal_reasons": sorted(refusal_reasons),
        },
        "projection_discriminator": {
            "exact_step_has_improving_factor": exact_step_useful,
            "projected_exact_step_has_improving_factor": projected_step_useful,
            "direct_projection": projection,
            "dense_coefficient_step_against_projected_exact_step": (
                coefficient_mismatch
            ),
        },
        "conditioning_discriminator": {
            "dense_coefficient_condition_number": dense_condition,
            "banked_projected_krylov_condition_number": (
                BANKED_PROJECTED_KRYLOV_CONDITION
            ),
            "dense_to_banked_ratio": (
                dense_condition / BANKED_PROJECTED_KRYLOV_CONDITION
            ),
        },
    }


def measure() -> dict[str, Any]:
    """Run the fixed-frame diagnostic and return its quantitative receipt."""

    configure_dtypes()
    fixture = carrier_measurement._load_module(
        FIXTURE_MODULE, "carrier_advance_fixture"
    )
    case = fixture.analytic_case()
    machine = fixture.cached_machine(
        case,
        fixture.FIXTURE_REQUESTS["coarse"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    coordinate = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact_analytic = fixture.exact_state(case, coordinate)
    zero_exterior = fixture.forward_operator(case, machine)
    exact_physical = fixture.exact_current_moments(case, zero_exterior, exact_analytic)
    exact_coefficients = zero_exterior.coupling_current_moments(exact_physical)
    exact_internal = fixture._internal_flux_image(zero_exterior, exact_coefficients)
    operator = fixture.forward_operator(case, machine, exact_analytic - exact_internal)
    exact_map = operator.flux_map()
    external = np.asarray(operator.external(), dtype=np.float64)

    with np.load(ROOT_BANK, allow_pickle=False) as bank:
        root = np.asarray(bank["root_state"], dtype=np.float64)
        seed = np.asarray(bank["seed_state"], dtype=np.float64)
    initial_exact = root + 0.02 * (seed - root)
    carrier = CoefficientCarrier.from_coordinates(
        coordinate,
        radial_knots=KNOTS_PER_AXIS,
        vertical_knots=KNOTS_PER_AXIS,
    )
    coefficients = carrier.project(initial_exact - external)

    carrier_receipt = json.loads(CARRIER_RECEIPT.read_text(encoding="utf-8"))
    root_receipt = json.loads(ROOT_RECEIPT.read_text(encoding="utf-8"))["fixtures"][
        "coarse"
    ]
    baseline_admitted = int(carrier_receipt["arms"]["A"]["admitted_advance_count"])
    banked_carrier_admitted = int(
        carrier_receipt["arms"]["C"]["admitted_advance_count"]
    )
    if baseline_admitted != 10 or banked_carrier_admitted != 1:
        raise RuntimeError("the banked admitted-advance comparators changed")

    attempts = []
    admitted_advances = 0
    refused_advances = 0
    for index in range(ATTEMPTED_ADVANCES):
        attempt, chosen = _measure_attempt(
            exact_map,
            operator,
            carrier,
            external,
            coefficients,
            index,
        )
        attempts.append(attempt)
        if chosen is None:
            refused_advances += 1
            break
        coefficients = chosen[0]
        admitted_advances += 1

    if admitted_advances != banked_carrier_admitted:
        raise RuntimeError("the diagnostic carrier trajectory changed")
    if refused_advances != 1:
        raise RuntimeError("the carrier did not expose one terminal refused advance")

    rejected_candidates = sum(
        attempt["candidate_ladder"]["rejected_candidate_count"] for attempt in attempts
    )
    return {
        "artifact": str(OUTPUT.relative_to(HERE)),
        "schema": "carrier-advance-diagnosis-1",
        "source_commit": _source_commit(),
        "measurement_scope": {
            "frame": "closed-form-oracle-coarse",
            "realised_plasma_cells": len(machine.node),
            "exact_state_values": root.size,
            "coefficient_count": carrier.coefficient_count,
            "knots_per_axis": KNOTS_PER_AXIS,
            "platform": jax.devices()[0].platform,
            "precision": "float64",
            "carrier_state": (
                "plasma-only flux with known external field restored before every "
                "exact map read"
            ),
            "candidate_selection": (
                "unchanged sequential factors from the banked dense carrier solve"
            ),
        },
        "evidence_inputs": {
            "root_bank": str(ROOT_BANK.relative_to(HERE)),
            "root_bank_sha256": _sha256(ROOT_BANK),
            "root_receipt": str(ROOT_RECEIPT.relative_to(HERE)),
            "root_receipt_sha256": _sha256(ROOT_RECEIPT),
            "carrier_receipt": str(CARRIER_RECEIPT.relative_to(HERE)),
            "carrier_receipt_sha256": _sha256(CARRIER_RECEIPT),
            "banked_projected_krylov_condition": (BANKED_PROJECTED_KRYLOV_CONDITION),
            "banked_root_terminal_residual": root_receipt["metric"][
                "fixed_point_residual"
            ]["recovery_value"],
        },
        "advance_summary": {
            "baseline_exact_value_admitted_advances": baseline_admitted,
            "banked_carrier_admitted_advances": banked_carrier_admitted,
            "diagnostic_carrier_admitted_advances": admitted_advances,
            "diagnostic_carrier_refused_advances": refused_advances,
            "diagnostic_rejected_candidates": rejected_candidates,
        },
        "attempts": attempts,
        "verdict": _verdict(attempts),
    }


def main() -> None:
    """Write the receipt and print its compact evidence summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    receipt = measure()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt["advance_summary"], sort_keys=True), flush=True)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
