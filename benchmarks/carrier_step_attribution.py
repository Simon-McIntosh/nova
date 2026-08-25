"""Attribute the coefficient carrier step to its differentiated residual.

The measurement replays the plasma-only carrier on the closed-form coarse
frame.  It compares traced Jacobian columns with central finite differences
along every carrier basis direction, separately for the exact and projected
residuals.  It then replaces the square projected-residual solve with the
least-squares Newton step of the exact residual restricted to the carrier
subspace and measures the resulting trajectory.
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
    dense_newton,
    relative_exact_residual,
)
from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "docs/figures/coefficient-space-newton/carrier-step-attribution.json"
ROOT_BANK = HERE / "scripts/oracle_rebaseline/root-coarse.npz"
ROOT_RECEIPT = HERE / "scripts/oracle_rebaseline/results.json"
CARRIER_RECEIPT = (
    HERE / "docs/figures/coefficient-space-newton/plasma-only-carrier.json"
)
ADVANCE_RECEIPT = HERE / "docs/figures/coefficient-space-newton/carrier-advance.json"
FIXTURE_MODULE = HERE / "scripts/analytic_oracle_fixtures/measure.py"
KNOTS_PER_AXIS = 6
NEWTON_STEPS = 4
GMRES_ITERATIONS = 30
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
BANKED_EXACT_ARM_RESIDUAL = 2.7110550053242652e-15
BANKED_STALLED_RESIDUAL = 3.847988208719086e-2
BANKED_STEP_ANGLE_DEGREES = 49.73527543299367
BANKED_STEP_NORM_RATIO = 0.012955053999124593


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


def _column_comparison(traced, finite_difference) -> dict[str, Any]:
    """Compare matrices column by column and fit one multiplicative scale."""

    traced_values = np.asarray(traced, dtype=np.float64)
    difference_values = np.asarray(finite_difference, dtype=np.float64)
    if traced_values.shape != difference_values.shape:
        raise ValueError("column comparison requires matching matrix shapes")
    tiny = np.finfo(np.float64).tiny
    column_denominators = np.maximum(np.linalg.norm(difference_values, axis=0), tiny)
    column_errors = (
        np.linalg.norm(traced_values - difference_values, axis=0) / column_denominators
    )
    traced_energy = float(np.vdot(traced_values, traced_values).real)
    fitted_scale = float(
        np.vdot(traced_values, difference_values).real / max(traced_energy, tiny)
    )
    scaled_errors = (
        np.linalg.norm(fitted_scale * traced_values - difference_values, axis=0)
        / column_denominators
    )
    return {
        "shape": list(traced_values.shape),
        "finite_difference_model": "central_difference_per_basis_direction",
        "scale_fit_model": "finite_difference_equals_scale_times_traced",
        "fitted_scale_factor": fitted_scale,
        "maximum_relative_column_error": float(np.max(column_errors)),
        "median_relative_column_error": float(np.median(column_errors)),
        "maximum_scale_fitted_relative_column_error": float(np.max(scaled_errors)),
        "relative_column_errors": column_errors.tolist(),
    }


def _central_difference_jacobian(
    function, coefficients
) -> tuple[np.ndarray, list[float]]:
    """Differentiate every carrier basis direction with a fixed roundoff rule."""

    values = np.asarray(coefficients, dtype=np.float64)
    scale = np.cbrt(np.finfo(np.float64).eps)
    baseline = np.asarray(function(jnp.asarray(values)), dtype=np.float64)
    columns = []
    steps = []
    for index, coefficient in enumerate(values):
        step = scale * max(1.0, abs(float(coefficient)))
        direction = np.zeros_like(values)
        direction[index] = step
        positive = np.asarray(
            function(jnp.asarray(values + direction)), dtype=np.float64
        )
        negative = np.asarray(
            function(jnp.asarray(values - direction)), dtype=np.float64
        )
        if positive.shape != baseline.shape or negative.shape != baseline.shape:
            raise RuntimeError("residual shape changed under a basis perturbation")
        columns.append((positive - negative) / (2.0 * step))
        steps.append(step)
    return np.stack(columns, axis=1), steps


def _evaluated(exact_map, carrier, external, coefficients):
    """Return total state, exact output, and the scored exact residual."""

    state = external + carrier.expand(coefficients)
    output = exact_map(state)
    return state, output, relative_exact_residual(output, state)


def _admissible(operator, output) -> bool:
    """Apply the carrier arm's unchanged finite limited-topology predicate."""

    _masks, topology = operator.read(output)
    return bool(jnp.all(jnp.isfinite(topology.axis)) & (~topology.diverted))


def _projected_residual_trajectory(
    exact_map, operator, carrier, external, initial_coefficients
) -> dict[str, Any]:
    """Replay the square projected-residual formulation to its stalled state."""

    def projected_residual(value):
        output = exact_map(external + carrier.expand(value))
        return carrier.project(output - external) - value

    coefficients = jnp.asarray(initial_coefficients)
    _state, _output, residual = _evaluated(exact_map, carrier, external, coefficients)
    trace = [float(residual)]
    accepted_factors = []
    for _ in range(NEWTON_STEPS):
        residual_vector = projected_residual(coefficients)
        jacobian = jax.jacfwd(projected_residual)(coefficients)
        step = jnp.linalg.solve(jacobian, -residual_vector)
        step.block_until_ready()
        chosen = None
        for factor in FACTORS:
            candidate = coefficients + factor * step
            _state, output, candidate_residual = _evaluated(
                exact_map, carrier, external, candidate
            )
            candidate_residual.block_until_ready()
            if _admissible(operator, output) and float(candidate_residual) < float(
                residual
            ):
                chosen = candidate, candidate_residual, factor
                break
        if chosen is None:
            break
        coefficients, residual, factor = chosen
        trace.append(float(residual))
        accepted_factors.append(factor)
    return {
        "coefficients": coefficients,
        "terminal_exact_relative_residual": float(residual),
        "trace": trace,
        "admitted_advances": len(accepted_factors),
        "accepted_factors": accepted_factors,
    }


def _exact_newton_step(exact_map, exact_state):
    """Return the production exact-value Newton--Krylov diagnostic step."""

    mapped, tangent = jax.linearize(exact_map, exact_state)
    exact_residual = mapped - exact_state
    qualified = fixed_point_solver._qualified_krylov_step(
        lambda vector: vector - tangent(vector),
        exact_residual,
        relative_exact_residual(mapped, exact_state),
        gmres_iterations=GMRES_ITERATIONS,
        condition_ratio_limit=np.e,
        preceding_condition_baseline=jnp.asarray(np.nan, dtype=exact_state.dtype),
    )
    qualified.unconditioned_step.block_until_ready()
    return qualified


def _step_utility(exact_map, operator, carrier, external, coefficients, step):
    """Score one coefficient-space direction without promoting a candidate."""

    _state, _output, current_residual = _evaluated(
        exact_map, carrier, external, coefficients
    )
    rows = []
    for factor in FACTORS:
        candidate = coefficients + factor * step
        _state, output, residual = _evaluated(exact_map, carrier, external, candidate)
        residual.block_until_ready()
        _masks, topology = operator.read(output)
        rows.append(
            {
                "factor": factor,
                "exact_relative_residual": float(residual),
                "improves_exact_residual": float(residual) < float(current_residual),
                "admissible": _admissible(operator, output),
                "topology_class": boundary_mode(topology).value,
            }
        )
    return rows


def measure() -> dict[str, Any]:
    """Run the fixed-frame attribution and return its quantitative receipt."""

    configure_dtypes()
    fixture = carrier_measurement._load_module(
        FIXTURE_MODULE, "carrier_step_attribution_fixture"
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
    external = jnp.asarray(operator.external(), dtype=jnp.float64)

    with np.load(ROOT_BANK, allow_pickle=False) as bank:
        root = np.asarray(bank["root_state"], dtype=np.float64)
        seed = np.asarray(bank["seed_state"], dtype=np.float64)
    initial_exact = root + 0.02 * (seed - root)
    carrier = CoefficientCarrier.from_coordinates(
        coordinate,
        radial_knots=KNOTS_PER_AXIS,
        vertical_knots=KNOTS_PER_AXIS,
    )
    initial_coefficients = carrier.project(initial_exact - external)
    projected_trajectory = _projected_residual_trajectory(
        exact_map,
        operator,
        carrier,
        external,
        initial_coefficients,
    )
    if projected_trajectory["admitted_advances"] != 1:
        raise RuntimeError(
            "projected-residual trajectory no longer stalls after one advance"
        )
    stalled_coefficients = projected_trajectory.pop("coefficients")
    exact_state, exact_output, stalled_residual = _evaluated(
        exact_map, carrier, external, stalled_coefficients
    )

    def exact_residual(value):
        state = external + carrier.expand(value)
        return exact_map(state) - state

    def projected_residual(value):
        output = exact_map(external + carrier.expand(value))
        return carrier.project(output - external) - value

    traced_exact_jacobian = jax.jacfwd(exact_residual)(stalled_coefficients)
    traced_projected_jacobian = jax.jacfwd(projected_residual)(stalled_coefficients)
    traced_exact_jacobian.block_until_ready()
    traced_projected_jacobian.block_until_ready()
    finite_exact_jacobian, finite_difference_steps = _central_difference_jacobian(
        exact_residual, stalled_coefficients
    )
    finite_projected_jacobian, _ = _central_difference_jacobian(
        projected_residual, stalled_coefficients
    )
    expanded_projected_jacobian = np.asarray(
        carrier.expansion, dtype=np.float64
    ) @ np.asarray(traced_projected_jacobian, dtype=np.float64)

    exact_residual_vector = exact_output - exact_state
    projected_residual_vector = projected_residual(stalled_coefficients)
    projected_step = jnp.linalg.solve(
        traced_projected_jacobian, -projected_residual_vector
    )
    corrected_step = jnp.linalg.lstsq(
        traced_exact_jacobian, -exact_residual_vector, rcond=None
    )[0]
    projected_step.block_until_ready()
    corrected_step.block_until_ready()
    qualified_exact_step = _exact_newton_step(exact_map, exact_state)
    projected_exact_step = carrier.expand(
        carrier.project(qualified_exact_step.unconditioned_step)
    )
    projected_exact_step.block_until_ready()

    def carrier_admissible(output):
        return jnp.asarray(_admissible(operator, output))

    corrected_result = dense_newton(
        exact_map,
        carrier,
        initial_coefficients,
        steps=NEWTON_STEPS,
        admissible=carrier_admissible,
        factors=FACTORS,
        external=external,
    )
    corrected_result.exact_output.block_until_ready()

    projected_comparison = _angle_and_norm(
        projected_exact_step, carrier.expand(projected_step)
    )
    corrected_comparison = _angle_and_norm(
        projected_exact_step, carrier.expand(corrected_step)
    )
    exact_column_comparison = _column_comparison(
        traced_exact_jacobian, finite_exact_jacobian
    )
    projected_column_comparison = _column_comparison(
        traced_projected_jacobian, finite_projected_jacobian
    )
    expanded_projected_comparison = _column_comparison(
        expanded_projected_jacobian, finite_exact_jacobian
    )

    advance_receipt = json.loads(ADVANCE_RECEIPT.read_text(encoding="utf-8"))
    banked_stalled = advance_receipt["attempts"][-1][
        "dense_coefficient_step_against_projected_exact_step"
    ]
    if not np.isclose(
        projected_comparison["angle_degrees"],
        banked_stalled["angle_degrees"],
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("projected-residual step angle changed from its bank")
    if not np.isclose(
        projected_comparison["compared_to_reference_norm_ratio"],
        banked_stalled["compared_to_reference_norm_ratio"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("projected-residual step norm ratio changed from its bank")

    exact_matches_finite_difference = (
        exact_column_comparison["maximum_relative_column_error"] <= 2.0e-6
        and abs(exact_column_comparison["fitted_scale_factor"] - 1.0) <= 2.0e-6
    )
    projected_matches_own_finite_difference = (
        projected_column_comparison["maximum_relative_column_error"] <= 2.0e-6
        and abs(projected_column_comparison["fitted_scale_factor"] - 1.0) <= 2.0e-6
    )
    restored_direction = corrected_comparison["angle_degrees"] < projected_comparison[
        "angle_degrees"
    ] and abs(np.log(corrected_comparison["compared_to_reference_norm_ratio"])) < abs(
        np.log(projected_comparison["compared_to_reference_norm_ratio"])
    )
    converged = float(corrected_result.exact_residual) <= 1.0e-10
    if not exact_matches_finite_difference:
        raise RuntimeError("exact-residual Jacobian failed its finite-difference check")
    if not projected_matches_own_finite_difference:
        raise RuntimeError(
            "projected-residual Jacobian failed its finite-difference check"
        )

    return {
        "artifact": str(OUTPUT.relative_to(HERE)),
        "schema": "carrier-step-attribution-1",
        "source_commit": _source_commit(),
        "measurement_scope": {
            "frame": "closed-form-oracle-coarse",
            "realised_plasma_cells": len(machine.node),
            "exact_state_values": root.size,
            "coefficient_count": carrier.coefficient_count,
            "knots_per_axis": KNOTS_PER_AXIS,
            "platform": jax.devices()[0].platform,
            "precision": "float64",
            "finite_difference_step_rule": (
                "binary64_epsilon_to_one_third_times_maximum_of_one_and_basis_coefficient_magnitude"
            ),
            "finite_difference_step_range": [
                min(finite_difference_steps),
                max(finite_difference_steps),
            ],
        },
        "evidence_inputs": {
            "root_bank": str(ROOT_BANK.relative_to(HERE)),
            "root_bank_sha256": _sha256(ROOT_BANK),
            "root_receipt": str(ROOT_RECEIPT.relative_to(HERE)),
            "root_receipt_sha256": _sha256(ROOT_RECEIPT),
            "carrier_receipt": str(CARRIER_RECEIPT.relative_to(HERE)),
            "carrier_receipt_sha256": _sha256(CARRIER_RECEIPT),
            "advance_receipt": str(ADVANCE_RECEIPT.relative_to(HERE)),
            "advance_receipt_sha256": _sha256(ADVANCE_RECEIPT),
            "banked_exact_arm_terminal_residual": BANKED_EXACT_ARM_RESIDUAL,
            "banked_carrier_terminal_residual": BANKED_STALLED_RESIDUAL,
            "banked_step_angle_degrees": BANKED_STEP_ANGLE_DEGREES,
            "banked_step_norm_ratio": BANKED_STEP_NORM_RATIO,
        },
        "residual_identity_demonstration": {
            "exact_residual": {
                "definition": "exact_map(total_state) minus total_state",
                "dimension": carrier.exact_size,
                "traced_jacobian_against_finite_difference": (exact_column_comparison),
            },
            "projected_residual": {
                "definition": (
                    "project(exact_map(total_state) minus external) minus coefficients"
                ),
                "dimension": carrier.coefficient_count,
                "traced_jacobian_against_its_finite_difference": (
                    projected_column_comparison
                ),
                "expanded_traced_jacobian_against_exact_residual_finite_difference": (
                    expanded_projected_comparison
                ),
            },
            "attribution": "PROJECTED_RESIDUAL_DIFFERENTIATED",
            "basis": (
                "The square dense Jacobian matches finite differences of the "
                "projected residual, while its exact-space expansion does not "
                "match finite differences of the exact residual. The rectangular "
                "exact-residual Jacobian matches all exact-residual basis "
                "derivatives."
            ),
        },
        "step_comparison": {
            "projected_residual_formulation": projected_comparison,
            "banked_projected_residual_formulation": {
                "angle_degrees": BANKED_STEP_ANGLE_DEGREES,
                "compared_to_reference_norm_ratio": BANKED_STEP_NORM_RATIO,
            },
            "exact_residual_least_squares_formulation": corrected_comparison,
            "corrected_step_utility_ladder": _step_utility(
                exact_map,
                operator,
                carrier,
                external,
                stalled_coefficients,
                corrected_step,
            ),
            "restored_direction": restored_direction,
        },
        "trajectory_comparison": {
            "projected_residual_formulation": projected_trajectory,
            "exact_residual_least_squares_formulation": {
                "terminal_exact_relative_residual": float(
                    corrected_result.exact_residual
                ),
                "trace": np.asarray(corrected_result.trace, dtype=np.float64).tolist(),
                "admitted_advances": corrected_result.admitted_advances,
                "newton_step_equivalents": corrected_result.newton_step_equivalents,
                "terminal_topology_class": boundary_mode(
                    operator.read(corrected_result.exact_output)[1]
                ).value,
            },
            "comparators": {
                "exact_value_arm_terminal_residual": BANKED_EXACT_ARM_RESIDUAL,
                "stalled_projected_residual_carrier": BANKED_STALLED_RESIDUAL,
            },
        },
        "verdict": {
            "classification": (
                "CARRIER_QUESTION_CLOSED_CONDITIONING_WON_CONVERGENCE_LOST"
                if restored_direction and not converged
                else "CORRECTION_REQUIRES_FOLLOW_ON"
            ),
            "projected_residual_was_differentiated": True,
            "exact_residual_jacobian_matches_finite_difference": (
                exact_matches_finite_difference
            ),
            "correction_restores_direction": restored_direction,
            "corrected_carrier_converged": converged,
            "production_route": (
                "exact_values"
                if not converged
                else "carrier_requires_separate_promotion_decision"
            ),
        },
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
    print(
        json.dumps(receipt["residual_identity_demonstration"], sort_keys=True),
        flush=True,
    )
    print(json.dumps(receipt["step_comparison"], sort_keys=True), flush=True)
    print(json.dumps(receipt["trajectory_comparison"], sort_keys=True), flush=True)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
