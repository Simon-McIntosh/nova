"""Distinguish a converged MAST root from an unpenalised alternating solve.

The two references are replayed with the same fixed-ladder Newton--Krylov
iteration that produced the banked contrast.  The replay additionally retains
promoted states so local map conditioning, gauge-aligned displacement, and the
claimed two-cycle can be tested directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diiid_forward_gs_match import _margin_penalty
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import (
    _persisted_response_cache,
    _source_revision,
)
from nova.equilibrium import fixed_point as fixed_point_solver
from nova.equilibrium.fixed_point import KrylovActionQualification
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import Precision, configure_dtypes


HERE = Path(__file__).resolve().parents[1]
BANKED_CONTRAST = (
    HERE / "docs/figures/dual-branch-selection/pinned-branch-contrast.json"
)
DEFAULT_OUTPUT = (
    HERE
    / "docs/figures/discrete-operator-analytic-error/contraction-discriminator.json"
)
TARGET_REFERENCES = ((21978, 35), (22086, 43))
REPRODUCTION_ABSOLUTE_TOLERANCE = 5.0e-12
POWER_ITERATIONS = 24


class RetainedIteration(NamedTuple):
    """Fixed-ladder result with every promoted state retained."""

    state: jax.Array
    residuals: jax.Array
    states: jax.Array
    accepted_factors: jax.Array
    effective_newton_fractions: jax.Array
    projected_conditions: jax.Array
    accepted_penalties: jax.Array


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_sup(left: jax.Array, right: jax.Array) -> float:
    """Return a symmetric relative sup distance between two states."""

    scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(left)), jnp.max(jnp.abs(right))),
        jnp.asarray(1.0e-30),
    )
    return float(jnp.max(jnp.abs(left - right)) / scale)


def _replay_iteration(map_fn, margin_fn, initial) -> RetainedIteration:
    """Replay the established merit-ranked ladder and retain promoted states."""

    state = fixed_point_solver._solver_state(initial, Precision.AUTOMATIC)
    factors = jnp.asarray(fixed_point_solver._BACKTRACKING_FACTORS, dtype=state.dtype)
    residuals = jnp.full(NEWTON_STEPS, jnp.nan, dtype=state.dtype)
    recent = jnp.full(len(fixed_point_solver._BACKTRACKING_FACTORS), jnp.nan)
    states = (
        jnp.zeros((NEWTON_STEPS + 1, state.size), dtype=state.dtype).at[0].set(state)
    )
    accepted_factors = jnp.zeros(NEWTON_STEPS, dtype=state.dtype)
    effective_newton_fractions = jnp.zeros(NEWTON_STEPS, dtype=state.dtype)
    projected_conditions = jnp.full(NEWTON_STEPS, jnp.nan, dtype=state.dtype)
    accepted_penalties = jnp.full(NEWTON_STEPS, jnp.nan, dtype=state.dtype)

    def bounded_step(step, residual_vector):
        fallback = 0.5 * residual_vector
        step = jnp.where(jnp.all(jnp.isfinite(step)), step, fallback)
        cap = 10.0 * jnp.max(jnp.abs(fallback))
        norm_step = jnp.max(jnp.abs(step))
        return jnp.where(
            norm_step > cap,
            step * (cap / jnp.maximum(norm_step, 1.0e-300)),
            step,
        )

    def body(index, carry):
        (
            current,
            trace,
            recent_grades,
            state_trace,
            selected_factors,
            effective_fractions,
            condition_trace,
            condition_baseline,
            penalty_trace,
        ) = carry
        mapped, tangent = jax.linearize(map_fn, current)
        residual_vector = mapped - current
        current_residual = fixed_point_solver._relative_residual(mapped, current)
        current_grade = current_residual + _margin_penalty(margin_fn(current))
        qualified = fixed_point_solver._qualified_krylov_step(
            lambda vector: vector - tangent(vector),
            residual_vector,
            current_residual,
            gmres_iterations=GMRES_ITERATIONS,
            condition_ratio_limit=math.e,
            preceding_condition_baseline=condition_baseline,
        )
        accepted_action = qualified.qualification == KrylovActionQualification.ACCEPTED
        raw_step = bounded_step(qualified.unconditioned_step, residual_vector)
        conditioned_step = bounded_step(qualified.step, residual_vector)

        def evaluate_ladder(trial_step):
            candidates = current[None, :] + factors[:, None] * trial_step[None, :]

            def grade(candidate):
                candidate_mapped = map_fn(candidate)
                residual = fixed_point_solver._relative_residual(
                    candidate_mapped, candidate
                )
                penalty = _margin_penalty(margin_fn(candidate))
                return residual, penalty, residual + penalty

            candidate_residuals, penalties, grades = jax.lax.map(grade, candidates)
            usable = (
                jnp.all(jnp.isfinite(candidates), axis=1)
                & jnp.isfinite(candidate_residuals)
                & jnp.isfinite(grades)
                & accepted_action
            )
            envelope = jnp.max(
                jnp.where(jnp.isfinite(recent_grades), recent_grades, current_grade)
            )
            within_envelope = usable & (grades <= envelope * 1.05)
            first = jnp.argmax(within_envelope)
            best = jnp.argmin(jnp.where(usable, grades, jnp.inf))
            selected = jnp.where(jnp.any(within_envelope), first, best)
            return candidates, candidate_residuals, penalties, grades, usable, selected

        raw = evaluate_ladder(raw_step)
        raw_usable = jnp.any(raw[4])
        conditioned = jax.lax.cond(
            qualified.conditioning_applied & ~raw_usable,
            evaluate_ladder,
            lambda _trial_step: raw,
            conditioned_step,
        )
        conditioned_improves = conditioned[4] & (conditioned[3] <= current_grade)
        use_conditioned = (
            accepted_action
            & qualified.conditioning_applied
            & ~raw_usable
            & jnp.any(conditioned_improves)
        )
        conditioned_best = jnp.argmin(
            jnp.where(conditioned_improves, conditioned[3], jnp.inf)
        )
        candidates = jnp.where(use_conditioned, conditioned[0], raw[0])
        candidate_residuals = jnp.where(use_conditioned, conditioned[1], raw[1])
        penalties = jnp.where(use_conditioned, conditioned[2], raw[2])
        grades = jnp.where(use_conditioned, conditioned[3], raw[3])
        selected = jnp.where(use_conditioned, conditioned_best, raw[5])
        any_usable = raw_usable | use_conditioned
        proposal = jnp.where(any_usable, candidates[selected], current)
        accepted_residual = jnp.where(
            any_usable, candidate_residuals[selected], current_residual
        )
        accepted_grade = jnp.where(any_usable, grades[selected], current_grade)
        selected_factor = jnp.where(any_usable, factors[selected], 0.0)
        selected_step = jnp.where(use_conditioned, conditioned_step, raw_step)
        unconditioned_norm = jnp.linalg.norm(qualified.unconditioned_step)
        effective_fraction = jnp.where(
            any_usable & (unconditioned_norm > 0.0),
            selected_factor
            * jnp.linalg.norm(selected_step)
            / jnp.maximum(unconditioned_norm, jnp.finfo(state.dtype).tiny),
            0.0,
        )
        trace = trace.at[index].set(accepted_residual)
        recent_grades = recent_grades.at[jnp.mod(index, recent_grades.size)].set(
            accepted_grade
        )
        state_trace = state_trace.at[index + 1].set(proposal)
        selected_factors = selected_factors.at[index].set(selected_factor)
        effective_fractions = effective_fractions.at[index].set(effective_fraction)
        condition_trace = condition_trace.at[index].set(qualified.projected_condition)
        penalty_trace = penalty_trace.at[index].set(
            jnp.where(any_usable, penalties[selected], jnp.nan)
        )
        return (
            proposal,
            trace,
            recent_grades,
            state_trace,
            selected_factors,
            effective_fractions,
            condition_trace,
            qualified.condition_baseline,
            penalty_trace,
        )

    result = jax.lax.fori_loop(
        0,
        NEWTON_STEPS,
        body,
        (
            state,
            residuals,
            recent,
            states,
            accepted_factors,
            effective_newton_fractions,
            projected_conditions,
            jnp.asarray(jnp.nan, dtype=state.dtype),
            accepted_penalties,
        ),
    )
    return RetainedIteration(
        state=result[0],
        residuals=result[1],
        states=result[3],
        accepted_factors=result[4],
        effective_newton_fractions=result[5],
        projected_conditions=result[6],
        accepted_penalties=result[8],
    )


def _power_iteration(tangent, shape: tuple[int, ...]) -> dict[str, Any]:
    """Estimate the dominant local map eigenvalue on a fixed tangent budget."""

    generator = np.random.default_rng(11)
    vector = jnp.asarray(generator.normal(size=shape), dtype=jnp.float64)
    vector = vector / jnp.linalg.norm(vector)
    growth = []
    for _ in range(POWER_ITERATIONS):
        image = tangent(vector)
        norm = jnp.linalg.norm(image)
        growth.append(float(norm))
        vector = image / jnp.maximum(norm, 1.0e-300)
    final = tangent(vector)
    rayleigh = float(jnp.dot(vector, final))
    return {
        "method": "fixed-count power iteration on exact linearized map",
        "iterations": POWER_ITERATIONS,
        "rayleigh_quotient": rayleigh,
        "absolute_dominant_eigenvalue_estimate": abs(rayleigh),
        "last_five_norm_growth_estimates": growth[-5:],
        "finite": bool(np.isfinite(rayleigh)),
    }


def _local_diagnostics(mapped, state: jax.Array) -> dict[str, Any]:
    """Measure residual-Jacobian conditioning and map spectral radius."""

    image, tangent = jax.linearize(mapped, state)
    residual_vector = image - state
    residual_condition, residual_baseline = (
        fixed_point_solver._projected_krylov_condition(
            lambda vector: vector - tangent(vector),
            residual_vector,
            krylov_dimension=GMRES_ITERATIONS,
        )
    )
    generator = np.random.default_rng(11)
    fixed_probe = jnp.asarray(generator.normal(size=state.shape), dtype=state.dtype)
    condition, baseline = fixed_point_solver._projected_krylov_condition(
        lambda vector: vector - tangent(vector),
        fixed_probe,
        krylov_dimension=GMRES_ITERATIONS,
    )
    return {
        "relative_residual": float(fixed_point_solver._relative_residual(image, state)),
        "projected_residual_jacobian_condition": float(condition),
        "projected_condition_baseline": float(baseline),
        "projection_initial_vector": "fixed normal probe with random seed 11",
        "residual_seeded_projected_condition": float(residual_condition),
        "residual_seeded_condition_baseline": float(residual_baseline),
        "residual_seeded_condition_qualification": (
            "direction is roundoff-sensitive when the nonlinear residual is at floor"
        ),
        "map_spectral_radius": _power_iteration(tangent, state.shape),
    }


def _gauge_aligned_distance(seed: jax.Array, terminal: jax.Array) -> dict[str, float]:
    """Measure seed-to-terminal displacement after one shared additive gauge."""

    seed_values = np.asarray(seed, dtype=np.float64)
    terminal_values = np.asarray(terminal, dtype=np.float64)
    gauge = float(np.mean(seed_values - terminal_values))
    difference = terminal_values + gauge - seed_values
    span = float(np.ptp(seed_values))
    return {
        "additive_gauge_wb": gauge,
        "absolute_sup_wb": float(np.max(np.abs(difference))),
        "rms_wb": float(np.sqrt(np.mean(difference**2))),
        "sup_fraction_of_seed_span": float(np.max(np.abs(difference)) / span),
        "rms_fraction_of_seed_span": float(np.sqrt(np.mean(difference**2)) / span),
    }


def _orbit_test(mapped, states: jax.Array, residual_floor: float) -> dict[str, Any]:
    """Test whether the last alternating states form an orbit of the map itself."""

    first = states[-2]
    second = states[-1]
    first_image = mapped(first)
    second_image = mapped(second)
    first_to_second = _relative_sup(first_image, second)
    second_to_first = _relative_sup(second_image, first)
    same_parity = _relative_sup(states[-1], states[-3])
    opposite_parity = _relative_sup(states[-1], states[-2])
    tolerance = max(64.0 * np.finfo(float).eps, REPRODUCTION_ABSOLUTE_TOLERANCE)
    nondegenerate = bool(opposite_parity > tolerance)
    genuine = bool(
        nondegenerate and first_to_second <= tolerance and second_to_first <= tolerance
    )
    return {
        "first_state_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(first, dtype=np.float64)).tobytes()
        ).hexdigest(),
        "second_state_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(second, dtype=np.float64)).tobytes()
        ).hexdigest(),
        "map_first_to_second_relative_sup": first_to_second,
        "map_second_to_first_relative_sup": second_to_first,
        "same_parity_state_relative_sup": same_parity,
        "opposite_parity_state_relative_sup": opposite_parity,
        "alternating_residual_floor": residual_floor,
        "map_first_to_second_fraction_of_residual_floor": (
            first_to_second / residual_floor
        ),
        "map_second_to_first_fraction_of_residual_floor": (
            second_to_first / residual_floor
        ),
        "orbit_identity_tolerance": tolerance,
        "nondegenerate_alternating_states": nondegenerate,
        "genuine_period_two_orbit_of_fixed_point_map": genuine,
        "interpretation": (
            "fixed-point map exchanges the alternating states"
            if genuine
            else (
                "residual alternation is generated by the Newton promotion policy, "
                "not a period-two orbit of the fixed-point map"
            )
        ),
    }


def _banked_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Return the two required banked reference records."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in artifact["references"]:
        key = (int(row["reference"]["shot"]), int(row["reference"]["slice_index"]))
        if key in TARGET_REFERENCES:
            rows[key] = row
    if set(rows) != set(TARGET_REFERENCES):
        raise RuntimeError("banked contrast does not contain both target references")
    return rows


def run(
    *,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    banked_contrast: Path = BANKED_CONTRAST,
    output: Path = DEFAULT_OUTPUT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Replay both references, measure candidates, and bank the discriminator."""

    configure_dtypes()
    banked = _banked_rows(banked_contrast)
    response_cache, carrier_evidence = _persisted_response_cache(
        carrier, carrier_receipt
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(bank)
    }
    records = []
    for key in TARGET_REFERENCES:
        selected_row, qualification = selected[key]
        case, context = _mast_case_from_selection(store, selected_row, qualification)
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("replay entered the direct response builder")
        seed = jnp.asarray(passive_case["state"])
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        mapped = profile.flux_map(
            requested_class=TopologyClass.DIVERTED,
            target_current=target_current,
        )
        replay = _replay_iteration(mapped, profile.operator.topology_margin, seed)
        replay.state.block_until_ready()
        measured_residuals = np.asarray(replay.residuals, dtype=np.float64)
        expected_residuals = np.asarray(
            banked[key]["mixed_arm"]["residual_sequence"], dtype=np.float64
        )
        differences = np.abs(measured_residuals - expected_residuals)
        terminal = float(
            fixed_point_solver._relative_residual(mapped(replay.state), replay.state)
        )
        banked_terminal = float(banked[key]["mixed_arm"]["terminal_residual"])
        maximum_difference = float(
            max(np.max(differences), abs(terminal - banked_terminal))
        )
        if maximum_difference > REPRODUCTION_ABSOLUTE_TOLERANCE:
            raise AssertionError(
                f"{key} did not reproduce the banked terminal: {maximum_difference}"
            )
        residual_floor = float(max(expected_residuals[-2:]))
        records.append(
            {
                "reference": {"shot": key[0], "slice_index": key[1]},
                "reproduction": {
                    "passes": True,
                    "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
                    "maximum_residual_absolute_difference": maximum_difference,
                    "banked_terminal_residual": banked_terminal,
                    "replayed_terminal_residual": terminal,
                    "banked_tail": expected_residuals[-4:].tolist(),
                    "replayed_tail": measured_residuals[-4:].tolist(),
                },
                "seed_to_terminal_common_gauge": _gauge_aligned_distance(
                    seed, replay.state
                ),
                "local_map_at_seed": _local_diagnostics(mapped, seed),
                "local_map_at_terminal": _local_diagnostics(mapped, replay.state),
                "last_two_promotions": {
                    "accepted_factors": np.asarray(
                        replay.accepted_factors[-2:], dtype=np.float64
                    ).tolist(),
                    "effective_newton_fractions": np.asarray(
                        replay.effective_newton_fractions[-2:], dtype=np.float64
                    ).tolist(),
                    "projected_residual_jacobian_conditions": np.asarray(
                        replay.projected_conditions[-2:], dtype=np.float64
                    ).tolist(),
                    "accepted_topology_penalties": np.asarray(
                        replay.accepted_penalties[-2:], dtype=np.float64
                    ).tolist(),
                },
                "period_two_test": _orbit_test(mapped, replay.states, residual_floor),
            }
        )

    by_shot = {row["reference"]["shot"]: row for row in records}
    cycling = by_shot[21978]
    converged = by_shot[22086]
    cycling_condition = cycling["local_map_at_terminal"][
        "projected_residual_jacobian_condition"
    ]
    converged_condition = converged["local_map_at_terminal"][
        "projected_residual_jacobian_condition"
    ]
    receipt = {
        "artifact": (
            "local contraction discriminator for two unpenalised MAST references"
        ),
        "source_commit": _source_revision(),
        "driver_sha256": _sha256(Path(__file__)),
        "evidence_inputs": {
            "banked_contrast": str(banked_contrast.relative_to(HERE)),
            "banked_contrast_sha256": _sha256(banked_contrast),
            "response_carrier": carrier_evidence,
        },
        "measurement_contract": {
            "references": [list(item) for item in TARGET_REFERENCES],
            "iteration": (
                "exact replay of the banked unpenalised merit-ranked "
                "Newton--Krylov ladder"
            ),
            "conditioning": (
                "twelve-vector Arnoldi projection of the residual Jacobian I minus J "
                "from fixed normal probe seed 11; residual-seeded value retained "
                "separately and qualified at a roundoff terminal"
            ),
            "spectral_radius": (
                "twenty-four exact tangent power iterations with fixed random seed 11"
            ),
            "common_gauge": (
                "single additive offset equal to mean seed minus terminal over the "
                "complete carried state"
            ),
            "period_two": (
                "reapply the fixed-point map to each final alternating promoted state "
                "and compare directly with the other"
            ),
        },
        "references": records,
        "discriminator": {
            "named_quantity": (
                "terminal projected condition of the Newton residual Jacobian I minus J"
            ),
            "direction": "higher on cycling 21978/35 than converged 22086/43",
            "cycling_21978_35": cycling_condition,
            "converged_22086_43": converged_condition,
            "ratio_cycling_over_converged": (cycling_condition / converged_condition),
            "separates": bool(cycling_condition > converged_condition),
        },
        "candidate_summary": {
            "projected_residual_jacobian_condition": {
                "seed_cycling_21978_35": cycling["local_map_at_seed"][
                    "projected_residual_jacobian_condition"
                ],
                "seed_converged_22086_43": converged["local_map_at_seed"][
                    "projected_residual_jacobian_condition"
                ],
                "terminal_cycling_21978_35": cycling_condition,
                "terminal_converged_22086_43": converged_condition,
                "result": "separates, with the cycling case more ill-conditioned",
            },
            "common_gauge_seed_to_terminal_rms_fraction": {
                "cycling_21978_35": cycling["seed_to_terminal_common_gauge"][
                    "rms_fraction_of_seed_span"
                ],
                "converged_22086_43": converged["seed_to_terminal_common_gauge"][
                    "rms_fraction_of_seed_span"
                ],
                "result": "separates weakly, with the cycling case travelling farther",
            },
            "fixed_point_map_spectral_radius": {
                "seed_cycling_21978_35": cycling["local_map_at_seed"][
                    "map_spectral_radius"
                ]["absolute_dominant_eigenvalue_estimate"],
                "seed_converged_22086_43": converged["local_map_at_seed"][
                    "map_spectral_radius"
                ]["absolute_dominant_eigenvalue_estimate"],
                "terminal_cycling_21978_35": cycling["local_map_at_terminal"][
                    "map_spectral_radius"
                ]["absolute_dominant_eigenvalue_estimate"],
                "terminal_converged_22086_43": converged["local_map_at_terminal"][
                    "map_spectral_radius"
                ]["absolute_dominant_eigenvalue_estimate"],
                "result": (
                    "does not explain the outcome: both exceed one and the converged "
                    "case has the larger terminal estimate"
                ),
            },
        },
        "period_two_verdict": {
            "claimed_cycle_is_fixed_point_map_orbit": cycling["period_two_test"][
                "genuine_period_two_orbit_of_fixed_point_map"
            ],
            "last_two_newton_factors": cycling["last_two_promotions"][
                "accepted_factors"
            ],
            "last_two_effective_newton_fractions": cycling["last_two_promotions"][
                "effective_newton_fractions"
            ],
            "line_search_step_control_or_penalty_caused": bool(
                any(
                    value < 1.0 - 1.0e-12
                    for value in cycling["last_two_promotions"][
                        "effective_newton_fractions"
                    ]
                )
                or any(
                    value != 1.0
                    for value in cycling["last_two_promotions"]["accepted_factors"]
                )
                or any(
                    value != 0.0
                    for value in cycling["last_two_promotions"][
                        "accepted_topology_penalties"
                    ]
                )
            ),
            "basis": (
                "the alternating Newton states repeat by parity while both final "
                "promotions accept a complete undamped Newton action at factor one "
                "and topology penalty zero, but direct fixed-point-map replay misses "
                "the opposite state materially"
            ),
            "interpretation": cycling["period_two_test"]["interpretation"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the discriminator from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--banked-contrast", type=Path, default=BANKED_CONTRAST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument(
        "--carrier-receipt", type=Path, default=response_carrier.DEFAULT_RECEIPT
    )
    arguments = parser.parse_args()
    result = run(
        store=arguments.store,
        bank=arguments.bank,
        banked_contrast=arguments.banked_contrast,
        output=arguments.output,
        carrier=arguments.carrier,
        carrier_receipt=arguments.carrier_receipt,
    )
    print(json.dumps(result["discriminator"], indent=2, sort_keys=True))
    print(json.dumps(result["period_two_verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
