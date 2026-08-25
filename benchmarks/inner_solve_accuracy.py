"""Measure inner Krylov accuracy on cycling and converging MAST references.

The benchmark replays the established unpenalised merit-ranked Newton ladder
at several GMRES dimensions.  It then forms the complete residual Jacobian at
the retained twelve-dimensional-Krylov terminal and compares that GMRES action
with the dense Newton direction.
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
from benchmarks.contraction_discriminator import (
    BANKED_CONTRAST,
    REPRODUCTION_ABSOLUTE_TOLERANCE,
    _banked_rows,
)
from benchmarks.diiid_forward_gs_match import _margin_penalty
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
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
DEFAULT_OUTPUT = (
    HERE / "docs/figures/discrete-operator-analytic-error/inner-solve-accuracy.json"
)
TARGET_REFERENCES = ((21978, 35), (22086, 43))
KRYLOV_BUDGETS = (12, 30, 60, 120)
ALTERNATION_IDENTITY_TOLERANCE = 5.0e-8
MEASURED_SLICE_SECONDS = 42.77e-3
CATALOG_SLICE_COUNT = 1_341_435


class RetainedIteration(NamedTuple):
    """Fixed-ladder result with promoted states and residuals retained."""

    state: jax.Array
    residuals: jax.Array
    states: jax.Array


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_identity(left: jax.Array, right: jax.Array) -> float:
    """Return symmetric relative sup distance between two states."""

    scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(left)), jnp.max(jnp.abs(right))),
        jnp.asarray(1.0e-30),
    )
    return float(jnp.max(jnp.abs(left - right)) / scale)


def _replay_iteration(
    map_fn, margin_fn, initial: jax.Array, gmres_iterations: int
) -> RetainedIteration:
    """Replay the banked merit-ranked ladder at one Krylov dimension."""

    state = fixed_point_solver._solver_state(initial, Precision.AUTOMATIC)
    factors = jnp.asarray(fixed_point_solver._BACKTRACKING_FACTORS, dtype=state.dtype)
    residuals = jnp.full(NEWTON_STEPS, jnp.nan, dtype=state.dtype)
    recent = jnp.full(len(fixed_point_solver._BACKTRACKING_FACTORS), jnp.nan)
    states = (
        jnp.zeros((NEWTON_STEPS + 1, state.size), dtype=state.dtype).at[0].set(state)
    )

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
        current, trace, recent_grades, state_trace, condition_baseline = carry
        mapped, tangent = jax.linearize(map_fn, current)
        residual_vector = mapped - current
        current_residual = fixed_point_solver._relative_residual(mapped, current)
        current_grade = current_residual + _margin_penalty(margin_fn(current))
        qualified = fixed_point_solver._qualified_krylov_step(
            lambda vector: vector - tangent(vector),
            residual_vector,
            current_residual,
            gmres_iterations=gmres_iterations,
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
        grades = jnp.where(use_conditioned, conditioned[3], raw[3])
        selected = jnp.where(use_conditioned, conditioned_best, raw[5])
        any_usable = raw_usable | use_conditioned
        proposal = jnp.where(any_usable, candidates[selected], current)
        accepted_residual = jnp.where(
            any_usable, candidate_residuals[selected], current_residual
        )
        accepted_grade = jnp.where(any_usable, grades[selected], current_grade)
        trace = trace.at[index].set(accepted_residual)
        recent_grades = recent_grades.at[jnp.mod(index, recent_grades.size)].set(
            accepted_grade
        )
        state_trace = state_trace.at[index + 1].set(proposal)
        return (
            proposal,
            trace,
            recent_grades,
            state_trace,
            qualified.condition_baseline,
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
            jnp.asarray(jnp.nan, dtype=state.dtype),
        ),
    )
    return RetainedIteration(state=result[0], residuals=result[1], states=result[3])


def _alternation_diagnostics(replay: RetainedIteration) -> dict[str, Any]:
    """Measure whether the last four promotions repeat by parity."""

    same_first = _state_identity(replay.states[-1], replay.states[-3])
    same_second = _state_identity(replay.states[-2], replay.states[-4])
    opposite = _state_identity(replay.states[-1], replay.states[-2])
    persists = bool(
        max(same_first, same_second) <= ALTERNATION_IDENTITY_TOLERANCE
        and opposite > ALTERNATION_IDENTITY_TOLERANCE
    )
    return {
        "identity_tolerance": ALTERNATION_IDENTITY_TOLERANCE,
        "same_parity_relative_sup": [same_first, same_second],
        "opposite_parity_relative_sup": opposite,
        "period_two_alternation_persists": persists,
    }


def _direction_summary(step: jax.Array) -> dict[str, Any]:
    """Serialize one direction without banking thousands of redundant values."""

    values = np.ascontiguousarray(np.asarray(step, dtype=np.float64))
    return {
        "size": int(values.size),
        "l2_norm_wb": float(np.linalg.norm(values)),
        "sup_norm_wb": float(np.max(np.abs(values))),
        "mean_wb": float(np.mean(values)),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _dense_newton_diagnostics(map_fn, terminal: jax.Array) -> dict[str, Any]:
    """Form the full residual Jacobian and compare exact and Krylov directions."""

    mapped = map_fn(terminal)
    right_hand_side = mapped - terminal
    jacobian = jax.jacfwd(lambda state: state - map_fn(state))(terminal)
    true_step = jnp.linalg.solve(jacobian, right_hand_side)
    gmres_step, gmres_info = jax.scipy.sparse.linalg.gmres(
        lambda vector: jacobian @ vector,
        right_hand_side,
        maxiter=12,
        restart=12,
        solve_method="batched",
    )
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    condition = singular_values[0] / singular_values[-1]
    cosine = jnp.dot(true_step, gmres_step) / jnp.maximum(
        jnp.linalg.norm(true_step) * jnp.linalg.norm(gmres_step),
        jnp.finfo(terminal.dtype).tiny,
    )
    angle = jnp.degrees(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))
    true_linear_residual = jnp.linalg.norm(
        jacobian @ true_step - right_hand_side
    ) / jnp.maximum(jnp.linalg.norm(right_hand_side), jnp.finfo(terminal.dtype).tiny)
    gmres_linear_residual = jnp.linalg.norm(
        jacobian @ gmres_step - right_hand_side
    ) / jnp.maximum(jnp.linalg.norm(right_hand_side), jnp.finfo(terminal.dtype).tiny)
    exact_trial = terminal + true_step
    gmres_trial = terminal + gmres_step
    values = jax.device_get(
        (
            condition,
            singular_values[0],
            singular_values[-1],
            angle,
            true_linear_residual,
            gmres_linear_residual,
            fixed_point_solver._relative_residual(map_fn(exact_trial), exact_trial),
            fixed_point_solver._relative_residual(map_fn(gmres_trial), gmres_trial),
            true_step,
            gmres_step,
        )
    )
    return {
        "state_size": int(terminal.size),
        "dense_jacobian_shape": [int(value) for value in jacobian.shape],
        "dense_jacobian_bytes": int(jacobian.size * jacobian.dtype.itemsize),
        "condition_number_2_norm": float(values[0]),
        "largest_singular_value": float(values[1]),
        "smallest_singular_value": float(values[2]),
        "true_newton_step": _direction_summary(values[8]),
        "gmres_12_step": _direction_summary(values[9]),
        "angle_degrees_true_newton_to_gmres_12": float(values[3]),
        "true_step_relative_linear_residual_l2": float(values[4]),
        "gmres_12_relative_linear_residual_l2": float(values[5]),
        "true_step_full_trial_nonlinear_residual": float(values[6]),
        "gmres_12_full_trial_nonlinear_residual": float(values[7]),
        "gmres_info": int(gmres_info),
    }


def _cost_contract() -> dict[str, Any]:
    """Report the fixed-shape forward-action cost of every Krylov option."""

    rows = []
    for budget in KRYLOV_BUDGETS:
        actions = budget + 2
        catalog_seconds = CATALOG_SLICE_COUNT * MEASURED_SLICE_SECONDS * actions
        rows.append(
            {
                "gmres_iterations": budget,
                "forward_action_equivalents_per_outer_step": actions,
                "seconds_per_slice_outer_step_at_measured_rate": (
                    MEASURED_SLICE_SECONDS * actions
                ),
                "catalog_hours_per_outer_step_at_measured_rate": (
                    catalog_seconds / 3600.0
                ),
            }
        )
    return {
        "accounting": (
            "production fixed-shape trace contract: one linearisation value, "
            "one tangent action per GMRES iteration, and one promotion read"
        ),
        "measured_seconds_per_slice": MEASURED_SLICE_SECONDS,
        "catalog_slice_count": CATALOG_SLICE_COUNT,
        "options": rows,
    }


def run(
    *,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    banked_contrast: Path = BANKED_CONTRAST,
    output: Path = DEFAULT_OUTPUT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Run the budget sweep and dense terminal comparisons."""

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
        sweeps = []
        terminal_for_dense = None
        for budget in KRYLOV_BUDGETS:
            replay = _replay_iteration(
                mapped, profile.operator.topology_margin, seed, budget
            )
            replay.state.block_until_ready()
            sequence = np.asarray(replay.residuals, dtype=np.float64)
            terminal = float(
                fixed_point_solver._relative_residual(
                    mapped(replay.state), replay.state
                )
            )
            row = {
                "gmres_iterations": budget,
                "residual_sequence": sequence.tolist(),
                "terminal_residual": terminal,
                "alternation": _alternation_diagnostics(replay),
            }
            if budget == 12:
                expected = np.asarray(
                    banked[key]["mixed_arm"]["residual_sequence"], dtype=np.float64
                )
                maximum_difference = float(
                    max(
                        np.max(np.abs(sequence - expected)),
                        abs(
                            terminal
                            - float(banked[key]["mixed_arm"]["terminal_residual"])
                        ),
                    )
                )
                if maximum_difference > REPRODUCTION_ABSOLUTE_TOLERANCE:
                    raise AssertionError(
                        f"{key} did not reproduce the banked terminal: "
                        f"{maximum_difference}"
                    )
                row["banked_reproduction"] = {
                    "passes": True,
                    "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
                    "maximum_absolute_difference": maximum_difference,
                }
                terminal_for_dense = replay.state
            sweeps.append(row)
        if terminal_for_dense is None:
            raise RuntimeError("the dense-comparison terminal was not retained")
        records.append(
            {
                "reference": {"shot": key[0], "slice_index": key[1]},
                "budget_sweep": sweeps,
                "dense_terminal_comparison": _dense_newton_diagnostics(
                    mapped, terminal_for_dense
                ),
            }
        )

    cycling = records[0]
    converged = records[1]
    cycling_angle = cycling["dense_terminal_comparison"][
        "angle_degrees_true_newton_to_gmres_12"
    ]
    converged_angle = converged["dense_terminal_comparison"][
        "angle_degrees_true_newton_to_gmres_12"
    ]
    cycling_dense = cycling["dense_terminal_comparison"]
    cycling_alternates = {
        row["gmres_iterations"]: row["alternation"]["period_two_alternation_persists"]
        for row in cycling["budget_sweep"]
    }
    conclusion = (
        "inner_solve_accuracy_does_not_explain_cycle"
        if all(cycling_alternates.values())
        and cycling_dense["true_step_full_trial_nonlinear_residual"]
        > cycling["budget_sweep"][0]["terminal_residual"]
        else "inner_solve_accuracy_remains_unresolved"
    )
    receipt = {
        "artifact": "inner Krylov accuracy discriminator on two MAST references",
        "source_commit": _source_revision(),
        "driver_sha256": _sha256(Path(__file__)),
        "evidence_inputs": {
            "banked_contrast": str(banked_contrast.relative_to(HERE)),
            "banked_contrast_sha256": _sha256(banked_contrast),
            "response_carrier": carrier_evidence,
        },
        "measurement_contract": {
            "references": [list(item) for item in TARGET_REFERENCES],
            "newton_promotions": NEWTON_STEPS,
            "gmres_iterations": list(KRYLOV_BUDGETS),
            "held_fixed": (
                "seed, fixed-point map, topology-margin function, target current, "
                "promotion count, merit ladder, conditioning policy, and precision"
            ),
            "dense_comparison_terminal": (
                "retained terminal of the reproduced twelve-iteration GMRES run"
            ),
        },
        "references": records,
        "cost": _cost_contract(),
        "verdict": {
            "name": conclusion,
            "cycling_angle_degrees": cycling_angle,
            "converged_angle_degrees": converged_angle,
            "cycling_true_step_full_trial_nonlinear_residual": cycling_dense[
                "true_step_full_trial_nonlinear_residual"
            ],
            "cycling_gmres_12_full_trial_nonlinear_residual": cycling_dense[
                "gmres_12_full_trial_nonlinear_residual"
            ],
            "cycling_alternation_by_gmres_iterations": cycling_alternates,
            "supports_inner_solve_accuracy_mechanism": bool(
                conclusion != "inner_solve_accuracy_does_not_explain_cycle"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the accuracy discriminator from the command line."""

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
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
