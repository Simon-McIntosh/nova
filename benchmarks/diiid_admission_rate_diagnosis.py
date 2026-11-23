"""Diagnose low trial admission on score-blind DIII-D frames.

The production trajectory is replayed with the established constrained profile,
current completion, diverted seed, exact tangent, Krylov qualification, and
four-factor nonmonotone selection.  The replay records the mutually exclusive
reason for every refused trial without changing which state is promoted.

A finer diagnostic ladder is evaluated on the lowest-admission frame only when
the production ladder offers no admissible trial.  Those observations never
participate in promotion and therefore cannot change the measured trajectory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
from nova.equilibrium import fixed_point as fixed_point_solver
from nova.equilibrium.fixed_point import KrylovActionQualification
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_DATA = repaired_solve.DEFAULT_DATA
DEFAULT_OUTPUT = (
    HERE / "docs/figures/diiid-forward-onboarding/admission-rate-diagnosis.json"
)
REPAIRED_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "repaired-solve-five-frame-remeasure.json"
)
FAVOURABLE_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/"
    "topology-qualified-mesh-convergence.json"
)
REQUESTED_CLASS = TopologyClass.DIVERTED
FIXED_FACTORS = repaired_solve.NONMONOTONE_FACTORS
FINE_FACTORS = tuple(2.0**-power for power in range(4, 17))
PROBE_CASE = ("d3d_shot_002495e835.parquet", 146)

ADMITTED = 0
LEFT_TOPOLOGY_CLASS = 1
NONFINITE_CANDIDATE_STATE = 2
NONFINITE_MAPPED_STATE_OR_RESIDUAL = 3
CALLER_PREDICATE_FAILED = 4
KRYLOV_ACTION_FAILED = 5
NOT_PROBED = 6

REASON_NAMES = {
    ADMITTED: "ADMITTED",
    LEFT_TOPOLOGY_CLASS: "LEFT_TOPOLOGY_CLASS",
    NONFINITE_CANDIDATE_STATE: "NONFINITE_CANDIDATE_STATE",
    NONFINITE_MAPPED_STATE_OR_RESIDUAL: "NONFINITE_MAPPED_STATE_OR_RESIDUAL",
    CALLER_PREDICATE_FAILED: "CALLER_PREDICATE_FAILED",
    KRYLOV_ACTION_FAILED: "KRYLOV_ACTION_FAILED",
    NOT_PROBED: "NOT_PROBED",
}


class DiagnosticTrajectory(NamedTuple):
    """Fixed-shape production replay plus non-promoting trial observations."""

    state: jax.Array
    residual: jax.Array
    accepted_factors: jax.Array
    fixed_reasons: jax.Array
    krylov_qualifications: jax.Array
    fine_reasons: jax.Array
    fine_scores: jax.Array
    fine_step_norms: jax.Array
    largest_fine_admissible_factors: jax.Array


def _source_commit() -> str:
    """Return the checked-out source identity used by the measurement."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_observations(map_fn, profile, candidates):
    """Measure finiteness, residual, topology, and the caller predicate."""

    def observe(candidate):
        candidate_mapped = map_fn(candidate)
        score = fixed_point_solver._relative_residual(candidate_mapped, candidate)
        state_finite = jnp.all(jnp.isfinite(candidate))
        mapped_finite = jnp.all(jnp.isfinite(candidate_mapped)) & jnp.isfinite(score)
        _masks, topology = profile.operator.read(candidate)
        diverted = topology.diverted
        caller_admitted = state_finite & diverted
        return score, state_finite, mapped_finite, diverted, caller_admitted

    return jax.lax.map(observe, candidates)


def _exclusive_reasons(
    state_finite,
    mapped_finite,
    diverted,
    caller_admitted,
    action_accepted,
):
    """Assign exactly one admission or refusal reason to every trial."""

    reasons = jnp.full(state_finite.shape, ADMITTED, dtype=jnp.int8)
    reasons = jnp.where(~caller_admitted, CALLER_PREDICATE_FAILED, reasons)
    reasons = jnp.where(~diverted, LEFT_TOPOLOGY_CLASS, reasons)
    reasons = jnp.where(~mapped_finite, NONFINITE_MAPPED_STATE_OR_RESIDUAL, reasons)
    reasons = jnp.where(~state_finite, NONFINITE_CANDIDATE_STATE, reasons)
    return jnp.where(action_accepted, reasons, KRYLOV_ACTION_FAILED)


def _bounded_step(step, residual_vector):
    """Apply the established finite fallback and relative step cap."""

    fallback = 0.5 * residual_vector
    step = jnp.where(jnp.all(jnp.isfinite(step)), step, fallback)
    cap = 10.0 * jnp.max(jnp.abs(fallback))
    norm_step = jnp.max(jnp.abs(step))
    return jnp.where(
        norm_step > cap,
        step * (cap / jnp.maximum(norm_step, 1.0e-300)),
        step,
    )


def _diagnose_trajectory(map_fn, profile, initial, *, probe_fine_ladder: bool):
    """Replay production selection while observing refusal mechanisms."""

    initial = fixed_point_solver._solver_state(
        initial, fixed_point_solver.Precision.AUTOMATIC
    )
    fixed_factors = jnp.asarray(FIXED_FACTORS, dtype=initial.dtype)
    fine_factors = jnp.asarray(FINE_FACTORS, dtype=initial.dtype)
    newton_steps = repaired_solve.NEWTON_STEPS
    gmres_iterations = repaired_solve.GMRES_ITERATIONS

    def body(index, carry):
        (
            state,
            residual,
            recent,
            accepted_factors,
            fixed_reasons,
            krylov_qualifications,
            fine_reasons,
            fine_scores,
            fine_step_norms,
            largest_fine_factors,
        ) = carry
        mapped, tangent = jax.linearize(map_fn, state)
        residual_vector = mapped - state
        current_residual = fixed_point_solver._relative_residual(mapped, state)
        qualified_step = fixed_point_solver._qualified_krylov_step(
            lambda vector: vector - tangent(vector),
            residual_vector,
            current_residual,
            gmres_iterations=gmres_iterations,
        )
        qualification = qualified_step.qualification
        action_accepted = qualification == KrylovActionQualification.ACCEPTED
        step = _bounded_step(qualified_step.step, residual_vector)
        trial_step = jnp.where(action_accepted, step, jnp.zeros_like(step))
        candidates = state[None, :] + fixed_factors[:, None] * trial_step[None, :]
        (
            scores,
            state_finite,
            mapped_finite,
            diverted,
            caller_admitted,
        ) = _candidate_observations(map_fn, profile, candidates)
        candidate_reasons = _exclusive_reasons(
            state_finite,
            mapped_finite,
            diverted,
            caller_admitted,
            action_accepted,
        )
        candidate_admitted = candidate_reasons == ADMITTED
        envelope = jnp.max(jnp.where(jnp.isfinite(recent), recent, current_residual))
        within_envelope = candidate_admitted & (scores <= envelope * (1.0 + 0.05))
        first = jnp.argmax(within_envelope)
        best_admissible = jnp.argmin(jnp.where(candidate_admitted, scores, jnp.inf))
        selected = jnp.where(jnp.any(within_envelope), first, best_admissible)
        any_admissible = jnp.any(candidate_admitted)
        proposal = jnp.where(any_admissible, candidates[selected], state)
        accepted_residual = jnp.where(
            any_admissible, scores[selected], current_residual
        )

        def observe_fine(_operand):
            fine_candidates = (
                state[None, :] + fine_factors[:, None] * trial_step[None, :]
            )
            (
                observed_scores,
                observed_state_finite,
                observed_mapped_finite,
                observed_diverted,
                observed_caller_admitted,
            ) = _candidate_observations(map_fn, profile, fine_candidates)
            observed_reasons = _exclusive_reasons(
                observed_state_finite,
                observed_mapped_finite,
                observed_diverted,
                observed_caller_admitted,
                action_accepted,
            )
            observed_admitted = observed_reasons == ADMITTED
            largest_index = jnp.argmax(observed_admitted)
            largest_factor = jnp.where(
                jnp.any(observed_admitted), fine_factors[largest_index], 0.0
            )
            step_norms = jnp.max(jnp.abs(fine_candidates - state[None, :]), axis=1)
            return observed_reasons, observed_scores, step_norms, largest_factor

        def skip_fine(_operand):
            return (
                jnp.full(fine_factors.shape, NOT_PROBED, dtype=jnp.int8),
                jnp.full(fine_factors.shape, jnp.nan, dtype=initial.dtype),
                jnp.full(fine_factors.shape, jnp.nan, dtype=initial.dtype),
                jnp.asarray(0.0, dtype=initial.dtype),
            )

        should_probe = (
            jnp.asarray(probe_fine_ladder) & ~any_admissible & action_accepted
        )
        (
            observed_fine_reasons,
            observed_fine_scores,
            observed_fine_step_norms,
            largest_fine_factor,
        ) = jax.lax.cond(should_probe, observe_fine, skip_fine, operand=None)

        recent = recent.at[jnp.mod(index, recent.size)].set(accepted_residual)
        accepted_factors = accepted_factors.at[index].set(
            jnp.where(any_admissible, fixed_factors[selected], 0.0)
        )
        fixed_reasons = fixed_reasons.at[index].set(candidate_reasons)
        krylov_qualifications = krylov_qualifications.at[index].set(qualification)
        fine_reasons = fine_reasons.at[index].set(observed_fine_reasons)
        fine_scores = fine_scores.at[index].set(observed_fine_scores)
        fine_step_norms = fine_step_norms.at[index].set(observed_fine_step_norms)
        largest_fine_factors = largest_fine_factors.at[index].set(largest_fine_factor)
        return (
            proposal,
            accepted_residual,
            recent,
            accepted_factors,
            fixed_reasons,
            krylov_qualifications,
            fine_reasons,
            fine_scores,
            fine_step_norms,
            largest_fine_factors,
        )

    result = jax.lax.fori_loop(
        0,
        newton_steps,
        body,
        (
            initial,
            jnp.asarray(jnp.inf, dtype=initial.dtype),
            jnp.full(len(FIXED_FACTORS), jnp.nan, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
            jnp.full((newton_steps, len(FIXED_FACTORS)), NOT_PROBED, dtype=jnp.int8),
            jnp.full(
                newton_steps,
                KrylovActionQualification.NOT_APPLICABLE,
                dtype=jnp.int32,
            ),
            jnp.full((newton_steps, len(FINE_FACTORS)), NOT_PROBED, dtype=jnp.int8),
            jnp.full((newton_steps, len(FINE_FACTORS)), jnp.nan, dtype=initial.dtype),
            jnp.full((newton_steps, len(FINE_FACTORS)), jnp.nan, dtype=initial.dtype),
            jnp.zeros(newton_steps, dtype=initial.dtype),
        ),
    )
    (
        state,
        residual,
        _recent,
        accepted_factors,
        fixed_reasons,
        krylov_qualifications,
        fine_reasons,
        fine_scores,
        fine_step_norms,
        largest_fine_factors,
    ) = result
    return DiagnosticTrajectory(
        state,
        residual,
        accepted_factors,
        fixed_reasons,
        krylov_qualifications,
        fine_reasons,
        fine_scores,
        fine_step_norms,
        largest_fine_factors,
    )


def _read_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate the two banked comparator receipts."""

    repaired = json.loads(REPAIRED_RECEIPT.read_text(encoding="utf-8"))
    favourable = json.loads(FAVOURABLE_RECEIPT.read_text(encoding="utf-8"))
    expected_admissions = [6, 6, 9, 4, 7]
    observed_admissions = [
        record["promoted_iteration_count"] for record in repaired["frame_records"]
    ]
    if observed_admissions != expected_admissions:
        raise RuntimeError("the banked score-blind admission counts changed")
    fine_solver = favourable["rungs"][1]["solver"]
    if (
        fine_solver["accepted_factor_counts"]["1.0"] != 87
        or fine_solver["terminal_relative_residual"] != 7.930534999195602e-5
    ):
        raise RuntimeError("the banked favourable native comparator changed")
    coarse_solver = favourable["rungs"][0]["solver"]
    if coarse_solver["accepted_factor_counts"]["1.0"] != 85:
        raise RuntimeError("the banked 85-of-89 full-step comparator changed")
    return repaired, favourable


def _reason_counts(reasons: np.ndarray) -> dict[str, int]:
    """Count the exclusive reason assigned to each observed trial."""

    counts = Counter(REASON_NAMES[int(reason)] for reason in reasons.ravel())
    return {name: int(counts.get(name, 0)) for name in REASON_NAMES.values()}


def _iteration_records(reasons: np.ndarray) -> list[dict[str, Any]]:
    """Make every fixed-ladder trial's reason explicit in the receipt."""

    return [
        {
            "iteration": iteration + 1,
            "reasons_by_factor": {
                str(factor): REASON_NAMES[int(reason)]
                for factor, reason in zip(FIXED_FACTORS, row, strict=True)
            },
        }
        for iteration, row in enumerate(reasons)
    ]


def _fine_probe_records(
    reasons: np.ndarray,
    scores: np.ndarray,
    step_norms: np.ndarray,
) -> list[dict[str, Any]]:
    """Report every diagnostic fine-ladder evaluation that actually ran."""

    records = []
    for iteration, (reason_row, score_row, norm_row) in enumerate(
        zip(reasons, scores, step_norms, strict=True)
    ):
        if np.all(reason_row == NOT_PROBED):
            continue
        records.append(
            {
                "iteration": iteration + 1,
                "trials": [
                    {
                        "factor": factor,
                        "reason": REASON_NAMES[int(reason)],
                        "relative_residual": float(score),
                        "state_step_sup_norm": float(norm),
                    }
                    for factor, reason, score, norm in zip(
                        FINE_FACTORS, reason_row, score_row, norm_row, strict=True
                    )
                ],
            }
        )
    return records


def _solve_frame(
    data: Path,
    case: repaired_solve.FrameCase,
    banked_record: dict[str, Any],
) -> dict[str, Any]:
    """Diagnose one fixed frame and assert trajectory identity."""

    started = perf_counter()
    row = repaired_solve._read_case(data / case.shot)
    profile, current, target_current_a, time_ms, seed = repaired_solve._prepare_frame(
        row, case.frame
    )
    mapped = profile.flux_map(jnp.asarray(current), REQUESTED_CLASS, target_current_a)
    probe_fine = (case.shot, case.frame) == PROBE_CASE
    trajectory = _diagnose_trajectory(
        mapped,
        profile,
        seed,
        probe_fine_ladder=probe_fine,
    )
    state = np.asarray(trajectory.state, dtype=float)
    terminal_mapped = np.asarray(mapped(trajectory.state), dtype=float)
    terminal_residual = float(
        np.max(np.abs(terminal_mapped - state))
        / max(np.max(np.abs(terminal_mapped)), 1.0e-30)
    )
    accepted_factors = np.asarray(trajectory.accepted_factors, dtype=float)
    fixed_reasons = np.asarray(trajectory.fixed_reasons, dtype=np.int8)
    qualification_codes = np.asarray(trajectory.krylov_qualifications, dtype=np.int32)
    fine_reasons = np.asarray(trajectory.fine_reasons, dtype=np.int8)
    fine_scores = np.asarray(trajectory.fine_scores, dtype=float)
    fine_step_norms = np.asarray(trajectory.fine_step_norms, dtype=float)
    largest_fine_factors = np.asarray(
        trajectory.largest_fine_admissible_factors, dtype=float
    )

    expected_counts = banked_record["accepted_factor_counts"]
    observed_counts = {
        str(factor): int(np.count_nonzero(accepted_factors == factor))
        for factor in (*FIXED_FACTORS, 0.0)
    }
    if observed_counts != expected_counts:
        raise RuntimeError(f"production trajectory changed for {case.shot}")
    if not np.isclose(
        terminal_residual,
        banked_record["terminal_relative_residual"],
        rtol=2.0e-12,
        atol=2.0e-14,
    ):
        raise RuntimeError(f"terminal residual changed for {case.shot}")
    admitted = fixed_reasons == ADMITTED
    if (
        int(np.count_nonzero(np.any(admitted, axis=1)))
        != banked_record["promoted_iteration_count"]
    ):
        raise RuntimeError(f"diagnostic admission count changed for {case.shot}")

    _masks, topology = profile.operator.read(trajectory.state)
    counts = _reason_counts(fixed_reasons)
    rejected_trials = fixed_reasons.size - counts["ADMITTED"]
    largest_fine = float(np.max(largest_fine_factors))
    return {
        "shot": case.shot,
        "frame": case.frame,
        "time_ms": time_ms,
        "target_current_a": float(target_current_a),
        "fixed_ladder": {
            "factors": list(FIXED_FACTORS),
            "offered_trial_count": int(fixed_reasons.size),
            "admitted_trial_count": counts["ADMITTED"],
            "rejected_trial_count": int(rejected_trials),
            "promoted_iteration_count": int(np.count_nonzero(accepted_factors)),
            "unpromoted_iteration_count": int(
                np.count_nonzero(accepted_factors == 0.0)
            ),
            "accepted_factor_counts": observed_counts,
            "exclusive_trial_reason_counts": counts,
            "reason_by_iteration": _iteration_records(fixed_reasons),
        },
        "krylov_qualification_counts": {
            qualification.name: int(
                np.count_nonzero(qualification_codes == int(qualification))
            )
            for qualification in KrylovActionQualification
        },
        "trajectory_identity": {
            "matches_banked_accepted_factor_counts": True,
            "banked_terminal_relative_residual": banked_record[
                "terminal_relative_residual"
            ],
            "diagnostic_terminal_relative_residual": terminal_residual,
            "terminal_diverted": bool(topology.diverted),
        },
        "fine_ladder_probe": (
            {
                "performed": True,
                "factors": list(FINE_FACTORS),
                "production_trajectory_affected": False,
                "fixed_ladder_refused_iterations_probed": int(
                    np.count_nonzero(np.any(fine_reasons != NOT_PROBED, axis=1))
                ),
                "largest_admissible_fraction": largest_fine,
                "admissible_fraction_found": bool(largest_fine > 0.0),
                "records": _fine_probe_records(
                    fine_reasons, fine_scores, fine_step_norms
                ),
            }
            if probe_fine
            else {"performed": False}
        ),
        "runtime_seconds": perf_counter() - started,
    }


def _dominant_reason(records: list[dict[str, Any]]) -> tuple[str, int]:
    """Return the most frequent exclusive refusal reason across the cohort."""

    refusal_counts: Counter[str] = Counter()
    for record in records:
        for reason, count in record["fixed_ladder"][
            "exclusive_trial_reason_counts"
        ].items():
            if reason not in {"ADMITTED", "NOT_PROBED"}:
                refusal_counts[reason] += count
    return refusal_counts.most_common(1)[0]


def run(data: Path, output: Path) -> dict[str, Any]:
    """Diagnose the fixed cohort and write the quantitative receipt."""

    configure_dtypes()
    repaired, favourable = _read_receipts()
    banked_by_case = {
        (record["shot"], int(record["frame"])): record
        for record in repaired["frame_records"]
    }
    records = [
        _solve_frame(data, case, banked_by_case[(case.shot, case.frame)])
        for case in repaired_solve.COHORT
    ]
    dominant_reason, dominant_count = _dominant_reason(records)
    probe = next(
        record for record in records if record["fine_ladder_probe"]["performed"]
    )
    largest_fine = probe["fine_ladder_probe"]["largest_admissible_fraction"]
    if largest_fine > 0.0:
        verdict = "STEP_LADDER_TOO_COARSE"
        basis = (
            "An unchanged-predicate trial is admissible below the production "
            "ladder floor of 0.125."
        )
    else:
        verdict = "ADMISSIBLE_REGION_NARROW"
        basis = "No nonzero trial on the diagnostic ladder down to 2^-16 is admissible."
    repaired_residuals = [
        record["terminal_relative_residual"] for record in repaired["frame_records"]
    ]
    unqualified_residuals = [
        record["previous_unqualified_relative_residual"]
        for record in repaired["frame_records"]
    ]
    favourable_coarse = favourable["rungs"][0]["solver"]
    favourable_native = favourable["rungs"][1]["solver"]
    receipt = {
        "artifact": "diiid_admission_rate_diagnosis",
        "source_commit": _source_commit(),
        "scope": {
            "diagnosis_only": True,
            "admission_predicate_changed": False,
            "production_factor_ladder_changed": False,
            "score_blind_frame_count": len(records),
            "newton_promotions_per_frame": repaired_solve.NEWTON_STEPS,
            "gmres_iterations_per_promotion": repaired_solve.GMRES_ITERATIONS,
            "requested_topology_class": "diverted",
        },
        "banked_comparators": {
            "repaired_receipt": str(REPAIRED_RECEIPT.relative_to(HERE)),
            "repaired_receipt_sha256": _sha256(REPAIRED_RECEIPT),
            "admission_counts_of_89": [
                record["promoted_iteration_count"]
                for record in repaired["frame_records"]
            ],
            "repaired_residual_range": [
                min(repaired_residuals),
                max(repaired_residuals),
            ],
            "unqualified_plateau_range": [
                min(unqualified_residuals),
                max(unqualified_residuals),
            ],
            "favourable_receipt": str(FAVOURABLE_RECEIPT.relative_to(HERE)),
            "favourable_receipt_sha256": _sha256(FAVOURABLE_RECEIPT),
            "favourable_coarse_full_step_admissions_of_89": favourable_coarse[
                "accepted_factor_counts"
            ]["1.0"],
            "favourable_coarse_terminal_relative_residual": favourable_coarse[
                "terminal_relative_residual"
            ],
            "favourable_native_full_step_admissions_of_89": favourable_native[
                "accepted_factor_counts"
            ]["1.0"],
            "favourable_native_terminal_relative_residual": favourable_native[
                "terminal_relative_residual"
            ],
        },
        "reason_contract": {
            "exclusive_precedence": [
                "KRYLOV_ACTION_FAILED",
                "NONFINITE_CANDIDATE_STATE",
                "NONFINITE_MAPPED_STATE_OR_RESIDUAL",
                "LEFT_TOPOLOGY_CLASS",
                "CALLER_PREDICATE_FAILED",
                "ADMITTED",
            ],
            "caller_predicate": (
                "candidate state is finite and emergent topology is diverted"
            ),
            "selection_rule": (
                "production factors and nonmonotone envelope are replayed "
                "unchanged; the fine ladder never promotes a state"
            ),
        },
        "frame_records": records,
        "cohort_summary": {
            "dominant_rejection_reason": dominant_reason,
            "dominant_rejection_count": dominant_count,
            "total_fixed_ladder_trials": int(
                sum(record["fixed_ladder"]["offered_trial_count"] for record in records)
            ),
            "total_rejected_fixed_ladder_trials": int(
                sum(
                    record["fixed_ladder"]["rejected_trial_count"] for record in records
                )
            ),
            "fine_probe_frame": {
                "shot": probe["shot"],
                "frame": probe["frame"],
            },
            "fine_probe_smallest_fraction": min(FINE_FACTORS),
            "largest_admissible_fraction_below_0.125": largest_fine,
        },
        "verdict": {
            "classification": verdict,
            "basis": basis,
            "qualification": (
                "This classifies trial admission only. It does not change the "
                "solver, admit a diagnostic trial, or establish convergence."
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run(arguments.data, arguments.output)
    print(json.dumps(receipt["cohort_summary"], sort_keys=True), flush=True)
    print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
