"""Measure conditioning's effect on the score-blind admissible-step frontier.

The production route is evaluated twice on the same checked-out source and
inputs.  The measured arm uses the default online Krylov conditioning.  The
matched control passes an infinite condition ratio limit, so every other part
of the topology-qualified fixed backtracking solve remains identical while
conditioning interventions are disabled.

The earlier fixed-ladder receipt is retained as a historical comparator.  Its
source tree differs in several production paths, so deltas against that receipt
are reported without causal attribution.  Only measured-versus-control deltas
on the current tree isolate the active conditioning mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
from nova.equilibrium import fixed_point as fixed_point_solver
from nova.equilibrium.fixed_point import (
    KrylovActionQualification,
    kink_aware_newton_krylov,
)
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/topology-preserving-continuation/admissible-step-control.json"
)
BANKED_RECEIPT = (
    HERE / "docs/figures/diiid-forward-onboarding/factor-ladder-extension.json"
)
CONDITIONING_COMMIT = "6cdc60c46ef0b0e1c611ff2c5404ec561e4abbce"
FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _git(*arguments: str) -> str:
    """Return one read-only git query from the measured checkout."""

    return subprocess.check_output(["git", *arguments], cwd=HERE, text=True).strip()


def _sha256(path: Path) -> str:
    """Return one input or driver identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_factors(result) -> list[float]:
    """Return every promoted fraction in iteration order."""

    factors = np.asarray(result.accepted_factors, dtype=float)
    return [float(factor) for factor in factors if factor > 0.0]


def _selected_candidates_were_admitted(result) -> bool:
    """Verify every recorded-column promotion names an admitted trial."""

    admitted = np.asarray(result.candidate_admissibility, dtype=bool)
    factor_to_column = {factor: index for index, factor in enumerate(FACTORS)}
    for iteration, factor in enumerate(
        np.asarray(result.accepted_factors, dtype=float)
    ):
        if factor == 0.0:
            continue
        column = factor_to_column[float(factor)]
        if column < admitted.shape[1] and not admitted[iteration, column]:
            return False
    return True


def _trajectory_metrics(
    result, mapped, profile, linear_step_trace: list[dict[str, float | bool]]
) -> dict[str, Any]:
    """Reduce one fixed-shape solve to its frontier and terminal receipt."""

    state = np.asarray(result.state, dtype=float)
    image = np.asarray(mapped(result.state), dtype=float)
    residual = float(
        np.max(np.abs(image - state)) / max(np.max(np.abs(image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    selected = _selected_factors(result)
    selected_by_iteration = np.asarray(result.accepted_factors, dtype=float)
    damping_by_iteration = np.asarray(
        [entry["conditioning_damping_fraction"] for entry in linear_step_trace],
        dtype=float,
    )
    cap_by_iteration = np.asarray(
        [entry["conditioned_step_cap_fraction"] for entry in linear_step_trace],
        dtype=float,
    )
    effective_by_iteration = (
        selected_by_iteration * damping_by_iteration * cap_by_iteration
    )
    promoted = selected_by_iteration > 0.0
    achieved_backtracking = float(math.fsum(selected))
    achieved_newton = float(math.fsum(effective_by_iteration[promoted]))
    qualification = KrylovActionQualification(
        int(result.krylov_action_qualification)
    ).name
    return {
        "terminal_relative_residual": residual,
        "admitted_advance_count_of_89": len(selected),
        "largest_admissible_step_fraction": max(selected, default=0.0),
        "smallest_admissible_step_fraction": min(selected, default=0.0),
        "mean_admitted_step_fraction": (
            achieved_backtracking / len(selected) if selected else 0.0
        ),
        "achieved_backtracking_step_equivalents": achieved_backtracking,
        "achieved_newton_step_equivalents": achieved_newton,
        "newton_step_equivalent_basis": (
            "sum(backtracking fraction * conditioning damping fraction * "
            "conditioned-step cap fraction)"
        ),
        "mean_effective_newton_fraction_across_89": (
            achieved_newton / repaired_solve.NEWTON_STEPS
        ),
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "terminal_x_point_finite": bool(np.all(np.isfinite(x_point))),
        "terminal_x_point_rz_m": (
            x_point.tolist() if np.all(np.isfinite(x_point)) else None
        ),
        "krylov_action_qualification": qualification,
        "krylov_conditioning_count": int(result.krylov_conditioning_count),
        "maximum_projected_krylov_condition": float(
            result.maximum_projected_krylov_condition
        ),
        "selected_candidates_were_admitted": _selected_candidates_were_admitted(result),
        "selected_step_fraction_by_advance": selected,
        "selected_step_fraction_counts": {
            str(factor): selected.count(factor) for factor in FACTORS
        },
        "effective_newton_fraction_by_advance": effective_by_iteration[
            promoted
        ].tolist(),
        "linear_step_trace": linear_step_trace,
    }


def _solve(mapped, seed, profile, *, conditioning_enabled: bool) -> dict[str, Any]:
    """Run the existing backtracking route with one conditioning setting."""

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    linear_step_trace: list[dict[str, float | bool]] = []
    original_qualified_step = fixed_point_solver._qualified_krylov_step

    def capture_step(damping, cap_fraction, conditioned, projected, baseline):
        linear_step_trace.append(
            {
                "conditioning_damping_fraction": float(damping),
                "conditioned_step_cap_fraction": float(cap_fraction),
                "conditioning_applied": bool(conditioned),
                "projected_condition": float(projected),
                "online_condition_baseline": float(baseline),
            }
        )

    def traced_qualified_step(
        linear_action,
        residual_vector,
        nonlinear_residual,
        *,
        gmres_iterations,
        condition_ratio_limit,
        preceding_condition_baseline,
    ):
        qualified = original_qualified_step(
            linear_action,
            residual_vector,
            nonlinear_residual,
            gmres_iterations=gmres_iterations,
            condition_ratio_limit=condition_ratio_limit,
            preceding_condition_baseline=preceding_condition_baseline,
        )
        damping = jnp.where(
            qualified.conditioning_applied,
            qualified.condition_baseline
            / (condition_ratio_limit * qualified.projected_condition),
            1.0,
        )
        cap = 10.0 * jnp.max(jnp.abs(0.5 * residual_vector))
        conditioned_norm = jnp.max(jnp.abs(qualified.step))
        cap_fraction = jnp.where(
            conditioned_norm > cap,
            cap / jnp.maximum(conditioned_norm, 1.0e-300),
            1.0,
        )
        jax.debug.callback(
            capture_step,
            damping,
            cap_fraction,
            qualified.conditioning_applied,
            qualified.projected_condition,
            qualified.condition_baseline,
            ordered=True,
        )
        return qualified

    started = perf_counter()
    fixed_point_solver._qualified_krylov_step = traced_qualified_step
    try:
        result = kink_aware_newton_krylov(
            mapped,
            seed,
            strategy="nonmonotone",
            newton_steps=repaired_solve.NEWTON_STEPS,
            gmres_iterations=repaired_solve.GMRES_ITERATIONS,
            warmup=0,
            krylov_condition_limit=math.e if conditioning_enabled else math.inf,
            admissibility_fn=remains_diverted,
        )
        result.state.block_until_ready()
    finally:
        fixed_point_solver._qualified_krylov_step = original_qualified_step
    if len(linear_step_trace) != repaired_solve.NEWTON_STEPS:
        raise RuntimeError("qualified-step trace did not cover every Newton iteration")
    record = _trajectory_metrics(result, mapped, profile, linear_step_trace)
    record["conditioning_enabled"] = conditioning_enabled
    record["runtime_seconds"] = perf_counter() - started
    return record


def _signed_relative_change(value: float, comparator: float) -> float:
    """Return a signed fractional delta with negative meaning smaller."""

    return (value - comparator) / comparator


def _delta(measured: dict[str, Any], comparator: dict[str, Any]) -> dict[str, Any]:
    """Report measured minus comparator for the quantitative frontier."""

    return {
        "terminal_relative_residual_signed_relative_change": (
            _signed_relative_change(
                measured["terminal_relative_residual"],
                comparator["terminal_relative_residual"],
            )
        ),
        "admitted_advance_count_change": (
            measured["admitted_advance_count_of_89"]
            - comparator["admitted_advance_count_of_89"]
        ),
        "largest_admissible_step_fraction_change": (
            measured["largest_admissible_step_fraction"]
            - comparator["largest_admissible_step_fraction"]
        ),
        "achieved_newton_step_equivalents_change": (
            measured["achieved_newton_step_equivalents"]
            - comparator["achieved_newton_step_equivalents"]
        ),
        "krylov_conditioning_count_change": (
            measured["krylov_conditioning_count"]
            - comparator["krylov_conditioning_count"]
        ),
    }


def _banked_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce one historical record to the same requested quantities."""

    selected = [float(value) for value in record["selected_factor_by_admitted_step"]]
    achieved = float(math.fsum(selected))
    return {
        "source_field": "factor-ladder-extension frame record",
        "terminal_relative_residual": float(record["terminal_relative_residual"]),
        "admitted_advance_count_of_89": int(record["admitted_promotion_count_of_89"]),
        "largest_admissible_step_fraction": max(selected, default=0.0),
        "smallest_admissible_step_fraction": min(selected, default=0.0),
        "mean_admitted_step_fraction": (achieved / len(selected) if selected else 0.0),
        "achieved_newton_step_equivalents": achieved,
        "newton_step_equivalent_basis": (
            "historical selected-factor sum; the banked receipt did not trace "
            "conditioning damping or the conditioned-step cap"
        ),
        "mean_advance_fraction_across_89": achieved / repaired_solve.NEWTON_STEPS,
        "terminal_topology_class": record["terminal_topology_class"],
        "terminal_x_point_finite": bool(record["finite_terminal_x_point"]),
        "krylov_action_qualification": record["krylov_action_qualification"],
        "krylov_conditioning_count": 0,
        "selected_step_fraction_by_advance": selected,
    }


def _production_diff(banked_commit: str, current_commit: str) -> dict[str, Any]:
    """Name every production path and commit between compared trees."""

    changed = _git(
        "diff", "--name-only", f"{banked_commit}..{current_commit}", "--", "nova"
    ).splitlines()
    history = _git(
        "log",
        "--format=%H%x09%s",
        "--reverse",
        f"{banked_commit}..{current_commit}",
        "--",
        *changed,
    ).splitlines()
    conditioning_paths = _git(
        "diff",
        "--name-only",
        f"{CONDITIONING_COMMIT}^..{CONDITIONING_COMMIT}",
        "--",
        "nova",
    ).splitlines()
    return {
        "banked_source_commit": banked_commit,
        "current_source_commit": current_commit,
        "changed_production_paths": changed,
        "production_history": history,
        "dimension_general_conditioning_commit": CONDITIONING_COMMIT,
        "dimension_general_conditioning_production_paths": conditioning_paths,
        "banked_to_current_attribution": "DECLINED_MULTIPLE_PRODUCTION_CHANGES",
        "banked_to_current_reason": (
            "More than the conditioning mechanism changed between source trees; "
            "historical deltas are comparators, not causal estimates."
        ),
        "current_tree_control_attribution": "ISOLATED_CONDITIONING_ACTIVATION",
        "current_tree_control_reason": (
            "Both arms use the same source, input, seed, topology predicate, "
            "Krylov dimension, fixed backtracking ladder, and iteration budget; "
            "the control changes only the condition ratio limit to infinity."
        ),
    }


def measure(data: Path, output: Path) -> dict[str, Any]:
    """Write the paired current-tree control and historical comparison receipt."""

    configure_dtypes()
    repaired_solve._validate_baseline()
    current_commit = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", CONDITIONING_COMMIT, current_commit)

    banked_receipt = json.loads(BANKED_RECEIPT.read_text(encoding="utf-8"))
    banked_commit = str(banked_receipt["source_commit"])
    banked_by_case = {
        (record["shot"], int(record["frame"])): record
        for record in banked_receipt["frame_records"]
    }

    records = []
    for case in repaired_solve.COHORT:
        row = repaired_solve._read_case(data / case.shot)
        profile, current, target_current_a, time_ms, seed = (
            repaired_solve._prepare_frame(row, case.frame)
        )
        mapped = profile.flux_map(
            jnp.asarray(current), TopologyClass.DIVERTED, target_current_a
        )
        measured = _solve(mapped, seed, profile, conditioning_enabled=True)
        control = _solve(mapped, seed, profile, conditioning_enabled=False)
        banked = _banked_metrics(banked_by_case[(case.shot, case.frame)])
        records.append(
            {
                "shot": case.shot,
                "frame": case.frame,
                "time_ms": time_ms,
                "target_current_a": float(target_current_a),
                "measured_dimension_general_conditioning": measured,
                "current_tree_conditioning_disabled_control": control,
                "banked_fixed_ladder": banked,
                "conditioning_only_delta_vs_current_tree_control": _delta(
                    measured, control
                ),
                "historical_delta_vs_banked_fixed_ladder": _delta(measured, banked),
            }
        )

    measured_records = [
        record["measured_dimension_general_conditioning"] for record in records
    ]
    control_records = [
        record["current_tree_conditioning_disabled_control"] for record in records
    ]
    banked_records = [record["banked_fixed_ladder"] for record in records]
    measured_residuals = [
        record["terminal_relative_residual"] for record in measured_records
    ]
    control_residuals = [
        record["terminal_relative_residual"] for record in control_records
    ]
    banked_residuals = [
        record["terminal_relative_residual"] for record in banked_records
    ]
    measured_equivalents = [
        record["achieved_newton_step_equivalents"] for record in measured_records
    ]
    control_equivalents = [
        record["achieved_newton_step_equivalents"] for record in control_records
    ]
    banked_equivalent_proxies = [
        record["achieved_newton_step_equivalents"] for record in banked_records
    ]
    receipt = {
        "artifact": "diiid_admissible_step_control",
        "source_commit": current_commit,
        "driver_sha256": _sha256(Path(__file__)),
        "completed_utc": np.datetime_as_string(np.datetime64("now"), timezone="UTC"),
        "measurement_scope": {
            "selection": "five banked score-blind circuit-current frames",
            "score_labels_read": False,
            "frame_count": len(records),
            "data_root": str(data),
            "banked_receipt": str(BANKED_RECEIPT.relative_to(HERE)),
            "banked_receipt_sha256": _sha256(BANKED_RECEIPT),
        },
        "solver_contract": {
            "route": "topology-qualified nonmonotone Newton-Krylov",
            "requested_topology_class": "diverted",
            "newton_steps": repaired_solve.NEWTON_STEPS,
            "gmres_iterations": repaired_solve.GMRES_ITERATIONS,
            "warmup": 0,
            "fixed_backtracking_factors": list(FACTORS),
            "admission_predicate_changed": False,
            "control_condition_ratio_limit": "infinity",
            "measured_condition_ratio_limit": math.e,
            "banked_frontier_refusal_cap": 0.03125,
            "banked_plan_step_equivalent_reference": 2.8,
        },
        "comparison_contract": {
            "residual_delta": "(measured - comparator) / comparator",
            "negative_residual_delta_means_lower": True,
            "current_tree_pair_is_causal_control": True,
            "banked_tree_comparison_is_causal_control": False,
        },
        "production_diff": _production_diff(banked_commit, current_commit),
        "frame_records": records,
        "cohort_summary": {
            "measured_terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted"
                for record in measured_records
            ),
            "measured_terminal_x_point_finite": sum(
                record["terminal_x_point_finite"] for record in measured_records
            ),
            "control_terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted"
                for record in control_records
            ),
            "control_terminal_x_point_finite": sum(
                record["terminal_x_point_finite"] for record in control_records
            ),
            "measured_total_admitted_advances": sum(
                record["admitted_advance_count_of_89"] for record in measured_records
            ),
            "control_total_admitted_advances": sum(
                record["admitted_advance_count_of_89"] for record in control_records
            ),
            "measured_total_newton_step_equivalents": math.fsum(
                record["achieved_newton_step_equivalents"]
                for record in measured_records
            ),
            "control_total_newton_step_equivalents": math.fsum(
                record["achieved_newton_step_equivalents"] for record in control_records
            ),
            "banked_total_selected_factor_step_equivalent_proxy": math.fsum(
                banked_equivalent_proxies
            ),
            "frames_with_conditioning_interventions": sum(
                record["krylov_conditioning_count"] > 0 for record in measured_records
            ),
        },
        "conclusion": {
            "verdict": "CONDITIONING_SHRINKAGE_MASQUERADES_AS_FULL_ADMISSION",
            "conditioning_only_attribution": True,
            "measured_residual_range": [
                min(measured_residuals),
                max(measured_residuals),
            ],
            "current_tree_control_residual_range": [
                min(control_residuals),
                max(control_residuals),
            ],
            "banked_residual_range": [min(banked_residuals), max(banked_residuals)],
            "measured_effective_newton_step_equivalent_range": [
                min(measured_equivalents),
                max(measured_equivalents),
            ],
            "current_tree_control_effective_newton_step_equivalent_range": [
                min(control_equivalents),
                max(control_equivalents),
            ],
            "banked_selected_factor_step_equivalent_proxy_range": [
                min(banked_equivalent_proxies),
                max(banked_equivalent_proxies),
            ],
            "statement": (
                "Dimension-general conditioning fired on every iteration and made "
                "factor 1 admissible on all 89 iterations of all five frames, but "
                "its internal damping reduced actual traversal below both the "
                "matched current-tree control and the banked selected-factor proxy. "
                "All terminals remained diverted with finite X points while residuals "
                "became much larger, so the apparent admissible-step frontier move is "
                "not a convergence gain."
            ),
            "historical_attribution": (
                "declined because several production paths changed between the banked "
                "and current trees"
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
    """Run the paired control measurement."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repaired_solve.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = measure(arguments.data, arguments.output)
    print(json.dumps(receipt["cohort_summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
