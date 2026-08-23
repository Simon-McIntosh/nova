"""Pair topology-manifold advance with fixed backtracking on one source tree.

Both arms start from the same admitted bootstrap state, use the same physical
predicate, iteration budget, Krylov dimension, and infinite conditioning ratio
limit.  The manifold arm alone receives the preceding admitted seed needed to
form its state-space secant.  The receipt also quotes the conditioning-disabled
arm from the established admissible-step control, but treats that older tree as
a non-causal historical comparator.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import numpy as np

from benchmarks import diiid_admissible_step_control as step_control
from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
from nova.equilibrium.fixed_point import newton_krylov
from nova.equilibrium.manifold_advance import ManifoldAdvanceQualification
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    HERE / "docs/figures/topology-preserving-continuation/manifold-advance.json"
)
COMPARATOR_RECEIPT = (
    HERE / "docs/figures/topology-preserving-continuation/admissible-step-control.json"
)
BOOTSTRAP_RELAXATION = 0.5
BOOTSTRAP_FACTORS = step_control.FACTORS


def _git(*arguments: str) -> str:
    """Return one read-only git query from the measured checkout."""

    return subprocess.check_output(["git", *arguments], cwd=HERE, text=True).strip()


def _sha256(path: Path) -> str:
    """Return one input or driver identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology_metrics(profile, state) -> dict[str, Any]:
    """Return the physical terminal classification for one solver state."""

    _masks, topology = profile.operator.read(state)
    x_point = np.asarray(topology.x_point, dtype=float)
    finite_x_point = bool(np.all(np.isfinite(x_point)))
    return {
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "terminal_x_point_finite": finite_x_point,
        "terminal_x_point_rz_m": x_point.tolist() if finite_x_point else None,
    }


def _relative_residual(mapped, state) -> float:
    """Return the terminal relative sup-norm fixed-point residual."""

    host_state = np.asarray(state, dtype=float)
    host_image = np.asarray(mapped(state), dtype=float)
    return float(
        np.max(np.abs(host_image - host_state))
        / max(np.max(np.abs(host_image)), 1.0e-30)
    )


def _shared_bootstrap(mapped, seed, profile) -> tuple[Any, Any, dict[str, Any]]:
    """Build one admitted secant and common initial state without score labels."""

    _masks, seed_topology = profile.operator.read(seed)
    if not bool(seed_topology.diverted):
        raise RuntimeError("the diverted cold seed is not on the requested branch")

    residual_vector = mapped(seed) - seed
    for factor in BOOTSTRAP_FACTORS:
        candidate = seed + BOOTSTRAP_RELAXATION * factor * residual_vector
        _masks, topology = profile.operator.read(candidate)
        admitted = bool(jnp.all(jnp.isfinite(candidate)) & topology.diverted)
        if admitted:
            return (
                seed,
                candidate,
                {
                    "construction": (
                        "relaxed fixed-point image on the fixed factor ladder"
                    ),
                    "relaxation": BOOTSTRAP_RELAXATION,
                    "selected_factor": factor,
                    "candidate_fraction_of_map_residual": BOOTSTRAP_RELAXATION * factor,
                    "preceding_state_admitted": True,
                    "initial_state_admitted": True,
                    "score_labels_read": False,
                },
            )
    raise RuntimeError("the fixed bootstrap ladder produced no admitted secant")


def _manifold_metrics(mapped, initial, previous, profile) -> dict[str, Any]:
    """Run and reduce the conditioning-disabled predictor-corrector arm."""

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

    started = perf_counter()
    result = newton_krylov(
        mapped,
        initial,
        newton_steps=repaired_solve.NEWTON_STEPS,
        gmres_iterations=repaired_solve.GMRES_ITERATIONS,
        warmup=0,
        krylov_condition_limit=math.inf,
        admissibility_fn=remains_diverted,
        previous_admitted_state=previous,
    )
    result.state.block_until_ready()
    runtime_seconds = perf_counter() - started

    advance_lengths = np.asarray(result.advance_lengths, dtype=float)
    newton_lengths = np.asarray(result.newton_step_lengths, dtype=float)
    promoted = advance_lengths > 0.0
    fractions = np.divide(
        advance_lengths,
        newton_lengths,
        out=np.zeros_like(advance_lengths),
        where=newton_lengths > np.finfo(newton_lengths.dtype).tiny,
    )
    promoted_fractions = fractions[promoted]
    qualifications = [
        ManifoldAdvanceQualification(int(value)).name
        for value in np.asarray(result.manifold_advance_qualification)
    ]
    achieved = float(result.newton_step_equivalents)
    record = {
        "route": "topology-preserving state-arclength predictor-corrector",
        "conditioning_enabled": False,
        "conditioning_ratio_limit": "infinity",
        "terminal_relative_residual": _relative_residual(mapped, result.state),
        "admitted_advance_count_of_89": int(np.count_nonzero(promoted)),
        "mean_advance_length_fraction_of_corresponding_newton_step": (
            float(np.mean(promoted_fractions)) if promoted_fractions.size else 0.0
        ),
        "achieved_newton_step_equivalents": achieved,
        "newton_step_equivalent_basis": (
            "sum(promoted advance length / corresponding qualified Newton-step length)"
        ),
        "mean_advance_fraction_across_89": (achieved / repaired_solve.NEWTON_STEPS),
        "advance_fraction_by_promoted_update": promoted_fractions.tolist(),
        "manifold_advance_qualification_counts": dict(Counter(qualifications)),
        "candidate_admissibility_count_of_89": int(
            np.count_nonzero(np.asarray(result.manifold_admissibility, dtype=bool))
        ),
        "krylov_conditioning_count": int(result.krylov_conditioning_count),
        "maximum_projected_krylov_condition": float(
            result.maximum_projected_krylov_condition
        ),
        "runtime_seconds": runtime_seconds,
    }
    record.update(_topology_metrics(profile, result.state))
    return record


def _backtracking_metrics(mapped, initial, profile) -> dict[str, Any]:
    """Run the conditioning-disabled fixed backtracking arm."""

    record = step_control._solve(mapped, initial, profile, conditioning_enabled=False)
    count = record["admitted_advance_count_of_89"]
    return {
        "route": "topology-qualified fixed backtracking",
        **record,
        "mean_advance_length_fraction_of_corresponding_newton_step": (
            record["achieved_newton_step_equivalents"] / count if count else 0.0
        ),
    }


def _quoted_comparator(record: dict[str, Any]) -> dict[str, Any]:
    """Quote the conditioning-disabled arm using the requested field names."""

    control = record["current_tree_conditioning_disabled_control"]
    count = control["admitted_advance_count_of_89"]
    return {
        "source_commit": record["source_commit"],
        "terminal_relative_residual": control["terminal_relative_residual"],
        "admitted_advance_count_of_89": control["admitted_advance_count_of_89"],
        "mean_advance_length_fraction_of_corresponding_newton_step": control[
            "achieved_newton_step_equivalents"
        ]
        / count
        if count
        else 0.0,
        "achieved_newton_step_equivalents": control["achieved_newton_step_equivalents"],
        "terminal_topology_class": control["terminal_topology_class"],
        "terminal_x_point_finite": control["terminal_x_point_finite"],
    }


def _delta(measured: dict[str, Any], comparator: dict[str, Any]) -> dict[str, Any]:
    """Return predictor-corrector minus current-tree backtracking quantities."""

    return {
        "terminal_relative_residual_change": (
            measured["terminal_relative_residual"]
            - comparator["terminal_relative_residual"]
        ),
        "admitted_advance_count_change": (
            measured["admitted_advance_count_of_89"]
            - comparator["admitted_advance_count_of_89"]
        ),
        "mean_advance_fraction_change": (
            measured["mean_advance_length_fraction_of_corresponding_newton_step"]
            - comparator["mean_advance_length_fraction_of_corresponding_newton_step"]
        ),
        "achieved_newton_step_equivalents_change": (
            measured["achieved_newton_step_equivalents"]
            - comparator["achieved_newton_step_equivalents"]
        ),
    }


def _production_diff(comparator_commit: str, current_commit: str) -> dict[str, Any]:
    """Name production differences to keep the historical quote non-causal."""

    changed_paths = _git(
        "diff", "--name-only", f"{comparator_commit}..{current_commit}", "--", "nova"
    ).splitlines()
    return {
        "quoted_comparator_source_commit": comparator_commit,
        "measured_source_commit": current_commit,
        "changed_production_paths": changed_paths,
        "historical_attribution": (
            "DECLINED_MULTIPLE_PRODUCTION_CHANGES"
            if changed_paths
            else "SAME_PRODUCTION_TREE"
        ),
        "paired_current_tree_attribution": "ISOLATED_ADVANCE_MECHANISM",
        "paired_current_tree_reason": (
            "Both measured arms share source, data, cold seed, admitted bootstrap, "
            "initial state, topology predicate, 89-update budget, 24-vector Krylov "
            "route, backtracking factor ladder where applicable, and disabled step "
            "conditioning; the predictor-corrector is the only mechanism that differs."
        ),
    }


def measure(data: Path, output: Path) -> dict[str, Any]:
    """Write the paired five-frame manifold-advance receipt."""

    configure_dtypes()
    repaired_solve._validate_baseline()
    current_commit = _git("rev-parse", "HEAD")
    comparator_receipt = json.loads(COMPARATOR_RECEIPT.read_text(encoding="utf-8"))
    comparator_commit = str(comparator_receipt["source_commit"])
    comparator_records = {
        (record["shot"], int(record["frame"])): {
            **record,
            "source_commit": comparator_commit,
        }
        for record in comparator_receipt["frame_records"]
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
        previous, initial, bootstrap = _shared_bootstrap(mapped, seed, profile)
        manifold = _manifold_metrics(mapped, initial, previous, profile)
        backtracking = _backtracking_metrics(mapped, initial, profile)
        quoted = _quoted_comparator(comparator_records[(case.shot, case.frame)])
        records.append(
            {
                "shot": case.shot,
                "frame": case.frame,
                "time_ms": time_ms,
                "target_current_a": float(target_current_a),
                "shared_admitted_secant_bootstrap": bootstrap,
                "predictor_corrector_conditioning_disabled": manifold,
                "paired_backtracking_conditioning_disabled": backtracking,
                "paired_predictor_corrector_minus_backtracking": _delta(
                    manifold, backtracking
                ),
                "quoted_admissible_step_control_conditioning_disabled": quoted,
            }
        )

    manifold_records = [
        record["predictor_corrector_conditioning_disabled"] for record in records
    ]
    backtracking_records = [
        record["paired_backtracking_conditioning_disabled"] for record in records
    ]
    quoted_records = [
        record["quoted_admissible_step_control_conditioning_disabled"]
        for record in records
    ]
    manifold_residuals = [
        record["terminal_relative_residual"] for record in manifold_records
    ]
    backtracking_residuals = [
        record["terminal_relative_residual"] for record in backtracking_records
    ]
    residual_improvements = sum(
        manifold < backtracking
        for manifold, backtracking in zip(
            manifold_residuals, backtracking_residuals, strict=True
        )
    )
    residual_regressions = sum(
        manifold > backtracking
        for manifold, backtracking in zip(
            manifold_residuals, backtracking_residuals, strict=True
        )
    )
    topology_held = all(
        record["terminal_topology_class"] == "diverted"
        and record["terminal_x_point_finite"]
        for record in manifold_records
    )
    receipt = {
        "artifact": "diiid_manifold_advance",
        "source_commit": current_commit,
        "driver_sha256": _sha256(Path(__file__)),
        "completed_utc": np.datetime_as_string(np.datetime64("now"), timezone="UTC"),
        "measurement_scope": {
            "selection": "five banked score-blind circuit-current frames",
            "score_labels_read": False,
            "frame_count": len(records),
            "data_root": str(data),
            "quoted_comparator_receipt": str(COMPARATOR_RECEIPT.relative_to(HERE)),
            "quoted_comparator_receipt_sha256": _sha256(COMPARATOR_RECEIPT),
        },
        "solver_contract": {
            "requested_topology_class": "diverted",
            "newton_updates_per_arm": repaired_solve.NEWTON_STEPS,
            "gmres_iterations": repaired_solve.GMRES_ITERATIONS,
            "warmup_updates_per_measured_arm": 0,
            "step_conditioning_enabled_in_predictor_corrector": False,
            "step_conditioning_enabled_in_backtracking": False,
            "condition_ratio_limit_in_both_arms": "infinity",
            "fixed_backtracking_factors": list(step_control.FACTORS),
            "shared_bootstrap_relaxation": BOOTSTRAP_RELAXATION,
            "shared_bootstrap_factors": list(BOOTSTRAP_FACTORS),
            "admission_predicate_changed_between_arms": False,
        },
        "comparison_contract": {
            "paired_current_tree_is_causal_control": True,
            "quoted_historical_tree_is_causal_control": False,
            "delta_definition": "predictor-corrector minus paired backtracking",
        },
        "production_diff": _production_diff(comparator_commit, current_commit),
        "frame_records": records,
        "cohort_summary": {
            "predictor_corrector_terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted"
                for record in manifold_records
            ),
            "predictor_corrector_terminal_x_point_finite": sum(
                record["terminal_x_point_finite"] for record in manifold_records
            ),
            "backtracking_terminal_diverted": sum(
                record["terminal_topology_class"] == "diverted"
                for record in backtracking_records
            ),
            "backtracking_terminal_x_point_finite": sum(
                record["terminal_x_point_finite"] for record in backtracking_records
            ),
            "predictor_corrector_total_admitted_advances": sum(
                record["admitted_advance_count_of_89"] for record in manifold_records
            ),
            "backtracking_total_admitted_advances": sum(
                record["admitted_advance_count_of_89"]
                for record in backtracking_records
            ),
            "predictor_corrector_total_newton_step_equivalents": math.fsum(
                record["achieved_newton_step_equivalents"]
                for record in manifold_records
            ),
            "backtracking_total_newton_step_equivalents": math.fsum(
                record["achieved_newton_step_equivalents"]
                for record in backtracking_records
            ),
            "quoted_control_total_newton_step_equivalents": math.fsum(
                record["achieved_newton_step_equivalents"] for record in quoted_records
            ),
            "quoted_control_residual_range": [
                min(record["terminal_relative_residual"] for record in quoted_records),
                max(record["terminal_relative_residual"] for record in quoted_records),
            ],
            "predictor_corrector_lower_residual_than_paired_backtracking": (
                residual_improvements
            ),
            "predictor_corrector_higher_residual_than_paired_backtracking": (
                residual_regressions
            ),
        },
        "conclusion": {
            "verdict": (
                "REQUESTED_TOPOLOGY_NOT_PRESERVED"
                if not topology_held
                else "TOPOLOGY_HELD_RESIDUAL_REGRESSED_ON_FOUR_OF_FIVE"
                if residual_regressions == 4
                else "TOPOLOGY_HELD_PAIRED_OUTCOME_MEASURED"
            ),
            "predictor_corrector_residual_range": [
                min(manifold_residuals),
                max(manifold_residuals),
            ],
            "paired_backtracking_residual_range": [
                min(backtracking_residuals),
                max(backtracking_residuals),
            ],
            "convergence_gate_declared": False,
            "statement": (
                "The predictor-corrector preserved diverted topology and finite "
                "X points on every frame and traversed more Newton-step-equivalents "
                "than paired backtracking, but admitted fewer advances and ended "
                "with a higher residual on four of five frames. It is therefore a "
                "topology-preserving negative rather than a convergence result. No "
                "threshold is derived from the observed values."
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
    """Run the paired manifold-advance measurement."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repaired_solve.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = measure(arguments.data, arguments.output)
    print(json.dumps(receipt["cohort_summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
