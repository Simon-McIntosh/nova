"""Admission-aware contracts and production receipt for Krylov conditioning."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from benchmarks import diiid_repaired_solve_remeasure as repaired_solve
    from nova.equilibrium.fixed_point import kink_aware_newton_krylov
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parents[1]
BANKED_CONTROL = (
    HERE / "docs/figures/topology-preserving-continuation/admissible-step-control.json"
)
DEFAULT_OUTPUT = (
    HERE / "docs/figures/topology-preserving-continuation/conditioning-repair.json"
)


def _diagonal_fixed_point(condition: float, dimension: int):
    """Return a linear map with a logarithmically distributed action spectrum."""
    diagonal = jnp.exp(jnp.linspace(0.0, -jnp.log(condition), dimension))
    action = jnp.diag(diagonal)
    tangent = jnp.eye(dimension) - action
    return lambda state: tangent @ state + jnp.ones(dimension)


def _admission_aware_solve(dimension: int, bound: float):
    """Run one conditioned step whose raw ladder lies outside a state bound."""
    return kink_aware_newton_krylov(
        _diagonal_fixed_point(200.0, dimension),
        jnp.zeros(dimension),
        strategy="nonmonotone",
        newton_steps=1,
        gmres_iterations=dimension,
        warmup=0,
        step_cap=1.0e6,
        admissibility_fn=lambda state: jnp.max(jnp.abs(state)) <= bound,
    )


@pytest.mark.parametrize("dimension", [8, 12, 24])
def test_conditioning_engages_only_after_raw_ladder_refusal(dimension: int):
    """An improving conditioned fallback engages at every projection dimension."""
    result = jax.jit(lambda: _admission_aware_solve(dimension, 1.0))()

    assert int(result.krylov_conditioning_count) == 1
    assert float(result.accepted_factors[0]) > 0.0
    assert 0.0 < float(result.effective_newton_fractions[0]) < 0.03125
    assert float(result.residual) < 1.0


def test_admitted_raw_ladder_preserves_disabled_arm_traversal():
    """Conditioning cannot shrink a step already admitted without damping."""
    dimension = 12
    common = {
        "strategy": "nonmonotone",
        "newton_steps": 1,
        "gmres_iterations": dimension,
        "warmup": 0,
        "step_cap": 1.0e6,
        "admissibility_fn": lambda _state: jnp.asarray(True),
    }
    measured = kink_aware_newton_krylov(
        _diagonal_fixed_point(200.0, dimension),
        jnp.zeros(dimension),
        **common,
    )
    disabled = kink_aware_newton_krylov(
        _diagonal_fixed_point(200.0, dimension),
        jnp.zeros(dimension),
        krylov_condition_limit=math.inf,
        **common,
    )

    assert int(measured.krylov_conditioning_count) == 0
    np.testing.assert_array_equal(measured.accepted_factors, disabled.accepted_factors)
    np.testing.assert_array_equal(measured.state, disabled.state)
    np.testing.assert_array_equal(measured.effective_newton_fractions, [1.0])


def test_conditioned_fallback_cannot_override_physical_refusal():
    """A damped candidate outside the caller's admissible set remains refused."""
    result = _admission_aware_solve(12, 0.1)

    assert int(result.krylov_conditioning_count) == 0
    np.testing.assert_array_equal(result.accepted_factors, [0.0])
    np.testing.assert_array_equal(result.effective_newton_fractions, [0.0])
    np.testing.assert_array_equal(result.state, np.zeros(12))


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    """Return one read-only git query from the measured checkout."""
    return subprocess.check_output(["git", *arguments], cwd=HERE, text=True).strip()


def _trajectory_metrics(result, mapped, profile) -> dict[str, Any]:
    """Reduce one production solve to the required traversal receipt."""
    state = np.asarray(result.state, dtype=float)
    image = np.asarray(mapped(result.state), dtype=float)
    residual = float(
        np.max(np.abs(image - state)) / max(np.max(np.abs(image)), 1.0e-30)
    )
    _masks, topology = profile.operator.read(result.state)
    x_point = np.asarray(topology.x_point, dtype=float)
    accepted = np.asarray(result.accepted_factors, dtype=float)
    effective = np.asarray(result.effective_newton_fractions, dtype=float)
    return {
        "intervention_count_out_of_89": int(result.krylov_conditioning_count),
        "admitted_advance_count_out_of_89": int(np.count_nonzero(accepted)),
        "achieved_newton_step_equivalents": float(math.fsum(effective)),
        "terminal_relative_residual": residual,
        "terminal_topology_class": (
            "diverted" if bool(topology.diverted) else "limited"
        ),
        "terminal_x_point_finite": bool(np.all(np.isfinite(x_point))),
    }


def _solve_frame(mapped, seed, profile, *, conditioning_enabled: bool):
    """Run one topology-qualified production arm."""

    def remains_diverted(candidate):
        _masks, topology = profile.operator.read(candidate)
        return jnp.all(jnp.isfinite(candidate)) & topology.diverted

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
    return _trajectory_metrics(result, mapped, profile)


def _banked_metrics(record: dict[str, Any], key: str) -> dict[str, Any]:
    """Select the four required values from one banked arm."""
    arm = record[key]
    return {
        "intervention_count_out_of_89": int(arm["krylov_conditioning_count"]),
        "admitted_advance_count_out_of_89": int(arm["admitted_advance_count_of_89"]),
        "achieved_newton_step_equivalents": float(
            arm["achieved_newton_step_equivalents"]
        ),
        "terminal_relative_residual": float(arm["terminal_relative_residual"]),
    }


def _dimension_engagement() -> list[dict[str, int]]:
    """Measure admission-aware engagement at each required projection dimension."""
    records = []
    for dimension in (8, 12, 24):
        result = jax.jit(lambda: _admission_aware_solve(dimension, 1.0))()
        records.append(
            {
                "projection_dimension": dimension,
                "intervention_count_out_of_1": int(result.krylov_conditioning_count),
            }
        )
    return records


def measure_production_receipt(data: Path, output: Path) -> dict[str, Any]:
    """Run the repaired and disabled arms and write their comparative receipt."""
    configure_dtypes()
    repaired_solve._validate_baseline()
    bank = json.loads(BANKED_CONTROL.read_text(encoding="utf-8"))
    banked_by_case = {
        (record["shot"], int(record["frame"])): record
        for record in bank["frame_records"]
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
        enabled = _solve_frame(mapped, seed, profile, conditioning_enabled=True)
        disabled = _solve_frame(mapped, seed, profile, conditioning_enabled=False)
        banked = banked_by_case[(case.shot, case.frame)]
        records.append(
            {
                "shot": case.shot,
                "frame": case.frame,
                "time_ms": time_ms,
                "repaired_conditioning_enabled": enabled,
                "repaired_conditioning_disabled_same_tree": disabled,
                "banked_pre_repair_conditioning_enabled": _banked_metrics(
                    banked, "measured_dimension_general_conditioning"
                ),
                "banked_pre_repair_conditioning_disabled": _banked_metrics(
                    banked, "current_tree_conditioning_disabled_control"
                ),
                "enabled_residual_no_worse_than_same_tree_disabled": (
                    enabled["terminal_relative_residual"]
                    <= disabled["terminal_relative_residual"]
                ),
            }
        )

    dimension_engagement = _dimension_engagement()
    source = Path("nova/equilibrium/fixed_point.py").read_text(encoding="utf-8")
    repaired_residual_gate = all(
        record["enabled_residual_no_worse_than_same_tree_disabled"]
        for record in records
    )
    production_engagement = sum(
        record["repaired_conditioning_enabled"]["intervention_count_out_of_89"]
        for record in records
    )
    receipt = {
        "artifact": "conditioning_shrinkage_repair",
        "completed_utc": datetime.now(UTC).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "measurement_scope": {
            "selection": "five banked score-blind circuit-current frames",
            "score_labels_read": False,
            "data_root": str(data),
            "newton_steps_per_frame": repaired_solve.NEWTON_STEPS,
            "projection_dimension": repaired_solve.GMRES_ITERATIONS,
            "banked_comparator": str(BANKED_CONTROL.relative_to(HERE)),
            "banked_comparator_sha256": _sha256(BANKED_CONTROL),
        },
        "comparison_contract": {
            "same_tree_pair_is_causal_control": True,
            "pair_difference": "condition ratio limit e versus infinity",
            "banked_pre_repair_values_are_context_only": True,
            "production_paths_changed_from_banked_source": _git(
                "diff",
                "--name-only",
                f"{bank['source_commit']}..HEAD",
                "--",
                "nova",
            ).splitlines(),
        },
        "mechanism": {
            "rule": (
                "try the raw fixed ladder first; use conditioned damping only "
                "after total raw-ladder refusal and only for a candidate that "
                "does not increase the current relative residual"
            ),
            "dataset_fitted_constants_absent": (
                "44.5" not in source and "27.781718445022726" not in source
            ),
            "dimension_specific_gate_absent": (
                "_PROJECTED_KRYLOV_CONDITION_DIMENSION" not in source
            ),
            "dimension_engagement": dimension_engagement,
        },
        "frame_records": records,
        "cohort_summary": {
            "repaired_total_interventions_out_of_445": production_engagement,
            "repaired_total_admitted_advances_out_of_445": sum(
                record["repaired_conditioning_enabled"][
                    "admitted_advance_count_out_of_89"
                ]
                for record in records
            ),
            "repaired_total_newton_step_equivalents": math.fsum(
                record["repaired_conditioning_enabled"][
                    "achieved_newton_step_equivalents"
                ]
                for record in records
            ),
            "disabled_total_newton_step_equivalents": math.fsum(
                record["repaired_conditioning_disabled_same_tree"][
                    "achieved_newton_step_equivalents"
                ]
                for record in records
            ),
            "frames_residual_no_worse_than_disabled": sum(
                record["enabled_residual_no_worse_than_same_tree_disabled"]
                for record in records
            ),
            "frame_count": len(records),
        },
        "conclusion": {
            "verdict": (
                "PASS"
                if repaired_residual_gate
                and production_engagement > 0
                and all(
                    record["intervention_count_out_of_1"] > 0
                    for record in dimension_engagement
                )
                else "FAIL"
            ),
            "residual_gate_passed": repaired_residual_gate,
            "production_engagement_retained": production_engagement > 0,
            "dimension_engagement_retained": all(
                record["intervention_count_out_of_1"] > 0
                for record in dimension_engagement
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
    """Run the production comparison when invoked as a script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repaired_solve.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = measure_production_receipt(arguments.data, arguments.output)
    print(json.dumps(receipt["cohort_summary"], sort_keys=True), flush=True)
    print(json.dumps(receipt["conclusion"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
