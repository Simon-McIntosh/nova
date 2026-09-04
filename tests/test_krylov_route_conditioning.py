"""Iteration-local mitigation for ill-conditioned Newton--Krylov steps."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.fixed_point import (
        KrylovActionQualification,
        kink_aware_newton_krylov,
        newton_krylov,
    )
    from nova.jax.config import configure_dtypes


def _diagonal_fixed_point(condition: float):
    """Return a map whose Newton action has the requested condition number."""
    diagonal = jnp.exp(jnp.linspace(0.0, -jnp.log(condition), 12))
    action = jnp.diag(diagonal)
    tangent = jnp.eye(12) - action
    offset = jnp.ones(12)
    return lambda state: tangent @ state + offset


def test_resolved_high_condition_step_is_undamped_and_reported_at_its_iteration():
    conditioned = newton_krylov(
        _diagonal_fixed_point(200.0),
        jnp.zeros(12),
        newton_steps=1,
        gmres_iterations=12,
        warmup=0,
        step_cap=1.0e6,
    )
    unconditioned = newton_krylov(
        _diagonal_fixed_point(200.0),
        jnp.zeros(12),
        newton_steps=1,
        gmres_iterations=12,
        warmup=0,
        step_cap=1.0e6,
        krylov_condition_limit=jnp.inf,
    )

    assert float(conditioned.inner_iteration_krylov_reductions[0]) <= np.sqrt(
        np.finfo(np.float64).eps
    )
    # A trusted linear solve stays undamped; only an unresolved solve may condition.
    assert int(conditioned.krylov_conditioning_count) == 0
    np.testing.assert_allclose(
        float(conditioned.maximum_projected_krylov_condition), 200.0, rtol=2.0e-6
    )
    # A trusted linear solve stays undamped; only an unresolved solve may condition.
    np.testing.assert_allclose(conditioned.state, unconditioned.state, rtol=2.0e-6)
    assert (
        KrylovActionQualification(int(conditioned.krylov_action_qualification))
        is KrylovActionQualification.ACCEPTED
    )


def test_well_conditioned_step_is_unchanged_and_reports_no_intervention():
    conditioned = newton_krylov(
        _diagonal_fixed_point(4.0),
        jnp.zeros(12),
        newton_steps=1,
        gmres_iterations=12,
        warmup=0,
        step_cap=1.0e6,
    )
    control = newton_krylov(
        _diagonal_fixed_point(4.0),
        jnp.zeros(12),
        newton_steps=1,
        gmres_iterations=12,
        warmup=0,
        step_cap=1.0e6,
        krylov_condition_limit=jnp.inf,
    )

    assert int(conditioned.krylov_conditioning_count) == 0
    np.testing.assert_allclose(
        float(conditioned.maximum_projected_krylov_condition), 4.0, rtol=2.0e-6
    )
    np.testing.assert_array_equal(conditioned.state, control.state)


def test_conditioning_receipt_is_fixed_shape_under_jit_and_vmap():
    def solve(condition):
        return newton_krylov(
            _diagonal_fixed_point(condition),
            jnp.zeros(12),
            newton_steps=1,
            gmres_iterations=12,
            warmup=0,
            step_cap=1.0e6,
        )

    result = jax.jit(jax.vmap(solve))(jnp.asarray([4.0, 80.0, 200.0]))
    assert np.all(
        np.asarray(result.inner_iteration_krylov_reductions)
        <= np.sqrt(np.finfo(np.float64).eps)
    )
    # Trusted linear solves stay undamped; only unresolved solves may condition.
    np.testing.assert_array_equal(result.krylov_conditioning_count, [0, 0, 0])
    np.testing.assert_allclose(
        result.maximum_projected_krylov_condition,
        [4.0, 80.0, 200.0],
        rtol=2.0e-6,
    )
    assert result.state.shape == (3, 12)


def test_kink_aware_route_carries_the_same_conditioning_receipt():
    result = kink_aware_newton_krylov(
        _diagonal_fixed_point(200.0),
        jnp.zeros(12),
        strategy="nonmonotone",
        newton_steps=1,
        gmres_iterations=12,
        warmup=0,
        step_cap=1.0e6,
    )

    # A trusted linear solve stays undamped; only an unresolved solve may condition.
    assert int(result.krylov_conditioning_count) == 0
    np.testing.assert_allclose(
        float(result.maximum_projected_krylov_condition), 200.0, rtol=2.0e-6
    )


def _digest(path: Path) -> str:
    """Return the content identity of one measurement input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conditioning_receipt(result) -> dict[str, int | float]:
    """Return the host-readable conditioning fields of one solve."""
    return {
        "conditioning_step_count": int(result.fixed_point.krylov_conditioning_count),
        "maximum_projected_condition": float(
            result.fixed_point.maximum_projected_krylov_condition
        ),
    }


def measure_receipt(output: Path) -> dict[str, Any]:
    """Measure parity and the event-resolved trajectory on the frozen cohort."""
    from benchmarks import parity_divergence_attribution as parity

    configure_dtypes()
    bank_path = Path(
        "docs/figures/forward-operator-refinement/event-resolved-amplification.json"
    )
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    preceding_separation: float | None = None
    preceding_eager_count = 0
    preceding_compiled_count = 0

    for shot, slice_index in parity._case_rows(parity.SHOT_STORE):
        profile, seed, target_current, time_s = parity._build_case(
            parity.SHOT_STORE, shot, slice_index
        )
        mapped = profile.flux_map(target_current=target_current)
        eager_map = mapped(seed.flux)
        compiled_map = jax.jit(mapped)(seed.flux)
        jax.block_until_ready(compiled_map)
        one_map = parity._difference(eager_map, compiled_map)
        eager, compiled = parity._solve_pair(
            profile, seed.flux, target_current, parity.NONLINEAR_UPDATES
        )
        solved_flux = parity._difference(eager.flux, compiled.flux)
        cases.append(
            {
                "shot": shot,
                "slice_index": slice_index,
                "time_s": time_s,
                "single_map_application": one_map,
                "solved_flux": solved_flux,
                "eager": _conditioning_receipt(eager),
                "compiled": _conditioning_receipt(compiled),
            }
        )

        if (shot, slice_index) != parity.TRAJECTORY_CASE:
            continue
        for update in range(1, parity.NONLINEAR_UPDATES + 1):
            eager, compiled = parity._solve_pair(
                profile, seed.flux, target_current, update
            )
            difference = parity._difference(eager.flux, compiled.flux)
            separation = float(difference["maximum_absolute_difference"])
            growth = (
                None
                if preceding_separation in (None, 0.0)
                else separation / preceding_separation
            )
            eager_receipt = _conditioning_receipt(eager)
            compiled_receipt = _conditioning_receipt(compiled)
            eager_count = int(eager_receipt["conditioning_step_count"])
            compiled_count = int(compiled_receipt["conditioning_step_count"])
            trajectory.append(
                {
                    "nonlinear_update": update,
                    "maximum_absolute_separation": separation,
                    "growth_from_preceding_update": growth,
                    "eager_residual": float(eager.fixed_point.residual),
                    "compiled_residual": float(compiled.fixed_point.residual),
                    "eager_conditioned_at_update": (
                        eager_count > preceding_eager_count
                    ),
                    "compiled_conditioned_at_update": (
                        compiled_count > preceding_compiled_count
                    ),
                    "eager": eager_receipt,
                    "compiled": compiled_receipt,
                }
            )
            preceding_separation = separation
            preceding_eager_count = eager_count
            preceding_compiled_count = compiled_count

    if len(trajectory) != parity.NONLINEAR_UPDATES:
        raise RuntimeError("the frozen trajectory case was not measured")
    nonzero = [
        row["maximum_absolute_separation"]
        for row in trajectory
        if row["maximum_absolute_separation"] > 0.0
    ]
    cumulative_growth = (
        0.0
        if not nonzero
        else trajectory[-1]["maximum_absolute_separation"] / nonzero[0]
    )
    burst_updates = [
        row["nonlinear_update"]
        for row in trajectory
        if row["growth_from_preceding_update"] is not None
        and row["growth_from_preceding_update"] >= 10.0
    ]
    maximum_solved_flux_difference = max(
        float(row["solved_flux"]["maximum_absolute_difference"]) for row in cases
    )
    maximum_one_map_difference = max(
        float(row["single_map_application"]["maximum_absolute_difference"])
        for row in cases
    )
    receipt = {
        "schema": "nova-krylov-route-conditioning/1.0",
        "completed_utc": datetime.now(UTC).isoformat(),
        "source": {
            "fixed_point_sha256": _digest(Path("nova/equilibrium/fixed_point.py")),
            "bank": str(bank_path),
            "bank_sha256": _digest(bank_path),
        },
        "mechanism": {
            "discriminator": "rectangular Arnoldi projection condition of I-J",
            "condition_limit": 44.5,
            "calibrated_krylov_dimension": 12,
            "mitigation": (
                "multiply the GMRES step by the cubed ratio of the banked quiet "
                "condition to the measured condition above the trigger limit"
            ),
            "fixed_shape": True,
            "data_dependent_control_flow": False,
            "jit_and_vmap_safe": True,
            "banked_burst_median_condition": bank["event_discrimination"][
                "burst_median_krylov_condition"
            ],
            "banked_quiet_median_condition": bank["event_discrimination"][
                "quiet_median_krylov_condition"
            ],
        },
        "trajectory_case": {
            "shot": parity.TRAJECTORY_CASE[0],
            "slice_index": parity.TRAJECTORY_CASE[1],
            "nonlinear_updates": parity.NONLINEAR_UPDATES,
            "per_update": trajectory,
            "new_cumulative_separation_growth": cumulative_growth,
            "banked_cumulative_separation_growth": bank["trajectory"][
                "cumulative_separation_growth"
            ],
            "alternate_seed_cumulative_growth": bank["predictions"][
                "different_seed_direction"
            ]["alternate_cumulative_separation_growth"],
            "remaining_burst_updates": burst_updates,
            "remaining_burst_count": len(burst_updates),
            "banked_burst_updates": bank["event_discrimination"]["burst_updates"],
            "banked_burst_count": len(bank["event_discrimination"]["burst_updates"]),
        },
        "held_out_parity": {
            "case_count": len(cases),
            "cases": cases,
            "new_maximum_flux_absolute_difference": maximum_solved_flux_difference,
            "banked_maximum_flux_absolute_difference": bank["trajectory"][
                "terminal_separation"
            ],
            "new_maximum_single_map_absolute_difference": maximum_one_map_difference,
            "banked_single_map_floor": 2.220446049250313e-16,
            "retired_tolerance": 1.0e-10,
            "verdict_against_retired_tolerance": "DECLINED",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the quantitative receipt from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/figures/diiid-forward-onboarding/krylov-route-conditioning.json"
        ),
    )
    arguments = parser.parse_args()
    receipt = measure_receipt(arguments.output)
    print(
        json.dumps(
            {
                "burst_updates": receipt["trajectory_case"]["remaining_burst_updates"],
                "cumulative_growth": receipt["trajectory_case"][
                    "new_cumulative_separation_growth"
                ],
                "maximum_flux_difference": receipt["held_out_parity"][
                    "new_maximum_flux_absolute_difference"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
