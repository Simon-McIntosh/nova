"""Termination receipts from a real multi-trip constrained command."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zarr

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

from benchmarks import constraint_jacobian_receipt as globalisation  # noqa: E402
from nova.equilibrium import reduced_newton  # noqa: E402
from scripts.labeller_batch import shard  # noqa: E402

TERMINATION = reduced_newton.FixedPointTerminationReason


@pytest.fixture(scope="module")
def multi_trip_command():
    """Build the measured row-96 command from its banked free response."""
    prepared = shard.prepare_labeller()
    prepared = shard._prepared_with_polarity(
        prepared, shard._shot_polarity(globalisation.SHOT)
    )
    group = zarr.open_group(
        str(shard.SHOT_STORE / f"{globalisation.SHOT}.zarr"), mode="r"
    )["efm"]
    state = globalisation._row_state(prepared, group, 96)
    banked_payload = json.loads(
        globalisation.BANKED_RECEIPT.read_text(encoding="utf-8")
    )
    banked = {
        (int(item["row"]), str(item["circuit"])): item
        for item in banked_payload["rows_detail"]
    }
    banked_centroid = float(
        banked[(96, "p6_upper")]["perturbed_solves"]["plus"]["centroid_z_m"]
    )
    banked_delta = banked_centroid - state["free_centroid_z_m"]
    target = state["free_centroid_z_m"] + globalisation.MULTI_TRIP_SCALE * banked_delta
    active_names = globalisation._circuit_names(prepared.policy_evidence)
    direction = globalisation.comparison._direction(
        active_names, "p6_upper", state["current"].size
    )
    pair = globalisation._explicit_pair(prepared.profile, target, direction)
    return prepared, state, pair, target


@pytest.mark.slow
def test_last_permitted_trip_reports_the_arrived_iterate(multi_trip_command):
    """The measured eighth trip is convergence, not budget exhaustion."""
    prepared, state, pair, target = multi_trip_command
    outcome = globalisation._multi_trip_case(
        prepared,
        state,
        pair,
        target=target,
        expected_current_a=4_743.685,
        trip_limit=globalisation.ACTIVE_SET_TRIPS,
    )

    assert outcome["trip_count"] == globalisation.ACTIVE_SET_TRIPS
    assert outcome["capped_application_count"] == 5
    assert outcome["target_within_tolerance"]
    assert abs(outcome["centroid_error_m"]) < 1.0e-12
    assert outcome["converged"], (
        "row 96 arrived after %d trips at %.3e m but reported %s"
        % (
            outcome["trip_count"],
            outcome["centroid_error_m"],
            outcome["termination_reason"],
        )
    )
    assert outcome["termination_reason"] == "converged"


@pytest.mark.slow
def test_exhaustion_carries_the_terminal_error(multi_trip_command):
    """A genuinely unfinished command publishes both terminal error measures."""
    prepared, state, pair, _target = multi_trip_command
    result = reduced_newton.solve_constrained_reduced_newton(
        prepared.profile,
        state["free"].state,
        constraint_pairs=(pair,),
        requested_class=state["requested"],
        target_current=state["target_current"],
        prescribed_current=jnp.asarray(state["current"]),
        tolerance=globalisation.FIXED_POINT_CRITERION,
        newton_steps=globalisation.NEWTON_STEPS,
        active_set_steps=1,
        constraint_current_step_cap=globalisation.CURRENT_STEP_CAP_A,
    )

    assert not result.converged
    exhausted = TERMINATION.ACTIVE_SET_ITERATION_BUDGET_EXHAUSTED
    assert result.termination_reason == exhausted
    assert np.isfinite(result.terminal_residual)
    physical_error = float(np.asarray(result.constraints[0].physical_residual)[0])
    assert np.isfinite(physical_error)
    assert abs(physical_error) > globalisation.CENTROID_TOLERANCE_M
