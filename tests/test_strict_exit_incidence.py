"""Regression checks for the allocation-free strict-exit batch proof."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import strict_exit_incidence as incidence
from benchmarks.efit_forward_parity_slice import _profile_function
from nova.equilibrium.solve_request import SampledFluxFunction
from nova.equilibrium.source import DomainProfile, ForwardSource


def _converged_arm(executed_trips: int, terminal_residual: float, state_sha: str):
    return {
        "executed_trips": executed_trips,
        "no_op_trips": incidence.TRIP_LIMIT - executed_trips,
        "terminal_residual": terminal_residual,
        "termination": incidence._termination_name(incidence.CONVERGED_REASON),
        "terminal_state_sha256": state_sha,
    }


def test_batched_self_check_requires_explicit_batched_mode():
    with pytest.raises(ValueError, match="--self-check requires --batched"):
        incidence._validate_arguments(batched=False, self_check=True)


def test_select_member_keeps_scalar_receipt_leaves_unindexed():
    selected = incidence._select_member(
        {
            "per_member": incidence.jnp.asarray([3, 5]),
            "scalar": incidence.jnp.asarray(7),
        },
        1,
    )

    assert int(selected["per_member"]) == 5
    assert int(selected["scalar"]) == 7


def test_batch_member_result_selects_the_sequential_fallback_member():
    results = ("first", "second")

    assert incidence._batch_member_result(results, 1, stacked=False) == "second"


def test_cpu_self_check_requires_the_debug_allocation(monkeypatch):
    environment = {
        "SLURM_JOB_ID": "42",
        "SLURM_JOB_NAME": "cpu-proof",
        "SLURM_JOB_PARTITION": "all_debug",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_MEM_PER_NODE": str(64 * 1024),
        "SLURM_TIMELIMIT": "00:55:00",
        "TMPDIR": "/tmp",
        "JAX_PLATFORMS": "cpu",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        incidence.jax,
        "devices",
        lambda: [SimpleNamespace(platform="cpu", device_kind="cpu")],
    )

    receipt = incidence._require_cpu_self_check()

    assert receipt["job_id"] == 42
    assert receipt["partition"] == "all_debug"
    assert receipt["cpu_count"] == 4
    assert receipt["requested_memory_mib"] == 64 * 1024
    assert receipt["execution_mode"] == "slurm_cpu_self_check"


def test_mast_profile_tables_become_dynamic_leaves_without_numerical_change():
    psi_norm = np.linspace(0.0, 1.0, 65)
    stored_p_prime = np.square(psi_norm)[None, :]
    stored_ff_prime = np.sin(psi_norm)[None, :]
    expected_p_prime = -stored_p_prime[0] / incidence.TOTAL_FLUX_FACTOR
    expected_ff_prime = -stored_ff_prime[0] / incidence.TOTAL_FLUX_FACTOR
    original = ForwardSource(
        core=DomainProfile(
            p_prime=_profile_function(psi_norm, expected_p_prime),
            ff_prime=_profile_function(psi_norm, expected_ff_prime),
        ),
        boundary_pressure=3.0,
        boundary_field_function=4.0,
    )

    sampled = incidence._mast_source_with_sampled_profiles(
        original,
        {
            "psi_norm": psi_norm,
            "pprime": stored_p_prime,
            "ffprime": stored_ff_prime,
        },
        0,
    )

    assert isinstance(sampled.core.p_prime, SampledFluxFunction)
    assert isinstance(sampled.core.ff_prime, SampledFluxFunction)
    assert sampled.boundary_pressure == original.boundary_pressure
    assert sampled.boundary_field_function == original.boundary_field_function
    evaluation = incidence.jnp.linspace(-0.02, 1.02, 101)
    np.testing.assert_array_equal(
        np.asarray(sampled.core.p_prime(evaluation)),
        np.asarray(original.core.p_prime(evaluation)),
    )
    np.testing.assert_array_equal(
        np.asarray(sampled.core.ff_prime(evaluation)),
        np.asarray(original.core.ff_prime(evaluation)),
    )


def test_diiid_transport_supplies_content_addressed_artifact_evidence(
    tmp_path,
):
    coordinate_digest = (
        "a45135511161237ad38db8e6515b66bf79471b9eb719779281a37dbda9bfffd8"
    )
    evidence = incidence._diiid_machine_artifact_evidence(
        tmp_path,
        [
            {
                "coordinate_transport": "verified test transport",
                "wall_coordinate_sha256": coordinate_digest,
            },
            {
                "coordinate_transport": "verified test transport",
                "wall_coordinate_sha256": coordinate_digest,
            },
        ],
    )

    assert evidence["manifest_sha256"] == (
        incidence.DEFAULT_MACHINE_ARTIFACT_DIGEST.removeprefix("sha256:")
    )
    assert evidence["wall_coordinate_sha256"] == coordinate_digest
    assert evidence["cache"] == str(tmp_path)


def test_batch_rows_persist_before_cross_arm_identity_assertion():
    control_one = _converged_arm(4, 1.0e-12, "state-a")
    exited_one = dict(control_one)
    control_two = _converged_arm(incidence.TRIP_LIMIT, 2.0e-10, "state-b")
    exited_two = {
        "executed_trips": 6,
        "no_op_trips": incidence.TRIP_LIMIT - 6,
        "terminal_residual": 3.0e-5,
        "termination": incidence._termination_name(incidence.SETTLED_REASON),
        "terminal_state_sha256": "state-c",
    }

    rows = [
        incidence._strict_exit_member_row(
            "member one",
            "initial-one",
            "state",
            control_one,
            exited_one,
            incidence.jnp.asarray([0.5, 1.0, 2.0]),
            incidence.jnp.asarray([0.5, 1.0, 2.0]),
        ),
        incidence._strict_exit_member_row(
            "member two",
            "initial-two",
            "state",
            control_two,
            exited_two,
            incidence.jnp.asarray([0.5, 1.0, 3.0]),
            incidence.jnp.asarray([0.5, 1.0, 2.5]),
        ),
    ]

    incidence._assert_cross_arm_identity(rows)

    assert [row["identity"] for row in rows] == ["member one", "member two"]
    for row in rows:
        assert row["without_exit"]["termination"]
        assert row["with_exit"]["termination"]
        assert row["terminal_state_difference"]
    member_one, member_two = rows
    assert member_one["terminal_state_difference"] == {
        "max_absolute_flux_difference": 0.0,
        "without_exit_converged": True,
        "with_exit_converged": True,
    }
    assert member_one["terminal_state_bit_identical_where_both_arms_converged"] is True
    difference_two = member_two["terminal_state_difference"]
    assert difference_two["max_absolute_flux_difference"] == 0.5
    assert member_two["terminal_state_difference"]["without_exit_converged"] is True
    assert member_two["terminal_state_difference"]["with_exit_converged"] is False
    assert member_two["terminal_state_bit_identical_where_both_arms_converged"] is None


def test_cross_arm_identity_refuses_a_both_converged_member_with_differing_state():
    control = _converged_arm(4, 1.0e-12, "state-a")
    exited = _converged_arm(4, 1.0e-12, "state-d")
    rows = [
        incidence._strict_exit_member_row(
            "member one",
            "initial-one",
            "state",
            control,
            exited,
            incidence.jnp.asarray([0.5, 1.0, 2.0]),
            incidence.jnp.asarray([0.5, 1.1, 2.0]),
        )
    ]

    with pytest.raises(RuntimeError, match="changed terminal state bits"):
        incidence._assert_cross_arm_identity(rows)
