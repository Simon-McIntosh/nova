"""CPU contracts for the device-batched forward labeller."""

from __future__ import annotations

import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.batched_labeller import BatchedLabeller

from tests.test_reduced_newton import machine as machine_fixture  # noqa: F401


def test_two_elements_match_compiled_route_and_padded_batch(machine_fixture):  # noqa: F811
    """Identical slices remain identical when sharded and when padded."""
    profile, seed = machine_fixture
    batch = np.stack((np.asarray(seed), np.asarray(seed)))
    labeller = BatchedLabeller(profile, newton_steps=1, active_set_steps=1)

    result = labeller.solve(batch)
    reference = reduced_newton.solve_reduced_newton_compiled(
        profile.operator,
        seed,
        newton_steps=1,
        active_set_steps=1,
    )

    np.testing.assert_array_equal(
        np.asarray(result.state[0]), np.asarray(reference.state)
    )
    np.testing.assert_array_equal(
        np.asarray(result.state[0]), np.asarray(result.state[1])
    )
    np.testing.assert_array_equal(
        np.asarray(result.converged), [reference.converged] * 2
    )
    np.testing.assert_array_equal(
        np.asarray(result.termination), [reference.termination_reason] * 2
    )

    padded = labeller.solve(
        np.concatenate((batch, batch[:1]), axis=0),
        active=np.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(padded.state[:2]), np.asarray(result.state)
    )
    np.testing.assert_array_equal(
        np.asarray(padded.converged[:2]), np.asarray(result.converged)
    )


def test_result_contains_fixed_shape_topology_fields(machine_fixture):  # noqa: F811
    """The topology payload is returned with the state, not read per field."""
    profile, seed = machine_fixture
    result = BatchedLabeller(profile, newton_steps=1, active_set_steps=1).solve(
        np.stack((np.asarray(seed), np.asarray(seed)))
    )

    labels = result.labelled_flux
    assert labels.psi.shape[0] == 2
    assert labels.domain_label.shape[0] == 2
    assert labels.o_point.shape == (2, 2)
    assert result.achieved_centroid.shape == (2, 2)
