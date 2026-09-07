"""CPU contracts for the device-batched forward labeller."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.batched_labeller import (
    CENTROID_REPORTING_QUANTUM,
    BatchedLabeller,
    _centroid_pair,
)
from nova.equilibrium.constraint import ConstraintMultiplier
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.solve_request import default_forward_compilation_cache_root
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_persistent_compilation_cache

from tests.test_reduced_newton import machine as machine_fixture  # noqa: F401


STATE_RTOL = 1.0e-12
STATE_ATOL = 1.0e-14
FIGURE_PATH = (
    Path(__file__).parents[1]
    / "docs/figures/playable-forward-solve/batched-labeller/terminal-state-parity.png"
)

configure_persistent_compilation_cache(default_forward_compilation_cache_root())


def _offer_prescribed_currents(profile) -> np.ndarray:
    """Expose the fixture's conductor response as traced prescribed currents."""
    operator = profile.operator
    response = jnp.concatenate(
        (
            jnp.asarray(operator.grid.source_target),
            jnp.asarray(operator.wall.source_target),
        )
    )
    operator.prescribed_field = PrescribedCurrentField(
        response=response,
        current=jnp.zeros(response.shape[1]),
    )
    return np.zeros(response.shape[1])


def _write_state_parity_figure(actual, reference) -> float:
    """Plot elementwise relative differences and return their maximum."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    absolute = np.abs(actual - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(reference.dtype).tiny)
    relative = absolute / denominator
    maximum = float(np.max(relative, initial=0.0))

    figure = Figure(figsize=(8.0, 3.8), layout="constrained")
    axis = figure.subplots()
    for element, values in enumerate(relative):
        axis.semilogy(values, ".", markersize=2.5, label=f"element {element}")
    axis.axhline(STATE_RTOL, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("terminal-state element")
    axis.set_ylabel("absolute relative difference")
    axis.set_title(f"Maximum relative difference: {maximum:.6e}")
    axis.legend(loc="upper right")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    return maximum


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

    reference_state = np.broadcast_to(
        np.asarray(reference.state), np.asarray(result.state).shape
    )
    maximum_relative_difference = _write_state_parity_figure(
        result.state, reference_state
    )
    assert maximum_relative_difference <= STATE_RTOL
    np.testing.assert_allclose(
        np.asarray(result.state),
        reference_state,
        rtol=STATE_RTOL,
        atol=STATE_ATOL,
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
    np.testing.assert_array_equal(
        np.asarray(result.trips), [reference.active_set_iterations] * 2
    )

    reference_centroid = np.asarray(
        profile.current_moment_observation(
            reference.state,
            support=MomentIntegralSupport.ALL_DOMAIN,
        ).stack()[1:]
    )
    reference_centroid = (
        np.rint(reference_centroid / CENTROID_REPORTING_QUANTUM)
        * CENTROID_REPORTING_QUANTUM
    )
    np.testing.assert_array_equal(
        np.asarray(result.achieved_centroid),
        np.broadcast_to(reference_centroid, np.asarray(result.achieved_centroid).shape),
    )
    reference_masks, reference_topology = profile.operator.read(reference.state)
    reference_labels = profile._labelled_flux(
        reference.state, reference_masks, reference_topology
    )
    np.testing.assert_array_equal(
        np.asarray(result.labelled_flux.domain_label),
        np.broadcast_to(
            np.asarray(reference_labels.domain_label),
            np.asarray(result.labelled_flux.domain_label).shape,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(result.labelled_flux.lcfs_vertex_count),
        [np.asarray(reference_labels.lcfs_vertex_count)] * 2,
    )
    achieved_class = np.asarray(
        [profile.operator.read(state)[1].diverted for state in result.state]
    )
    np.testing.assert_array_equal(
        achieved_class, [np.asarray(reference_topology.diverted)] * 2
    )

    padded = labeller.solve(
        batch,
        active=np.asarray([True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(padded.state[0]), np.asarray(result.state[0])
    )
    np.testing.assert_array_equal(
        np.asarray(padded.converged[0]), np.asarray(result.converged[0])
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


def test_masked_conditioning_keeps_the_augmented_multiplier(machine_fixture):  # noqa: F811
    """Both guard branches retain the row slot and a conditioned row executes."""
    profile, seed = machine_fixture
    prescribed_current = _offer_prescribed_currents(profile)
    target_current = abs(float(np.sum(np.asarray(profile.operator.cell_current(seed)))))
    requested_class = int(TopologyClass.LIMITED)
    observed = np.asarray(
        profile.current_moment_observation(
            seed,
            support=MomentIntegralSupport.ALL_DOMAIN,
            target_current=target_current,
        ).stack()[1:]
    )
    batch = np.stack((np.asarray(seed), np.asarray(seed)))
    result = BatchedLabeller(
        profile,
        newton_steps=1,
        active_set_steps=1,
    ).solve(
        batch,
        prescribed_current=np.stack((prescribed_current, prescribed_current)),
        target_current=np.asarray([target_current, target_current]),
        requested_class=np.asarray([requested_class, requested_class]),
        reference_centroid=np.stack((observed, observed + np.asarray([0.0, 1.0]))),
        centroid_target=np.asarray([[observed[1]], [observed[1] + 1.0e-3]]),
        active=np.asarray([False, True]),
    )

    np.testing.assert_array_equal(np.asarray(result.conditioned), [False, True])
    np.testing.assert_array_equal(np.asarray(result.state[0]), np.asarray(seed))
    assert np.asarray(result.state).shape == batch.shape


def test_centroid_pair_matches_the_slice_response_context(machine_fixture):  # noqa: F811
    """The traced pair retains the response derived for its own slice context."""
    profile, seed = machine_fixture
    _offer_prescribed_currents(profile)
    target_current = abs(float(np.sum(np.asarray(profile.operator.cell_current(seed)))))
    requested_class = int(TopologyClass.LIMITED)
    target = float(
        np.asarray(
            profile.current_moment_observation(
                seed,
                support=MomentIntegralSupport.ALL_DOMAIN,
                target_current=target_current,
            ).centroid_z
        )
    )
    actual = _centroid_pair(
        profile,
        seed,
        target,
        requested_class=requested_class,
        target_current=target_current,
    )
    seeded = dataclasses.replace(
        actual,
        unknown=ConstraintMultiplier(multiplier_scale=jnp.asarray([1.0])),
    )
    (expected,), _selection = profile.derived_constraint_pairs(
        (seeded,),
        jnp.asarray(seed),
        requested_class=jnp.asarray(requested_class),
        target_current=jnp.asarray(target_current),
    )

    np.testing.assert_allclose(
        np.asarray(actual.unknown.direction),
        np.asarray(expected.unknown.direction),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(actual.unknown.ampere_scale),
        np.asarray(expected.unknown.ampere_scale),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
