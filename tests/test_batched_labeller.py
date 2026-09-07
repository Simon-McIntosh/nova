"""CPU contracts for the device-batched forward labeller."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from nova.equilibrium import reduced_newton
from nova.equilibrium.batched_labeller import (
    CENTROID_REPORTING_QUANTUM,
    BatchedLabeller,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.solve_request import default_forward_compilation_cache_root
from nova.jax.config import configure_persistent_compilation_cache

from tests.test_reduced_newton import machine as machine_fixture  # noqa: F401


STATE_RTOL = 1.0e-12
STATE_ATOL = 1.0e-14
FIGURE_PATH = (
    Path(__file__).parents[1]
    / "docs/figures/playable-forward-solve/batched-labeller/terminal-state-parity.png"
)

configure_persistent_compilation_cache(default_forward_compilation_cache_root())


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
