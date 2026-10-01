r"""Contract for the fused same-state read on the integral-state branch.

``ForwardProfile._integral_state`` serves two requests from the same trial
flux: the discrete domain/topology read and the current-moment read.  The
qualification pass is the expensive part and both requests are post-processing
of the value it returns, so each branch of that method serves them from one
pass.  The two arms differ in which moments they form:

* the clipped-support arm (``use_linear_moments`` true) reaches
  ``current_moments_and_observation``, which derives moments, observation
  measure, domain masks and achieved topology from one ``_support_partition``;
* the point-current arm reaches ``read_with_current_moments``, which derives
  the moments from the masks the read already returned rather than qualifying
  the same flux a second time.

Both are exact substitutions, and the point arm is the one whose second
qualification is removable: the separate request decomposes into ``read`` plus
``cell_current_moments``, and the latter issues its own ``_fixed_design_read``.
What is asserted here is that the substitution changes no value — the fused
route is held to the two-read route on the state it returns, and a full host
solve is held to the same terminal equilibrium with the fused entry point
replaced by the two-read decomposition.
"""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    from nova.equilibrium.forward_operator import ForwardFluxOperator

from nova.equilibrium.topology import NoQualifiedAxisError
from tests.test_equilibrium_forward_solve import _with_direct_samples

pytest_plugins = ("tests.test_equilibrium_forward_solve",)

#: Host-map evaluations the terminal-state pairs are driven for. Each pair is
#: compared against itself, so the budget sets only how far the shared map is
#: driven, not what "terminal" means.
HOST_EVALUATIONS = 12


def _assert_same_topology(left, right):
    """Hold two achieved topology reads to the same values."""
    for name in ("axis", "boundary", "wall_point"):
        np.testing.assert_array_equal(
            np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        )
    for name in (
        "axis_flux",
        "boundary_flux",
        "x_point",
        "x_point_flux",
        "wall_point_flux",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        )


def _assert_same_moments(left, right):
    """Hold two current-moment triples to the same values."""
    for observed, expected in zip(left, right, strict=True):
        np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def _counted_fixed_design_reads(action):
    """Return how many qualification passes ``action`` issues."""
    original = ForwardFluxOperator._fixed_design_read
    counter = {"reads": 0}

    def counted(self, *args, **kwargs):
        """Count one qualification pass, then run it."""
        counter["reads"] += 1
        return original(self, *args, **kwargs)

    with mock.patch.object(ForwardFluxOperator, "_fixed_design_read", counted):
        action()
    return counter["reads"]


def _unfused_point_read(self, psi, requested_class=None):
    """Serve the two requests from the two reads the fusion removes."""
    masks, topology = self.read(psi, requested_class)
    return masks, topology, self.cell_current_moments(psi, requested_class)


def _unfused_linear_read(self, psi, requested_class=None):
    """Serve the two requests from the two reads the fusion removes.

    The two requests disagree about unqualified states: ``read`` raises on the
    host when no axis candidate resolves, while the fused route reaches only
    ``_support_partition``, which returns the achieved labels either way.  Where
    the separate request would refuse a state the fused route serves, the
    substitute takes the partition's own masks and topology, so the comparison
    runs over the states both routes serve.
    """
    partition = self._support_partition(psi, requested_class)
    try:
        masks, topology = self.read(psi, requested_class)
    except NoQualifiedAxisError:
        masks, topology = partition[0], partition[1]
    return (
        self.cell_current_moments(psi, requested_class),
        self._clipped_integral_measure(partition),
        masks,
        topology,
    )


def test_the_point_arm_serves_both_requests_from_one_read(machine):
    """The fused point read is the two-read decomposition, one pass cheaper."""
    profile, seed, _vacuum = machine
    operator = profile.operator
    assert operator.use_linear_moments is False

    fused = operator.read_with_current_moments(seed)
    fused_reads = _counted_fixed_design_reads(
        lambda: operator.read_with_current_moments(seed)
    )

    masks, topology = operator.read(seed)
    unfused = (masks, topology, operator.cell_current_moments(seed))
    unfused_reads = _counted_fixed_design_reads(
        lambda: (operator.read(seed), operator.cell_current_moments(seed)),
    )

    assert fused_reads == 1
    assert unfused_reads == 2
    np.testing.assert_array_equal(
        np.asarray(fused[0].label), np.asarray(unfused[0].label)
    )
    np.testing.assert_array_equal(
        np.asarray(fused[0].psi_norm), np.asarray(unfused[0].psi_norm)
    )
    _assert_same_topology(fused[1], unfused[1])
    _assert_same_moments(fused[2], unfused[2])


def test_the_linear_arm_serves_both_requests_from_one_partition(machine):
    """The fused linear read carries the moment request's own support masks."""
    profile, seed, _vacuum = machine
    operator, sampled_seed = _with_direct_samples(profile, seed)
    assert operator.use_linear_moments is True

    moments, measure, masks, topology = operator.current_moments_and_observation(
        sampled_seed
    )
    fused_reads = _counted_fixed_design_reads(
        lambda: operator.current_moments_and_observation(sampled_seed),
    )
    two_request_reads = _counted_fixed_design_reads(
        lambda: (
            operator.read(sampled_seed),
            operator.cell_current_moments(sampled_seed),
        ),
    )

    assert fused_reads == 1
    assert two_request_reads == 2

    _assert_same_moments(moments, operator.cell_current_moments(sampled_seed))
    _reference_masks, reference_topology = operator.read(sampled_seed)
    _assert_same_topology(topology, reference_topology)
    np.testing.assert_array_equal(
        np.asarray(measure.masks.label), np.asarray(masks.label)
    )


def test_the_point_arm_holds_the_terminal_state_of_the_two_read_route(machine):
    """A solve through the fused point read lands on the two-read terminal state."""
    profile, seed, _vacuum = machine
    fused = profile.solve(seed, route="host", evaluations=HOST_EVALUATIONS)

    calls = {"unfused": 0}

    def counted_unfused(self, psi, requested_class=None):
        """Serve the requests the pre-fusion way, counting the substitution."""
        calls["unfused"] += 1
        return _unfused_point_read(self, psi, requested_class)

    with mock.patch.object(
        ForwardFluxOperator, "read_with_current_moments", counted_unfused
    ):
        unfused = profile.solve(seed, route="host", evaluations=HOST_EVALUATIONS)

    assert calls["unfused"] >= 1
    np.testing.assert_array_equal(np.asarray(fused.flux), np.asarray(unfused.flux))
    np.testing.assert_array_equal(
        np.asarray(fused.moments.plasma_current),
        np.asarray(unfused.moments.plasma_current),
    )


def test_the_linear_arm_holds_the_terminal_state_of_the_two_read_route(machine):
    """A solve through the fused linear read lands on the two-read terminal state."""
    profile, seed, _vacuum = machine
    operator, sampled_seed = _with_direct_samples(profile, seed)
    sampled = replace(profile, operator=operator)

    fused = sampled.solve(sampled_seed, route="host", evaluations=HOST_EVALUATIONS)

    calls = {"unfused": 0}

    def counted_unfused(self, psi, requested_class=None):
        """Serve the requests the pre-fusion way, counting the substitution."""
        calls["unfused"] += 1
        return _unfused_linear_read(self, psi, requested_class)

    with mock.patch.object(
        ForwardFluxOperator, "current_moments_and_observation", counted_unfused
    ):
        unfused = sampled.solve(
            sampled_seed, route="host", evaluations=HOST_EVALUATIONS
        )

    assert calls["unfused"] >= 1
    np.testing.assert_array_equal(np.asarray(fused.flux), np.asarray(unfused.flux))
