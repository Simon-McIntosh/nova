"""The achieved-moment read carries the class the solve used.

A diagnostic read that rebuilds a topology from a solved flux must accept
the same ``requested_class`` the solve ran with and forward it into
``_integral_state``, so a converged slice is read under the admission regime
that produced it rather than through an unconstrained emergent read.
Without the pass-through the class is dropped at the call boundary and the
emergent read may refuse a flux the classed solve already converged.

Every profile here observes the seam through a stub ``_integral_state`` that
records the ``requested_class`` it receives, exactly as the public
observation kernels do: the device under test is the forward call inside
``current_moment_observation``, and the stub is the oracle for whether the
class actually arrives.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.observation import (
    ConstraintPinSet,
    MomentIntegralSupport,
    MomentPin,
    PinUncertainty,
)
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.jax.config import configure_dtypes

COORDINATE = np.asarray([[0.8, -0.1], [1.0, 0.0], [1.2, 0.1], [1.4, 0.0]])
CELL_CURRENT = jnp.asarray([100.0, 200.0, 300.0, 400.0])
MASKS = SimpleNamespace(core=jnp.ones(4, dtype=bool))
TOPOLOGY = SimpleNamespace(
    axis_flux=jnp.asarray(0.0),
    boundary_flux=jnp.asarray(1.0),
    flux_span=jnp.asarray(1.0),
    diverted=jnp.asarray(False),
)
SUPPORT_INTEGRALS = SimpleNamespace(
    volume=jnp.zeros(4),
    area=jnp.ones(4),
    radial_volume=jnp.zeros(4),
    cell_current=CELL_CURRENT,
    pressure_volume=jnp.zeros(4),
    field_volume=jnp.zeros(4),
)


def _refusing_read_profile():
    """Return a profile whose unconstrained read refuses the flux.

    The stub ``_integral_state`` models the two admission regimes the plan
    measured: an unconstrained read must find its boundary emergently and
    raises ``NoQualifiedAxisError``, while the same flux read under the class
    the solve used is admitted and returns a finite current image.
    """
    profile = object.__new__(ForwardProfile)
    profile.operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=COORDINATE), use_linear_moments=False
    )

    def integral_state(flux, requested_class=None, target_current=None):
        del flux, target_current
        if requested_class is None:
            raise NoQualifiedAxisError(
                "no qualified magnetic-axis candidate has a resolved component"
            )
        return (
            SimpleNamespace(cell_current=CELL_CURRENT),
            SUPPORT_INTEGRALS,
            MASKS,
            TOPOLOGY,
            None,
        )

    profile._integral_state = integral_state
    return profile


def _recording_read_profile():
    """Return a profile that records every class it is read under.

    A flux that reads cleanly without a class yields the same current image
    under any class; the observation records which classes were supplied so a
    test can assert both that the argument is forwarded and that forwarding
    it does not move the moment.
    """
    profile = object.__new__(ForwardProfile)
    profile.operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=COORDINATE), use_linear_moments=False
    )
    seen: list[object] = []

    def integral_state(flux, requested_class=None, target_current=None):
        del flux, target_current
        seen.append(requested_class)
        return (
            SimpleNamespace(cell_current=CELL_CURRENT),
            SUPPORT_INTEGRALS,
            MASKS,
            TOPOLOGY,
            None,
        )

    profile._integral_state = integral_state
    return profile, seen


def test_unconstrained_read_refusal_recovers_with_the_solved_class():
    configure_dtypes()
    profile = _refusing_read_profile()
    flux = jnp.zeros(4)
    support = MomentIntegralSupport.ALL_DOMAIN

    with pytest.raises(NoQualifiedAxisError, match="no qualified"):
        profile.current_moment_observation(flux, support=support)

    observed = profile.current_moment_observation(
        flux, support=support, requested_class=TopologyClass.LIMITED
    )
    assert np.isfinite(float(observed.plasma_current))
    assert np.isfinite(float(observed.centroid_z))
    assert observed.support is support


def test_class_argument_changes_admission_not_the_moment():
    configure_dtypes()
    profile, seen = _recording_read_profile()
    flux = jnp.zeros(4)
    support = MomentIntegralSupport.ALL_DOMAIN

    unclassed = profile.current_moment_observation(flux, support=support)
    assert seen == [None]

    classed = profile.current_moment_observation(
        flux, support=support, requested_class=TopologyClass.DIVERTED
    )
    assert seen == [None, TopologyClass.DIVERTED]

    assert classed.plasma_current == unclassed.plasma_current
    assert classed.centroid_z == unclassed.centroid_z
    assert classed.centroid_r == unclassed.centroid_r


def test_current_moment_map_forwards_the_solved_class():
    configure_dtypes()
    profile, seen = _recording_read_profile()
    flux = jnp.zeros(4)
    support = MomentIntegralSupport.ALL_DOMAIN

    profile.current_moment_map(
        flux, support=support, requested_class=TopologyClass.LIMITED
    )
    assert seen == [TopologyClass.LIMITED]

    profile.current_moment_map(flux, support=support)
    assert seen == [TopologyClass.LIMITED, None]


def test_integral_observation_forwards_the_solved_class():
    configure_dtypes()
    profile, seen = _recording_read_profile()
    flux = jnp.zeros(4)

    profile.integral_observation(flux, requested_class=TopologyClass.LIMITED)
    assert seen == [TopologyClass.LIMITED]

    profile.integral_observation(flux)
    assert seen == [TopologyClass.LIMITED, None]


def test_constraint_residual_forwards_the_solved_class():
    configure_dtypes()
    profile, seen = _recording_read_profile()
    flux = jnp.zeros(4)
    pins = ConstraintPinSet(
        moments=(
            MomentPin(
                "plasma_current",
                1000.0,
                PinUncertainty(1.0, "A", "trusted current interval"),
                MomentIntegralSupport.ALL_DOMAIN,
            ),
        )
    )

    profile.constraint_residual(flux, pins, requested_class=TopologyClass.LIMITED)
    assert seen == [TopologyClass.LIMITED]


def test_curved_boundary_support_promotes_every_cut_cell_before_moment_selection():
    """A centroid-excluded cell cut by the curve still carries current."""
    configure_dtypes()
    cells = (
        np.asarray([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]),
        np.asarray([[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]]),
    )
    centres = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    atomic_mesh = AtomicCellMesh.from_cells(cells, centroids=centres)
    operator = object.__new__(ForwardFluxOperator)
    operator.polarity = 1
    operator.moment_geometry = SimpleNamespace(atomic_mesh=atomic_mesh)
    operator._support_curve_centre = centres
    operator._support_curve_scale = np.ones((2, 2))
    operator.shared_node_flux = lambda flux: flux
    operator.support_flux_coefficients = lambda *_args: jnp.asarray(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    labels = jnp.asarray([0, 1], dtype=jnp.int8)
    masks = DomainMasks(label=labels, psi_norm=jnp.asarray([0.0, 1.0]))
    topology = SimpleNamespace(boundary_flux=jnp.asarray(0.0))
    physical = jnp.asarray(1.0 - atomic_mesh.node_coordinates[:, 0])

    support = ForwardFluxOperator._profile_support(
        operator, masks, topology, physical, jnp.zeros(1)
    )

    assert bool(support.boundary[0])
    assert bool(support.included[0])
    assert float(support.area[0]) > 0.0
    promoted = ForwardFluxOperator._moment_support_masks(masks, support)
    assert bool(promoted.profile_participation[0])
