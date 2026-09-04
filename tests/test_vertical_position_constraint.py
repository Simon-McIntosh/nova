"""Vertical current-centroid row expressed through the generic protocol."""

from __future__ import annotations

from types import MethodType

import jax.numpy as jnp
import numpy as np

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.observation import MomentIntegralSupport
from nova.jax.config import configure_dtypes
from tests.test_equilibrium_constraint_protocol import _profile


class _CentroidObservation:
    def __init__(self, height):
        self.centroid_z = height

    def value(self, name):
        if name != "centroid_z":
            raise KeyError(name)
        return self.centroid_z


def test_vertical_centroid_reports_physical_circuit_compensation() -> None:
    configure_dtypes()
    profile = _profile()

    def observe(self, flux, *, support, target_current=None):
        del self, target_current
        assert support is MomentIntegralSupport.ALL_DOMAIN
        return _CentroidObservation(0.25 * flux[0] - 0.5 * flux[1])

    profile.current_moment_observation = MethodType(observe, profile)
    pair = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=CircuitCurrentUnknown(
            direction=jnp.asarray([1.0]),
            ampere_scale=jnp.asarray([2000.0]),
        ),
        binding=ConstraintBinding(
            target=jnp.asarray([0.5]),
            tolerance=jnp.asarray([1.0e-8]),
            scale=jnp.asarray([0.5]),
            initial_unknown=jnp.asarray([0.0]),
            payload=None,
            policy="imposed",
        ),
    )

    result = profile._solve_augmented_constraints(
        jnp.asarray([0.25, 0.0]),
        None,
        constraint_pairs=(pair,),
        warmup=0,
        gmres_iterations=3,
        active_set_steps=2,
        stop_on_active_set_settlement=False,
    )

    record = result.constraints[0]
    np.testing.assert_allclose(result.flux, [2.0, 0.0], atol=1.0e-8)
    np.testing.assert_allclose(record.observed, [0.5], atol=1.0e-8)
    np.testing.assert_allclose(record.physical_unknown, [2.0], atol=1.0e-6)
    assert bool(record.qualified[0])
    assert pair.binding.policy == "imposed"
