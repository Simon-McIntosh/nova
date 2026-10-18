"""Vertical current-centroid pin on the public augmented forward solve."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    CurrentCentroidConstraint,
    UniformRadialFieldUnknown,
    VerticalPositionPin,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.observation import CurrentMomentObservation, MomentIntegralSupport
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes


ELONGATION = 1.8
RADIAL_HALF_WIDTH = 0.2
FREE_HEIGHT = 0.5
SEED = np.asarray([1.0, 1.0, 0.30, 1.0, 1.0, 0.90], dtype=np.float64)
TARGETS = (0.7, 0.9)


def _carrier_residual(flux):
    first, first_r, first_z, second, second_r, second_z = flux
    low, high = first_z / first, second_z / second
    height = CurrentCentroidConstraint.booked_height(flux.reshape((2, 3)))
    return jnp.stack(
        (
            first - 1.0,
            second - 1.0,
            first_r - first,
            second_r - second,
            (high - low) / (2 * RADIAL_HALF_WIDTH) - ELONGATION,
            (height - FREE_HEIGHT) / ELONGATION**2,
        )
    )


class _CarrierOperator:
    """A six-moment flux map with one declared uniform radial-field image."""

    def __init__(self):
        self.grid = SimpleNamespace(
            coordinate=jnp.asarray([[1.0, 0.0]] * 5 + [[1.0, -1.0 / (2 * jnp.pi)]])
        )
        self.wall = SimpleNamespace(coordinate=jnp.zeros((0, 2)))
        self.sample = None
        self.source = SimpleNamespace(closure_degrees=0)
        self.prescribed_current_field = None
        self.program_identity = "elongated-moment-carrier"

    def with_source(self, source):
        del source
        return self

    def with_clip_mode(self, clip_mode):
        del clip_mode
        return self

    def flux_map(self, *args):
        del args
        return lambda flux: flux - _carrier_residual(flux)

    def flux_map_with_shadow(self, *args):
        del args
        return lambda flux, shadow: jnp.where(
            shadow, flux, flux - _carrier_residual(flux)
        )

    def residual_shadow_mask(self, flux, requested_class=None, previous_shadow=None):
        del requested_class, previous_shadow
        return jnp.zeros_like(flux, dtype=bool)


class _CarrierEquilibrium(NamedTuple):
    flux: jax.Array
    fixed_point: object
    constraints: tuple[object, ...]
    finite: object
    topology: object = None
    normalisation: object = None
    constraint_outer: object = None


def _profile():
    profile = object.__new__(ForwardProfile)
    profile.operator = _CarrierOperator()
    profile.newton_steps = 8
    profile._request_compilation_cache = set()

    def receipt(flux, history, *_args, constraints=(), **_kwargs):
        return _CarrierEquilibrium(
            flux, history, constraints, SimpleNamespace(passed=jnp.asarray(True))
        )

    def observe(self, flux, *, support, requested_class=None, target_current=None):
        del requested_class, target_current
        assert support is MomentIntegralSupport.ALL_DOMAIN
        moments = flux.reshape((2, 3))
        total = jnp.sum(moments[:, 0])
        return CurrentMomentObservation(
            total,
            jnp.sum(moments[:, 1]) / total,
            CurrentCentroidConstraint.booked_height(moments),
            support,
        )

    profile._receipt = receipt
    profile.current_moment_observation = MethodType(observe, profile)
    return profile


def _request(profile, target, *, active=True):
    return ForwardSolveRequest.from_defaults(
        carrier_identity="elongated-moment-carrier",
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(SEED),
        policy_overrides={
            "newton_steps": 8,
            "gmres_iterations": 7,
            "active_set_steps": 2,
            "kernel_tolerance": 1e-12,
            "qualification_tolerance": 1e-12,
            "current_pin": False,
            "compilation_cache": False,
        },
        vertical_pin=VerticalPositionPin(target, active=active),
    )


def _booked_terminal_height(flux):
    moments = np.asarray(flux).reshape((2, 3))
    return float(np.sum(moments[:, 2]) / np.sum(moments[:, 0]))


def _terminal_elongation(flux):
    moments = np.asarray(flux).reshape((2, 3))
    low, high = moments[:, 2] / moments[:, 0]
    return float((high - low) / (2 * RADIAL_HALF_WIDTH))


def test_vertical_pin_declares_a_physical_field_source():
    configure_dtypes()
    operator = _CarrierOperator()
    radial = VerticalPositionPin(0.7, source_scale=1e-3).pair(operator)
    assert isinstance(radial.unknown, UniformRadialFieldUnknown)
    np.testing.assert_allclose(np.asarray(radial.unknown.flux_basis), [0] * 5 + [1])
    operator.prescribed_current_field = SimpleNamespace(circuit_count=2)
    coil = VerticalPositionPin(
        0.7, coil_direction=np.asarray([0.0, 1.0]), source_scale=100.0
    ).pair(operator)
    assert isinstance(coil.unknown, CircuitCurrentUnknown)
    np.testing.assert_array_equal(np.asarray(coil.unknown.direction[:, 0]), [0, 1])


@pytest.mark.slow
def test_elongated_free_public_request_drifts():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    receipt = profile.solve(_request(profile, TARGETS[0], active=False))
    flux = receipt.equilibrium.flux
    assert bool(receipt.qualified)
    assert float(receipt.equilibrium.fixed_point.residual) <= 1e-12
    assert _terminal_elongation(flux) >= 1.6
    assert abs(_booked_terminal_height(flux) - FREE_HEIGHT) <= 1e-10
    assert abs(_booked_terminal_height(flux) - TARGETS[0]) >= 0.1
    np.testing.assert_allclose(receipt.constraints[0].physical_unknown, [0.0])


@pytest.mark.slow
@pytest.mark.parametrize("target", TARGETS)
def test_public_pinned_request_converges_to_booked_height(target):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    receipt = profile.solve(_request(profile, target))
    flux = receipt.equilibrium.flux
    assert bool(receipt.qualified)
    assert float(receipt.equilibrium.fixed_point.residual) <= 1e-12
    assert _terminal_elongation(flux) >= 1.6
    assert abs(_booked_terminal_height(flux) - target) <= 1e-10
    assert len(receipt.constraints) == 1
    assert bool(receipt.constraints[0].qualified[0])
