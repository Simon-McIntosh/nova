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
    ConstraintContext,
    CurrentCentroidConstraint,
    UniformRadialFieldUnknown,
    VerticalPositionPin,
    assemble_augmented_system,
    uniform_radial_field_jet,
)
from nova.equilibrium import fixed_point
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.observation import CurrentMomentObservation, MomentIntegralSupport
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes


ELONGATION = 1.8
FREE_HEIGHT = 0.6
SEED = np.asarray([1.0, 0.8, 0.30, 1.0, 1.2, 0.90], dtype=np.float64)
TARGETS = (0.7, 0.9)


def _carrier_residual(flux):
    first, first_r, first_z, second, second_r, second_z = flux
    low, high = first_z / first, second_z / second
    radial_span = second_r / second - first_r / first
    height = CurrentCentroidConstraint.booked_height(flux.reshape((2, 3)))
    return jnp.stack(
        (
            first - 1.0,
            second - 1.0,
            first_r - 0.8 * first,
            second_r - 1.2 * second,
            (high - low) / radial_span - ELONGATION,
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


def _request(
    profile,
    target,
    *,
    active=True,
    coil_direction=None,
    source_scale=1.0,
    compilation_cache=False,
):
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
            "compilation_cache": compilation_cache,
        },
        vertical_pin=VerticalPositionPin(
            target,
            active=active,
            coil_direction=coil_direction,
            source_scale=source_scale,
        ),
    )


def _booked_terminal_height(flux):
    moments = np.asarray(flux).reshape((2, 3))
    return float(np.sum(moments[:, 2]) / np.sum(moments[:, 0]))


def _terminal_elongation(flux):
    moments = np.asarray(flux).reshape((2, 3))
    radial_span = float(np.ptp(moments[:, 1] / moments[:, 0]))
    vertical_span = float(np.ptp(moments[:, 2] / moments[:, 0]))
    assert radial_span >= 0.39
    return vertical_span / radial_span


def test_vertical_pin_declares_a_physical_field_source():
    configure_dtypes()
    operator = _CarrierOperator()
    radial = VerticalPositionPin(0.7, source_scale=1e-3).pair(operator)
    assert isinstance(radial.unknown, UniformRadialFieldUnknown)
    np.testing.assert_allclose(np.asarray(radial.unknown.flux_basis), [0] * 5 + [1])
    np.testing.assert_allclose(
        np.asarray(uniform_radial_field_jet(jnp.asarray([[1.5, -0.5]])))[0],
        [1.5 * np.pi, np.pi, -3.0 * np.pi, 0.0, -2.0 * np.pi, 0.0],
    )
    operator.prescribed_current_field = PrescribedCurrentField(
        response=jnp.asarray([[0.0, 0.0]] * 5 + [[0.0, 1.0]]),
        current=jnp.zeros(2),
    )
    coil = VerticalPositionPin(
        0.7, coil_direction=np.asarray([0.0, 1.0]), source_scale=100.0
    ).pair(operator)
    assert isinstance(coil.unknown, CircuitCurrentUnknown)
    np.testing.assert_array_equal(np.asarray(coil.unknown.direction[:, 0]), [0, 1])
    delta = coil.unknown.flux_delta(
        SimpleNamespace(operator=operator),
        ConstraintContext(jnp.zeros(6), None, None, None),
        coil.functional,
        coil.binding.payload,
        jnp.asarray([0.25]),
    )
    np.testing.assert_allclose(delta, [0.0] * 5 + [25.0])
    profile = _profile()
    profile.operator.prescribed_current_field = operator.prescribed_current_field
    receipt = profile.solve(
        _request(
            profile, 0.7, coil_direction=np.asarray([0.0, 1.0]), source_scale=100.0
        )
    )
    assert bool(receipt.qualified)
    assert float(receipt.equilibrium.fixed_point.residual) <= 1e-12
    assert abs(_booked_terminal_height(receipt.equilibrium.flux) - 0.7) <= 1e-10


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
    seed_moments = SEED.reshape((2, 3))
    expected_free = float(seed_moments[:, 2].sum() / seed_moments[:, 0].sum())
    assert abs(_booked_terminal_height(flux) - expected_free) <= 1e-10
    assert abs(_booked_terminal_height(flux) - TARGETS[0]) >= 0.09
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


@pytest.mark.slow
def test_public_request_passes_pin_operands_to_one_cache_identity(monkeypatch):
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    profile._configure_solve_compilation_cache = lambda enabled: None
    original = fixed_point.newton_krylov
    seen = []

    def inspect_runtime_pair(*args, **kwargs):
        runtime_pairs = kwargs.get("map_arguments", ())
        assert len(runtime_pairs) == 1 and len(runtime_pairs[0]) == 1
        pair = runtime_pairs[0][0]
        seen.append(
            (
                float(np.asarray(pair.binding.target)[0]),
                float(np.asarray(pair.unknown.field_scale)),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(fixed_point, "newton_krylov", inspect_runtime_pair)
    first = profile.solve(
        _request(profile, TARGETS[0], compilation_cache=True, source_scale=1.0)
    )
    second = profile.solve(
        _request(profile, TARGETS[1], compilation_cache=True, source_scale=2.0)
    )
    assert seen == [(TARGETS[0], 1.0), (TARGETS[1], 2.0)]
    assert len(profile._request_compilation_cache) == 1
    assert not first.compilation_cache_hit
    assert second.compilation_cache_hit
    for receipt, target in ((first, TARGETS[0]), (second, TARGETS[1])):
        assert bool(receipt.qualified)
        assert float(receipt.equilibrium.fixed_point.residual) <= 1e-12
        assert abs(_booked_terminal_height(receipt.equilibrium.flux) - target) <= 1e-10


@pytest.mark.slow
def test_one_augmented_newton_program_serves_free_and_pinned_requests():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    operator = profile.operator

    def production(seed, pair):
        system = assemble_augmented_system(
            profile,
            seed,
            (pair,),
            base_map=profile.flux_map(),
            base_shadow_mask=lambda flux: operator.residual_shadow_mask(flux),
            base_promoted_shadow_mask=lambda flux, previous: (
                operator.residual_shadow_mask(flux, previous_shadow=previous)
            ),
            base_shadowed_map=operator.flux_map_with_shadow(),
            requested_class=None,
            target_current=None,
        )
        history = fixed_point.newton_krylov(
            system.map_fn,
            system.initial,
            newton_steps=8,
            gmres_iterations=7,
            warmup=0,
            active_set_steps=2,
            convergence_tolerance=1e-12,
            shadow_mask_fn=system.shadow_mask_fn,
            promoted_shadow_mask_fn=system.promoted_shadow_mask_fn,
            shadowed_map_fn=system.shadowed_map_fn,
            row_jvp_observers=system.row_jvp_observers,
        )
        flux, unknown = system.split(history.state)
        return flux, unknown, history.residual, history.converged

    seed = jnp.asarray(SEED)
    free = _request(profile, TARGETS[0], active=False).vertical_pin.pair(operator)
    program = jax.jit(production)
    compiled = program.lower(seed, free).compile()
    for target, active, expected in [
        (TARGETS[0], False, float(SEED[2] + SEED[5]) / float(SEED[0] + SEED[3])),
        *[(height, True, height) for height in TARGETS],
    ]:
        pair = _request(profile, target, active=active).vertical_pin.pair(operator)
        flux, unknown, residual, converged = compiled(seed, pair)
        assert bool(converged)
        assert float(residual) <= 1e-12
        assert _terminal_elongation(flux) >= 1.6
        assert abs(_booked_terminal_height(flux) - expected) <= 1e-10
        if not active:
            np.testing.assert_array_equal(unknown, [0.0])
