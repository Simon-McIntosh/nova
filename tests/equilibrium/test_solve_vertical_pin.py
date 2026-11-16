"""Vertical current-centroid pin on the public augmented forward solve."""

from __future__ import annotations

from dataclasses import replace
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
from nova.equilibrium import fixed_point, reduced_newton
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.equilibrium.observation import CurrentMomentObservation, MomentIntegralSupport
from nova.equilibrium.solve_request import ExplicitSolveSeed, ForwardSolveRequest
from nova.jax.config import configure_dtypes


ELONGATION = 1.8
VERTICAL_FORCING = 0.1
SEED = np.asarray([1.0, 0.8, 0.24, 1.0, 1.2, 0.96], dtype=np.float64)
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
            jnp.ones_like(height) * VERTICAL_FORCING / ELONGATION**2,
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
    include_pin=True,
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
        vertical_pin=(
            VerticalPositionPin(
                target,
                active=active,
                coil_direction=coil_direction,
                source_scale=source_scale,
            )
            if include_pin
            else None
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
def test_elongated_free_public_request_stalls():
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    receipt = profile.solve(_request(profile, TARGETS[0], include_pin=False))
    flux = receipt.equilibrium.flux
    assert not bool(receipt.qualified)
    assert float(receipt.equilibrium.fixed_point.residual) >= 1e-3
    assert _terminal_elongation(flux) >= 1.6
    assert not bool(receipt.equilibrium.fixed_point.converged)
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
    substituted_target = None

    def inspect_runtime_pair(*args, **kwargs):
        nonlocal substituted_target
        runtime_pairs = kwargs.get("map_arguments", ())
        assert len(runtime_pairs) == 1 and len(runtime_pairs[0]) == 1
        pair = runtime_pairs[0][0]
        seen.append(
            (
                float(np.asarray(pair.binding.target)[0]),
                float(np.asarray(pair.unknown.field_scale)),
            )
        )
        if substituted_target is not None:
            pair = replace(
                pair,
                binding=replace(pair.binding, target=jnp.asarray([substituted_target])),
            )
            kwargs["map_arguments"] = ((pair,),)
        return original(*args, **kwargs)

    monkeypatch.setattr(fixed_point, "newton_krylov", inspect_runtime_pair)
    free = profile.solve(
        _request(profile, TARGETS[0], include_pin=False, compilation_cache=True)
    )
    first = profile.solve(
        _request(profile, TARGETS[0], compilation_cache=True, source_scale=1.0)
    )
    second = profile.solve(
        _request(profile, TARGETS[1], compilation_cache=True, source_scale=2.0)
    )
    assert seen == [(0.0, 1.0), (TARGETS[0], 1.0), (TARGETS[1], 2.0)]
    assert len(profile._request_compilation_cache) == 1
    assert not free.compilation_cache_hit
    assert first.compilation_cache_hit
    assert second.compilation_cache_hit
    assert (
        sum(not receipt.compilation_cache_hit for receipt in (free, first, second)) == 1
    )
    assert not bool(free.qualified)
    assert float(free.equilibrium.fixed_point.residual) >= 1e-3
    for receipt, target in ((first, TARGETS[0]), (second, TARGETS[1])):
        assert bool(receipt.qualified)
        assert float(receipt.equilibrium.fixed_point.residual) <= 1e-12
        assert abs(_booked_terminal_height(receipt.equilibrium.flux) - target) <= 1e-10
    substituted_target = 0.8
    shifted = profile.solve(_request(profile, TARGETS[0]))
    assert abs(_booked_terminal_height(shifted.equilibrium.flux) - 0.8) <= 1e-10
    assert not bool(shifted.qualified)


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
        if active:
            assert bool(converged)
            assert float(residual) <= 1e-12
        else:
            assert not bool(converged)
            assert float(residual) >= 1e-3
        assert _terminal_elongation(flux) >= 1.6
        assert abs(_booked_terminal_height(flux) - expected) <= 1e-10
        if not active:
            np.testing.assert_array_equal(unknown, [0.0])


@pytest.mark.slow
@pytest.mark.parametrize("route", ["newton", "rebuilt", "continuation", "descent"])
@pytest.mark.parametrize("source", ["radial", "circuit"])
@pytest.mark.parametrize("shadowed", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_compensator_image_and_tangent_on_every_proposal_route(
    route, source, shadowed, active, monkeypatch
):
    """Observe every scored proposal, including off-root inactive unknowns."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    profile.operator.grid.coordinate = profile.operator.grid.coordinate.at[4, 1].set(
        -1.0 / (4 * jnp.pi)
    )
    direction = None
    if source == "circuit":
        profile.operator.prescribed_current_field = PrescribedCurrentField(
            response=jnp.asarray([[0.0, 0.0]] * 4 + [[0.0, 0.5], [0.0, 1.0]]),
            current=jnp.zeros(2),
        )
        direction = np.asarray([0.0, 1.0])
    pair = VerticalPositionPin(
        0.7, active=active, coil_direction=direction, initial_unknown=0.25
    ).pair(profile.operator)
    target = jnp.asarray([1.0, 0.8, 0.3, 1.0, 1.4, 0.9])

    def base_map(flux):
        return target

    system = assemble_augmented_system(
        profile,
        jnp.asarray(SEED),
        (pair,),
        base_map=base_map,
        base_shadow_mask=lambda flux: jnp.zeros_like(flux, dtype=bool),
        base_promoted_shadow_mask=lambda flux, previous: previous,
        base_shadowed_map=lambda flux, shadow: jnp.where(shadow, flux, target),
        requested_class=None,
        target_current=None,
    )
    images, tangents, frozen_changes = [], [], []
    original_image = type(pair.unknown).flux_delta

    def observed_image(*args):
        delta = original_image(*args)
        jax.debug.callback(lambda image: images.append(np.asarray(image)), delta)
        return delta

    monkeypatch.setattr(type(pair.unknown), "flux_delta", observed_image)

    @jax.jit
    def propose(runtime_pair):
        def raw_map(state):
            if shadowed:
                return system.shadowed_map_fn(
                    state,
                    jnp.zeros_like(state, dtype=bool).at[4].set(True),
                    (runtime_pair,),
                )
            return system.map_fn(state, (runtime_pair,))

        def observed_map(state):
            mapped = raw_map(state)
            jax.debug.callback(
                lambda value: frozen_changes.append(np.asarray(value)),
                mapped[4] - state[4],
            )
            unknown_direction = jnp.zeros_like(state).at[-1].set(1.0)
            tangent = jax.jvp(raw_map, (state,), (unknown_direction,))[1]
            jax.debug.callback(
                lambda value: tangents.append(np.asarray(value)),
                tangent[:6],
            )
            return mapped

        state = system.initial
        mapped, tangent = jax.linearize(observed_map, state)
        residual = fixed_point._relative_residual(mapped, state)
        merit = fixed_point._smooth_relative_sup_merit(mapped, state)
        if route == "rebuilt":
            return fixed_point._rebuilt_model_promotion(
                observed_map,
                state,
                residual,
                merit,
                gmres_iterations=7,
                maximum_step=jnp.asarray(1.0),
                initial_damping=jnp.asarray(1e-3),
                model_trust_selection=False,
            )
        if route == "descent":
            return fixed_point._steepest_descent_promotion(
                observed_map, state, residual, merit, False
            )
        jacobian = jax.vmap(tangent)(jnp.eye(state.size)).T
        # Frozen flux coordinates have zero residual and a null Newton row.
        step = jnp.linalg.lstsq(
            jnp.eye(state.size) - jacobian, mapped - state, rcond=None
        )[0]

        def model(candidate):
            return mapped + tangent(candidate - state)

        return fixed_point._backtracked_promotion(
            observed_map,
            model,
            state,
            step if route == "newton" else -step,
            step,
            residual,
            merit,
            jnp.asarray(0.1),
            False,
        )

    proposal = propose(pair)
    jax.block_until_ready(proposal)
    jax.effects_barrier()
    assert bool(proposal.accepted), route
    assert not np.array_equal(proposal.state, system.initial), route
    if route in ("newton", "continuation"):
        assert bool(proposal.recovery_activated) == (route == "continuation")
    assert len(images) > 1 and len(tangents) > 1, "candidate evaluations must run"
    images, tangents = np.stack(images), np.stack(tangents)
    if shadowed:
        np.testing.assert_allclose(frozen_changes, 0.0, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(tangents[:, 4], 0.0)
    else:
        assert np.any(np.abs(frozen_changes) > 1e-3), "live coordinate must move"
    if active:
        assert np.any(images != 0), "active source is the known-present control"
        assert np.any(tangents != 0), "active tangent is the known-present control"
    else:
        np.testing.assert_array_equal(images, np.zeros_like(images))
        np.testing.assert_array_equal(tangents, np.zeros_like(tangents))


@pytest.mark.parametrize("active", [False, True])
def test_reduced_compensator_image_and_tangent(active):
    """The reduced reconstruction obeys the runtime binding off the root."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile = _profile()
    profile.operator.prescribed_current_field = PrescribedCurrentField(
        response=jnp.asarray([[0.0, 0.0]] * 5 + [[0.0, 1.0]]),
        current=jnp.zeros(2),
    )
    pair = VerticalPositionPin(
        0.7, active=active, coil_direction=np.asarray([0.0, 1.0])
    ).pair(profile.operator)
    augmentation = reduced_newton._RowAugmentation(
        profile, (pair,), (slice(0, 1),), 1, jnp.asarray(1.0), None, None
    )

    @jax.jit
    def evaluate(runtime_pair, unknown):
        image, tangent = jax.jvp(
            lambda value: augmentation.flux_delta(value, (runtime_pair,)),
            (unknown,),
            (jnp.ones_like(unknown),),
        )
        row = augmentation.rows(jnp.asarray(SEED), unknown, None, (runtime_pair,), None)
        return image, tangent, row

    # Both bindings go through the same compiled program and nonzero unknown.
    unknown = jnp.asarray([0.25])
    image, tangent, row = evaluate(pair, unknown)
    control = replace(pair, binding=replace(pair.binding, payload=jnp.asarray(True)))
    control_image, control_tangent, _ = evaluate(control, unknown)
    assert np.any(np.asarray(control_image) != 0)
    assert np.any(np.asarray(control_tangent) != 0)
    if active:
        np.testing.assert_array_equal(image, control_image)
        np.testing.assert_array_equal(tangent, control_tangent)
    else:
        np.testing.assert_array_equal(image, np.zeros_like(image))
        np.testing.assert_array_equal(tangent, np.zeros_like(tangent))
        np.testing.assert_array_equal(row, unknown)


def test_compensator_gate_delegates_to_shared_arm_runner(monkeypatch, tmp_path):
    """One owner executes CPU arms and decides whether their receipts completed."""
    from scripts.prototypes.diiid_core_hold.regressions import gate as shared_gate
    from scripts.prototypes.diiid_inactive_compensator import gate as gates

    assert getattr(gates, "core_gate", None) is shared_gate
    calls = []

    class RunnerReached(Exception):
        pass

    def observe(*args):
        calls.append(args)
        raise RunnerReached

    monkeypatch.setattr(gates, "core_gate", observe)
    with pytest.raises(RunnerReached):
        gates.arm(tmp_path, "revision", tmp_path, "arm", gates.PIN)
    assert len(calls) == 1
    assert calls[0][:3] == (tmp_path, "revision", gates.PIN)
