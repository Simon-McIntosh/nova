"""An elongated current carrier held by a booked-support centroid row."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from nova.equilibrium import solve
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintContext,
    CurrentCentroidConstraint,
    UniformRadialFieldUnknown,
    VerticalPositionPin,
)
from nova.equilibrium.forward import ForwardProfile
from nova.jax.config import configure_dtypes


ELONGATION = 1.8
RADIAL_HALF_WIDTH = 0.2
FREE_HEIGHT = 0.5
FIELD_LEVERAGE = 1.0
TARGETS = (0.7, 0.9)
SEED = np.asarray([1.0, 1.0, 0.30, 1.0, 1.0, 0.90], dtype=np.float64)


def carrier_map(state, operands):
    """The vertical restoring row weakens as the carrier elongates."""
    elongation, field_basis = operands
    first, first_r, first_z, second, second_r, second_z, field = state
    low, high = first_z / first, second_z / second
    height = CurrentCentroidConstraint.booked_height(state[:-1].reshape((2, 3)))
    return jnp.stack(
        (
            first - 1.0,
            second - 1.0,
            first_r - first,
            second_r - second,
            (high - low) / (2 * RADIAL_HALF_WIDTH) - elongation,
            (height - FREE_HEIGHT) / elongation**2 - field_basis * field,
        )
    )


def centroid_height(state, operands):
    del operands
    return CurrentCentroidConstraint.booked_height(state[:-1].reshape((2, 3)))


def binding(target, active):
    return ConstraintBinding(
        target=[target],
        tolerance=[1e-10],
        scale=[1.0],
        initial_unknown=[0.0],
        payload=jnp.asarray(active),
    )


def operands(leverage=FIELD_LEVERAGE):
    return (jnp.asarray(ELONGATION), jnp.asarray(leverage))


def test_vertical_pin_declares_a_physical_field_source():
    configure_dtypes()
    operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=jnp.asarray([[1.0, 0.25]])),
        wall=SimpleNamespace(coordinate=jnp.asarray([[1.5, -0.5]])),
        sample=None,
        prescribed_current_field=SimpleNamespace(circuit_count=2),
    )
    radial = VerticalPositionPin(0.7, source_scale=1e-3).pair(operator)
    assert isinstance(radial.unknown, UniformRadialFieldUnknown)
    np.testing.assert_allclose(
        np.asarray(radial.unknown.flux_basis),
        -2 * np.pi * np.asarray([0.25, -0.75]),
    )
    coil = VerticalPositionPin(
        0.7, coil_direction=np.asarray([0.0, 1.0]), source_scale=100.0
    ).pair(operator)
    assert isinstance(coil.unknown, CircuitCurrentUnknown)
    np.testing.assert_array_equal(np.asarray(coil.unknown.direction[:, 0]), [0, 1])
    free = VerticalPositionPin(0.7, active=False).pair(operator)
    profile = SimpleNamespace(
        current_moment_observation=lambda *args, **kwargs: SimpleNamespace(
            value=lambda name: jnp.asarray(0.5)
        )
    )
    row = free.functional.residual(
        profile,
        ConstraintContext(jnp.zeros(1), None, None, None),
        jnp.asarray([0.0]),
        free.binding.payload,
        free.binding.target,
        free.binding.scale,
    )
    np.testing.assert_array_equal(np.asarray(row), [0.0])
    held = radial.functional.residual(
        profile,
        ConstraintContext(jnp.zeros(1), None, None, None),
        jnp.asarray([0.0]),
        radial.binding.payload,
        radial.binding.target,
        radial.binding.scale,
    )
    np.testing.assert_allclose(np.asarray(held), [-0.2])


def test_typed_request_passes_vertical_pin_to_converged_route():
    configure_dtypes()
    operator = SimpleNamespace(
        grid=SimpleNamespace(coordinate=jnp.asarray([[1.0, 0.0]])),
        wall=SimpleNamespace(coordinate=jnp.asarray([[1.5, 0.0]])),
        sample=None,
    )
    pin = VerticalPositionPin(0.7)
    captured = {}

    class Reached(RuntimeError):
        pass

    def reached(*args, **kwargs):
        captured.update(kwargs)
        raise Reached

    active = SimpleNamespace(
        operator=operator,
        solve=reached,
        _request_compilation_cache_key=lambda *args: (),
        _configure_solve_compilation_cache=lambda *args: None,
        _request_compilation_cache=set(),
    )
    active._with_source = lambda *args, **kwargs: active
    policy = SimpleNamespace(
        current_pin=False,
        exact_kernels=True,
        cached_machine=True,
        compilation_cache=False,
        kernel_options=lambda: {},
    )
    request = SimpleNamespace(
        source_profile=None,
        clip_mode="chord",
        constraint_pairs=(),
        vertical_pin=pin,
        route="newton_krylov",
        policy=policy,
        target_current=None,
        current=None,
        prescribed_current=None,
        enforce=(),
        constraint_pins=None,
        seed_policy=SimpleNamespace(resolve=lambda *args, **kwargs: SEED),
    )
    with pytest.raises(Reached):
        ForwardProfile._solve_imposed_request(active, request)
    assert len(captured["constraint_pairs"]) == 1
    assert isinstance(
        captured["constraint_pairs"][0].functional, CurrentCentroidConstraint
    )


@pytest.mark.slow
def test_elongated_free_carrier_drifts_from_both_targets():
    configure_dtypes()
    assert ELONGATION >= 1.6
    result = solve.pinned_fixed_point(
        carrier_map, centroid_height, SEED, operands(), binding(TARGETS[0], False)
    )
    assert bool(result.converged)
    height = float(centroid_height(result.state, operands()))
    np.testing.assert_allclose(height, 0.5, atol=1e-10)
    assert all(abs(height - target) > 1e-3 for target in TARGETS)
    assert abs(float(result.state[-1])) <= 1e-12


@pytest.mark.slow
def test_pinned_carrier_holds_two_booked_centroid_targets():
    configure_dtypes()
    fields = []
    for target in TARGETS:
        result = solve.pinned_fixed_point(
            carrier_map, centroid_height, SEED, operands(), binding(target, True)
        )
        assert bool(result.converged)
        assert float(result.residual) <= 1e-12
        observed = float(centroid_height(result.state, operands()))
        assert abs(observed - target) <= 1e-10
        fields.append(float(result.state[-1]))
    assert fields[0] != fields[1]


@pytest.mark.slow
def test_one_program_serves_free_and_two_pinned_carriers():
    configure_dtypes()
    program = jax.jit(
        lambda state, ops, target, active: solve.pinned_fixed_point(
            carrier_map, centroid_height, state, ops, binding(target, active)
        )
    )
    compiled = program.lower(
        SEED, operands(), jnp.asarray(TARGETS[0]), jnp.asarray(False)
    ).compile()
    cases = [(TARGETS[0], False, FREE_HEIGHT)] + [(t, True, t) for t in TARGETS]
    for target, active, expected in cases:
        result = compiled(SEED, operands(), jnp.asarray(target), jnp.asarray(active))
        assert bool(result.converged)
        assert float(result.residual) <= 1e-12
        assert abs(float(centroid_height(result.state, operands())) - expected) <= 1e-10
