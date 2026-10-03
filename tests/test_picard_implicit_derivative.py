"""Derivative contracts for a converged Picard fixed point."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.picard_derivative_contract import (
    _central_difference_steps,
    _selected_probe,
)
from nova.equilibrium.fixed_point import picard
from nova.jax.config import configure_dtypes


def _solve(control, *, evaluations: int = 80):
    """Converge a scalar contraction while keeping the control explicit."""
    return picard(
        lambda state, value: 0.2 * state + value,
        jnp.zeros(1),
        evaluations=evaluations,
        relaxation=0.7,
        map_arguments=(control,),
        implicit_tolerance=1.0e-12,
    )


def test_picard_uses_the_terminal_fixed_point_tangent():
    """Reverse and JVP match the fixed-point response and a central difference."""
    configure_dtypes()
    control = jnp.asarray([2.0])
    direction = jnp.ones_like(control)

    def response(value):
        return _solve(value).state[0]

    reverse = jax.grad(response)(control)[0]
    _, forward = jax.jvp(response, (control,), (direction,))
    step = 1.0e-4
    central = (
        response(control + step * direction) - response(control - step * direction)
    ) / (2.0 * step)
    implicit = 1.0 / 0.8

    assert bool(_solve(control).converged)
    np.testing.assert_allclose(reverse, implicit, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(forward, implicit, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(reverse, forward, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(central, implicit, rtol=0.0, atol=1.0e-10)


def test_picard_refuses_an_implicit_tangent_before_convergence():
    """A bounded non-fixed-point iterate advertises NaN rather than a tangent."""
    configure_dtypes()
    control = jnp.asarray([2.0])
    direction = jnp.ones_like(control)

    def response(value):
        return _solve(value, evaluations=1).state

    result = _solve(control, evaluations=1)
    _, tangent = jax.jvp(response, (control,), (direction,))

    assert not bool(result.converged)
    assert bool(jnp.all(jnp.isnan(tangent)))


def test_picard_no_selection_control_agrees_to_machine_precision():
    """The smooth explicit-control route retains the same implicit response."""
    configure_dtypes()
    control = jnp.asarray([2.0])

    def response(value):
        return _solve(value).state[0]

    reverse = jax.grad(response)(control)[0]
    _, forward = jax.jvp(response, (control,), (jnp.ones_like(control),))

    np.testing.assert_allclose(reverse, forward, rtol=0.0, atol=1.0e-10)


def test_derivative_witness_selection_preserves_the_default_and_an_explicit_probe():
    """A named physical witness cannot be replaced by the largest response."""
    reverse = jnp.asarray([4.0, -2.0, 1.0, -3.0])
    conductor = jnp.asarray([460.0, 200.0, 90.0, 293.0])
    requested = (293.18627897744824, 87.95588369323447, 29.318627897744825)

    assert _selected_probe(reverse, None) == 0
    assert _selected_probe(reverse, 3) == 3
    np.testing.assert_allclose(
        _central_difference_steps(conductor, 0, None), (0.46, 0.138, 0.046)
    )
    assert _central_difference_steps(conductor, 3, requested) == requested
