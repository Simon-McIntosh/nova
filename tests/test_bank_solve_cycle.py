"""A fresh residual line search escapes a continuous piecewise-linear cycle."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402
from tests.test_bank_solve_driver import _toy_read  # noqa: E402


def test_cycle_lane_is_accepted_within_two_steps(monkeypatch):
    """Newton flips between ±2 outside a kink; a half update reaches zero."""
    scale = 1e-7

    def residual(_operands, _base, state, support, *_args):
        value = state[0]
        equation = scale * (value + 2 * jnp.clip(value, -1.0, 1.0))
        return support_read.StepResidual(
            (equation, state[1], state[2]),
            (*state, jnp.asarray(1.0)),
            support.reading,
            support.valid,
        )

    def step(_operands, state, _support, *_args, **_kwargs):
        value = state[0]
        equation = value + 2 * jnp.clip(value, -1.0, 1.0)
        slope = jnp.where(jnp.abs(value) < 1.0, 3.0, 1.0)
        update = (-equation / slope, jnp.zeros_like(value), jnp.zeros_like(value))
        size = sum(item.size for item in state)
        eye = jnp.eye(size)
        factor = support_read.NewtonFactor(
            eye,
            jnp.ones(size),
            jnp.ones(size),
            eye,
            jnp.zeros(size, jnp.int32),
        )
        return support_read.NewtonStep(
            update,
            1.0,
            1.0,
            0.0,
            jnp.asarray(True),
            factor,
            jnp.asarray(True),
        )

    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", residual)
    monkeypatch.setattr(support_read, "newton_step", step)
    states = (jnp.array([[2.0], [-2.0], [0.0]]), jnp.zeros((3, 1)), jnp.zeros((3, 1)))
    points = jnp.zeros((3, 1, 2))
    result = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            points,
            points,
            None,
            None,
            None,
            None,
            None,
            max_steps=6,
        )
    )(states)
    jax.block_until_ready(result)
    print(
        "CYCLE",
        np.asarray(result.steps),
        np.asarray(result.residual),
        np.asarray(result.residual_path),
    )
    assert int(jnp.max(result.steps)) <= 2, "Newton remained in the period-two cycle"
    assert bool(jnp.all(result.converged))
    assert float(jnp.max(result.residual)) <= 1e-10
    assert int(result.steps[2]) == 0
    np.testing.assert_array_equal(np.asarray(result.moments[0]), 0.0)
