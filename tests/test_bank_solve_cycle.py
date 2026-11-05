"""A fresh residual line search escapes a continuous piecewise-linear cycle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

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
        eye = jnp.eye(size, dtype=jnp.float64)
        factor = support_read.NewtonFactor(
            eye,
            jnp.ones(size, dtype=jnp.float64),
            jnp.ones(size, dtype=jnp.float64),
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
    states = (
        jnp.array([[2.0], [-2.0], [0.0]], dtype=jnp.float64),
        jnp.zeros((3, 1), dtype=jnp.float64),
        jnp.zeros((3, 1), dtype=jnp.float64),
    )
    points = jnp.zeros((3, 1, 2), dtype=jnp.float64)
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


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_diverted_lanes_leave_near_root_cycle():
    """Restart the certificate lanes where fresh residuals first permit freezing."""
    from scripts.prototypes.bank_solve_read import measure as reference

    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        reference.machine("diverted")
    )
    lanes = 16
    scales = jnp.linspace(2e-5, 6e-5, lanes)
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    seeds = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
    previous = jnp.broadcast_to(roots, (lanes, *roots.shape))

    def run(state, operands, limit):
        return support_read.solve_batch(
            operands,
            cache,
            state,
            seeds,
            previous,
            convention,
            policy,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            max_steps=limit,
        )

    checkpoint = jax.jit(run, static_argnums=2)(states, bank, 2)
    assert float(jnp.max(checkpoint.residual)) < 1e-6
    assert not bool(jnp.all(checkpoint.converged))
    result = jax.jit(run, static_argnums=2)(checkpoint.moments, bank, 8)
    jax.block_until_ready(result)
    print("DIVERTED_CYCLE", np.asarray(result.steps), np.asarray(result.residual))
    assert bool(jnp.all(result.converged)), "cycle lanes exceed eight restart steps"
    assert float(jnp.max(result.residual)) <= 1e-10


def test_unchanged_decision_unlocks_when_residual_stalls(monkeypatch):
    """A constant residual must release a lock even with identical selections."""
    from tests.test_bank_solve_driver import _toy_residual, _unit_update

    def stationary_step(*args, **kwargs):
        correction = _unit_update(*args, **kwargs)
        return correction._replace(
            update=tuple(jnp.zeros_like(v) for v in correction.update)
        )

    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "newton_step", stationary_step)
    states = tuple(jnp.full((1, 1), 1e-7, dtype=jnp.float64) for _ in range(3))
    points = jnp.zeros((1, 1, 2), dtype=jnp.float64)
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
            max_steps=4,
        )
    )(states)
    jax.block_until_ready(result)
    print("STALL_RELEASE", np.asarray(result.unfreezes), np.asarray(result.freezes))
    assert not bool(result.converged[0])
    assert int(result.unfreezes[0]) == 2
    assert int(result.freezes[0]) == 2
