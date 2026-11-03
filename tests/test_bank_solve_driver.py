"""The staged bank driver skips the Jacobian when its batch reuses a factor."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402


class _Reading(NamedTuple):
    membership: jax.Array
    normal_form_cells: jax.Array
    fragment_selected: jax.Array


class _Stationary(NamedTuple):
    position: jax.Array


class _Support(NamedTuple):
    stationary: _Stationary
    reading: _Reading
    valid: jax.Array
    cells: jax.Array
    saddle_live: jax.Array
    rebuilds: jax.Array
    rebuilt: jax.Array


def _toy_read(_operands, _cache, states, seeds, previous, *_args, **_kwargs):
    lanes = states[0].shape[0]
    return _Support(
        _Stationary(previous),
        _Reading(
            jnp.full((lanes, 1), 0.5),
            jnp.zeros((lanes, 1), bool),
            jnp.ones((lanes, 1, 1), bool),
        ),
        jnp.ones(lanes, bool),
        jnp.zeros((lanes, 1), jnp.int32),
        jnp.zeros(lanes, bool),
        jnp.zeros(lanes, jnp.int32),
        jnp.zeros(lanes, bool),
    )


def _toy_residual(_operands, base, state, support, _profile, _quadrature):
    return support_read.StepResidual(
        state,
        (*state, jnp.asarray(1.0)),
        support.reading,
        support.valid,
    )


def test_batch_reuses_factor_without_rebuilding_jacobians(monkeypatch):
    """A batch-level branch skips every Jacobian on contracting steps."""
    builds = []

    def jacobian(_operands, moments, *_args, **_kwargs):
        jax.debug.callback(lambda value: builds.append(int(value)), moments[0][0])
        return 2 * jnp.eye(3, dtype=moments[0].dtype)

    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "step_jacobian", jacobian)
    states = tuple(jnp.asarray([[1.0], [2.0], [3.0]]) for _ in range(3))
    seeds = jnp.zeros((3, 1, 2))
    run = jax.jit(
        lambda values: support_read.solve_batch(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=8,
            tolerance=0.1,
        )
    )
    result = run(states)
    jax.block_until_ready(result)
    assert bool(jnp.all(result.converged))
    assert np.array_equal(np.asarray(result.refactors), [1, 1, 1])
    assert len(builds) == 3, f"Jacobian rebuilt {len(builds)} times for three lanes"
    assert np.array_equal(np.asarray(result.reads), np.asarray(result.steps) + 1)
    assert len(set(np.asarray(result.steps))) > 1


def test_step_cap_reports_nonconvergence(monkeypatch):
    monkeypatch.setattr(support_read, "read_batch", _toy_read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(
        support_read,
        "step_jacobian",
        lambda _operands, moments, *_args, **_kwargs: (
            2 * jnp.eye(3, dtype=moments[0].dtype)
        ),
    )
    state = tuple(jnp.asarray([1.0]) for _ in range(3))
    seeds = jnp.zeros((1, 2))
    result = jax.jit(
        lambda values: support_read.solve(
            None,
            None,
            values,
            seeds,
            seeds,
            None,
            None,
            None,
            None,
            None,
            max_steps=1,
            tolerance=1e-10,
        )
    )(state)
    jax.block_until_ready(result)
    assert not bool(result.converged)
    assert int(result.steps) == 1
    assert float(result.residual) > 1e-10
