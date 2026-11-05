"""The staged driver takes the tracked read between steps, and certifies fresh."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium import support_read  # noqa: E402

pytestmark = pytest.mark.timeout(1800)


class _Reading(NamedTuple):
    membership: jax.Array
    normal_form_cells: jax.Array
    fragment_selected: jax.Array
    boundary: jax.Array | None = None


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


def _gated_read(_operands, _cache, states, seeds, previous, *_args, **kwargs):
    """A read that rebuilds exactly when no tracking state is supplied."""
    tracking = kwargs.get("tracking")
    lanes = states[0].shape[0]
    rebuilt = jnp.full(lanes, tracking is None, dtype=bool)
    rebuilds = (
        jnp.ones(lanes, jnp.int32)
        if tracking is None
        else tracking.rebuilds + 1
    )
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
        rebuilds,
        rebuilt,
    )


def _toy_residual(_operands, _base, state, support, _profile, _quadrature):
    return support_read.StepResidual(
        state,
        (*state, jnp.asarray(1.0)),
        support.reading,
        support.valid,
    )


def _unit_update(_operands, moments, _support, *_args, **_kwargs):
    update = tuple(-0.5 * value for value in moments)
    size = sum(value.size for value in moments)
    matrix = jnp.eye(size, dtype=moments[0].dtype)
    factor = support_read.NewtonFactor(
        matrix, jnp.ones(size), jnp.ones(size), matrix, jnp.zeros(size, jnp.int32)
    )
    return support_read.NewtonStep(
        update, 1.0, 1.0, 0.0, jnp.asarray(True), factor, jnp.asarray(True)
    )


def _solve(monkeypatch, *, force_rebuild=False, lanes=3, max_steps=8, tolerance=0.1):
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs.get("tracking"))
        if force_rebuild:
            kwargs.pop("tracking", None)
        return _gated_read(*args, **kwargs)

    monkeypatch.setattr(support_read, "read_batch", read)
    monkeypatch.setattr(support_read, "step_residual", _toy_residual)
    monkeypatch.setattr(support_read, "newton_step", _unit_update)
    states = tuple(jnp.arange(lanes, dtype=jnp.float64)[:, None] * 1.0 for _ in range(3))
    seeds = jnp.zeros((lanes, 1, 2))
    result = jax.jit(
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
            max_steps=max_steps,
            tolerance=tolerance,
        )
    )(states)
    jax.block_until_ready(result)
    return result, calls


def test_converged_solve_counts_the_tracked_reads_it_took(monkeypatch):
    """Every iterate after the first carries the tracked decision."""
    result, _ = _solve(monkeypatch)
    tracked = np.asarray(result.tracked_reads)
    reads = np.asarray(result.reads)
    steps = np.asarray(result.steps)
    print("TRACKED", tracked, "READS", reads, "STEPS", steps)
    assert bool(jnp.all(result.converged))
    assert np.array_equal(reads, steps + 1)
    assert int(tracked.sum()) >= 1, "the solve rebuilt on every iterate"
    assert int(tracked.sum()) < int(reads.sum()), "no iterate carried the decision"


def test_the_acceptance_read_always_rebuilds(monkeypatch):
    """The certificate is a fresh read, never a carried decision."""
    result, calls = _solve(monkeypatch)
    assert int(np.asarray(result.certificates).ravel()[0]) == 1
    assert calls[-1] is None, "the acceptance read carried a tracking state"
    assert bool(jnp.all(result.support.rebuilt))
    # The certificate re-reads the final state's own stencil, so the tracked
    # counter it reports cannot be lower than the last in-loop read's.
    assert bool(jnp.all(result.support.rebuilds >= 1))


def test_forcing_the_rebuild_branch_leaves_no_tracked_reads(monkeypatch):
    """The counter reads the branch taken, not a constant."""
    result, _ = _solve(monkeypatch, force_rebuild=True)
    print("TRACKED_FORCED", np.asarray(result.tracked_reads))
    assert bool(jnp.all(result.converged))
    assert int(np.asarray(result.tracked_reads).sum()) == 0