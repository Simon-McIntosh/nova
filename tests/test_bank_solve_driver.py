"""The staged bank driver skips the Jacobian when its batch reuses a factor."""

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


def test_first_batched_read_has_one_rebuild_flag_per_lane(monkeypatch):
    """The first read must match the shape of subsequent tracked reads."""
    monkeypatch.setattr(
        support_read,
        "stationary_stencil",
        lambda _bank, _cache, _state, _seed, warm, *_args, **_kwargs: (
            support_read.Stencil(
                _Stationary(warm), jnp.zeros((1,), jnp.int32), jnp.asarray(False)
            )
        ),
    )
    monkeypatch.setattr(
        support_read,
        "_rebuilt_reading",
        lambda *_args: _Reading(
            jnp.asarray([0.5]), jnp.asarray([False]), jnp.asarray([[True]])
        ),
    )
    monkeypatch.setattr(
        support_read,
        "_verdict",
        lambda stencil, reading, rebuilt, rebuilds: _Support(
            stencil.solved,
            reading,
            jnp.asarray(True),
            stencil.cells,
            stencil.saddle_live,
            rebuilds,
            rebuilt,
        ),
    )
    states = tuple(jnp.ones((3, 1)) for _ in range(3))
    seeds = jnp.zeros((3, 1, 2))
    first = support_read.read_batch(None, None, states, seeds, seeds, None, None)
    assert first.rebuilt.shape == (3,)
    assert first.rebuilds.shape == (3,)


def test_batch_reuses_factor_without_rebuilding_jacobians(monkeypatch):
    """A batch-level branch skips every Jacobian on contracting steps."""
    builds = []

    def jacobian(_operands, moments, *_args, **_kwargs):
        jax.debug.callback(lambda value: builds.append(int(value)), moments[0][0])
        return 1.5 * jnp.eye(3, dtype=moments[0].dtype)

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


@pytest.fixture(scope="module")
def diverted_case():
    from scripts.prototypes.bank_newton_step.measure import machine

    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    return bank, cache, roots, seeds, convention, policy, profile, quadrature, moments


def _device_solve(case, *, reuse_factor, freeze_active_set):
    bank, cache, roots, seeds, convention, policy, profile, quadrature, moments = case

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    run = jax.jit(
        lambda state: support_read.solve(
            bank,
            cache,
            state,
            seeds,
            roots,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=reuse_factor,
            freeze_active_set=freeze_active_set,
        )
    )
    result = run(moments)
    jax.block_until_ready(result)
    return result


@pytest.fixture(scope="module")
def fresh_solution(diverted_case):
    return _device_solve(diverted_case, reuse_factor=False, freeze_active_set=False)


@pytest.fixture(scope="module")
def reused_solution(diverted_case, fresh_solution):
    assert bool(fresh_solution.converged)
    return _device_solve(diverted_case, reuse_factor=True, freeze_active_set=False)


@pytest.fixture(scope="module")
def frozen_solution(diverted_case, reused_solution):
    assert bool(reused_solution.converged)
    return _device_solve(diverted_case, reuse_factor=True, freeze_active_set=True)


@pytest.mark.slow
def test_fresh_factor_trajectory_matches_prototype(diverted_case, fresh_solution):
    bank, cache, roots, seeds, convention, policy, profile, quadrature, moments = (
        diverted_case
    )

    @jax.jit
    def read(state):
        return support_read.read(bank, cache, state, seeds, roots, convention, policy)

    @jax.jit
    def step(state, support):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )

    @jax.jit
    def norm(state, support):
        residual = support_read.step_residual(
            bank, state, state, support, profile, quadrature
        )
        return jnp.max(jnp.abs(jnp.concatenate(residual.residual)))

    staged = fresh_solution
    assert bool(staged.converged)
    expected_residuals = []
    expected_updates = []
    state = moments
    for index in range(60):
        support = read(state)
        fresh_norm = float(norm(state, support))
        expected_residuals.append(fresh_norm)
        if fresh_norm <= 1e-10:
            break
        correction = step(state, support)
        update = float(jnp.max(jnp.abs(jnp.concatenate(correction.update))))
        expected_updates.append(update)
        print(
            f"TRAJECTORY step={index} residual={fresh_norm:.12e} update={update:.12e}"
        )
        state = tuple(
            value + change
            for value, change in zip(state, correction.update, strict=True)
        )
    assert len(expected_updates) == int(staged.steps)
    np.testing.assert_allclose(
        np.asarray(staged.residual_path[: len(expected_residuals)]),
        expected_residuals,
        rtol=1e-9,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray(staged.update_path[: len(expected_updates)]),
        expected_updates,
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.slow
def test_factor_reuse_trajectory_converges(fresh_solution, reused_solution):
    fresh, reused = fresh_solution, reused_solution
    print("FRESH", np.asarray(fresh.residual_path[: int(fresh.steps) + 1]))
    print("REUSED", np.asarray(reused.residual_path[: int(reused.steps) + 1]))
    assert bool(fresh.converged) and bool(reused.converged)
    assert int(reused.steps) <= int(fresh.steps) + 2
    assert int(reused.refactors) < int(reused.steps)


@pytest.mark.slow
def test_active_set_freeze_trajectory_converges(reused_solution, frozen_solution):
    reused, frozen = reused_solution, frozen_solution
    print("UNFROZEN", np.asarray(reused.residual_path[: int(reused.steps) + 1]))
    print("FROZEN", np.asarray(frozen.residual_path[: int(frozen.steps) + 1]))
    assert bool(reused.converged) and bool(frozen.converged)
    assert int(frozen.freezes) >= 1
    assert float(frozen.residual) <= 1e-10
