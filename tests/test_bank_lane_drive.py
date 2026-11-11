"""Independent drive lanes retain their own accepted state and secant history."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

from nova.equilibrium import drive_continuation as driver  # noqa: E402
from nova.equilibrium import support_read as bank  # noqa: E402
from tests import test_bank_solve_driver as solver_toy  # noqa: E402


class Stationary(NamedTuple):
    position: object


class Support(NamedTuple):
    stationary: Stationary
    reading: object = None


class Result(NamedTuple):
    moments: tuple
    support: Support
    residual: object
    converged: object
    steps: object
    reads: object


def run(start=1.0, target=1.03, stiffness=(1.0, 1e8), **options):
    """Solve a cubic with exact Newton directions and a finite update budget."""
    start = jnp.broadcast_to(jnp.asarray(start, dtype=jnp.float64), (2,))
    history = []
    stiffness = jnp.asarray(stiffness)

    def solve(
        _operands,
        _cache,
        moments,
        _seeds,
        previous,
        _convention,
        _policy,
        drive,
        _gradient,
        _quadrature,
        _axis_sign,
        *,
        max_steps,
        tolerance,
        **_options,
    ):
        state = moments[0][:, 0]
        steps = jnp.zeros(2, jnp.int32)
        for _ in range(max_steps):
            error = state - drive
            norm = jnp.abs(error + stiffness * error**3)
            active = norm > tolerance
            correction = (error + stiffness * error**3) / (1 + 3 * stiffness * error**2)
            state = jnp.where(active, state - correction, state)
            steps += active.astype(jnp.int32)
        error = state - drive
        norm = jnp.abs(error + stiffness * error**3)
        history.append((np.asarray(drive), np.asarray(steps), np.asarray(norm)))
        return Result(
            (state[:, None],),
            Support(Stationary(previous)),
            norm,
            norm <= tolerance,
            steps,
            steps + 1,
        )

    result = driver.advance_drive(
        None,
        None,
        (start[:, None],),
        jnp.zeros((2, 1, 2)),
        jnp.zeros((2, 1, 2)),
        None,
        None,
        lambda d: d,
        None,
        None,
        start_drive=options.pop("start_drive", 1.0),
        target_drive=target,
        solve=solve,
        initial_relative_step=0.01,
        **options,
    )
    return result, history


def test_scalar_drives_equal_uniform_lane_drives():
    scalar, _ = run(stiffness=(1e8, 1e8))
    lanes, _ = run(
        start_drive=jnp.ones(2), target=jnp.full(2, 1.03), stiffness=(1e8, 1e8)
    )
    assert scalar.certified and lanes.certified
    np.testing.assert_array_equal(scalar.result.moments, lanes.result.moments)
    np.testing.assert_array_equal(lanes.drive, jnp.full(2, scalar.drive))
    assert scalar.updates == lanes.updates
    assert scalar.reads == lanes.reads
    assert len(scalar.substeps) == len(lanes.substeps)
    for shared, independent in zip(scalar.substeps, lanes.substeps, strict=True):
        np.testing.assert_allclose(independent.drive, shared.drive, rtol=0, atol=0)
        np.testing.assert_allclose(
            independent.sub_step, shared.sub_step, rtol=0, atol=0
        )


def test_stiff_lane_does_not_shrink_easy_lane():
    shared, _ = run()
    independent, history = run(start_drive=jnp.ones(2), target=jnp.full(2, 1.03))
    assert independent.certified and shared.certified
    first, second = independent.substeps[:2]
    np.testing.assert_array_equal(first.converged, [True, False])
    assert second.sub_step[0] == pytest.approx(first.sub_step[0] * driver.GROWTH)
    assert second.sub_step[1] == pytest.approx(first.sub_step[1] * driver.SHRINK)
    easy_arrival = next(
        i + 1
        for i, step in enumerate(independent.substeps)
        if step.converged[0] and step.final[0]
    )
    assert easy_arrival < len(shared.substeps)
    for index in range(easy_arrival, len(history)):
        assert history[index][0][0] == 1.03
        assert history[index][1][0] == 0
    print(
        "INDEPENDENCE",
        "easy_subsolves",
        easy_arrival,
        "shared_subsolves",
        len(shared.substeps),
    )


def test_different_start_target_and_stationary_lane_certify():
    start = jnp.asarray([1.0, 2.0])
    target = jnp.asarray([1.03, 1.97])
    result, _ = run(start, target, start_drive=start)
    assert result.certified
    np.testing.assert_array_equal(result.drive, target)
    np.testing.assert_allclose(
        result.result.moments[0][:, 0], target, rtol=0, atol=1e-10
    )
    assert np.all(result.result.residual <= 1e-10)
    held, history = run(start, jnp.asarray([1.0, 1.97]), start_drive=start)
    assert held.certified and np.all(held.result.residual <= 1e-10)
    assert all(row[1][0] == 0 for row in history)


@pytest.mark.parametrize("reuse", [False, True])
def test_solve_batch_maps_lane_profile_into_residual_and_booking(monkeypatch, reuse):
    monkeypatch.setattr(bank, "read_batch", solver_toy._toy_read)
    observed = []

    def residual(_operands, _base, state, support, profile, _quadrature):
        return bank.StepResidual(
            tuple(value - profile for value in state),
            (*state, jnp.asarray(1.0)),
            support.reading,
            support.valid,
        )

    def newton(_operands, state, _support, profile, *_args, **_kwargs):
        cached = _kwargs.get("factor_cache") is not None
        jax.debug.callback(
            lambda reuse, target: observed.append((bool(reuse), float(target))),
            jnp.asarray(cached),
            profile,
        )
        template = solver_toy._unit_update(None, state, None)
        return template._replace(
            update=tuple(0.875 * (profile - value) for value in state)
        )

    monkeypatch.setattr(bank, "step_residual", residual)
    monkeypatch.setattr(bank, "newton_step", newton)
    targets = jnp.asarray([0.25, -0.75])
    moments = tuple(jnp.zeros((2, 1)) for _ in range(3))
    seeds = jnp.zeros((2, 1, 2))
    result = jax.jit(
        lambda state: bank.solve_batch(
            None,
            None,
            state,
            seeds,
            seeds,
            None,
            None,
            lambda lane: targets[lane],
            None,
            None,
            max_steps=14,
            reuse_factor=reuse,
        )
    )(moments)
    jax.block_until_ready(result)
    jax.effects_barrier()
    cached_profiles = [profile for cached, profile in observed if cached]
    assert np.all(result.steps > 1)
    if reuse:
        assert len(cached_profiles) >= 4
        assert set(cached_profiles) == set(np.asarray(targets))
        np.testing.assert_array_equal(result.refactors, jnp.ones(2, jnp.int32))
    else:
        assert not cached_profiles
    assert np.all(result.converged)
    for values in result.moments:
        np.testing.assert_allclose(values[:, 0], targets, rtol=0, atol=1e-10)
    assert np.all(result.residual <= 1e-10)
    print(
        "PROFILE_BRANCH",
        "reuse",
        reuse,
        "steps",
        np.asarray(result.steps),
        "cached_calls",
        len(cached_profiles),
        "cached_profiles",
        cached_profiles,
    )


@pytest.mark.parametrize("warm_support", [False, True])
@pytest.mark.parametrize("stop", ["floor", "ceiling"])
def test_rejected_first_attempt_returns_incumbent_state(warm_support, stop):
    """A failed continuation must not expose a rejected trial as its state."""
    state = (jnp.ones((2, 1)),)
    seeds = jnp.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
    incumbent = Support(Stationary(seeds), state[0])

    def rejected(_operands, _cache, moments, _seeds, previous, *_args, **_kwargs):
        return Result(
            (moments[0] + 100.0,),
            Support(Stationary(previous + 100.0), moments[0] + 100.0),
            jnp.ones(2),
            jnp.zeros(2, bool),
            jnp.ones(2, jnp.int32),
            jnp.ones(2, jnp.int32),
        )

    result = driver.advance_drive(
        None,
        None,
        state,
        seeds,
        seeds,
        None,
        None,
        lambda d: d,
        None,
        None,
        start_drive=jnp.ones(2),
        target_drive=jnp.full(2, 1.03),
        max_substeps=1,
        minimum_relative_step=1.0 if stop == "floor" else 1e-9,
        **({"initial_support": incumbent} if warm_support else {}),
        solve=rejected,
    )
    assert not result.certified
    np.testing.assert_array_equal(result.drive, jnp.ones(2))
    np.testing.assert_array_equal(result.result.moments[0], state[0])
    assert not np.any(result.result.converged)
    assert result.reason == f"sub-step {stop} reached"
    np.testing.assert_array_equal(result.result.support.stationary.position, seeds)
    np.testing.assert_array_equal(result.support_available, [warm_support] * 2)
    if warm_support:
        np.testing.assert_array_equal(result.result.support.reading, incumbent.reading)
    else:
        assert np.all(np.isnan(result.result.support.reading))


def test_rejected_attempt_keeps_last_accepted_lane_support():
    """Accepted support survives alongside a lane without any admitted read."""
    state = (jnp.ones((2, 1)),)
    seeds = jnp.asarray([[[1.0, 2.0]], [[3.0, 4.0]]])
    calls = 0

    def solve(_operands, _cache, moments, _seeds, previous, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        change = 1.0 if calls == 1 else 100.0
        accepted = jnp.asarray([calls == 1, False])
        return Result(
            (moments[0] + change,),
            Support(Stationary(previous + change), moments[0] + change),
            jnp.where(accepted, 0.0, 1.0),
            accepted,
            jnp.ones(2, jnp.int32),
            jnp.ones(2, jnp.int32),
        )

    result = driver.advance_drive(
        None,
        None,
        state,
        seeds,
        seeds,
        None,
        None,
        lambda d: d,
        None,
        None,
        start_drive=jnp.ones(2),
        target_drive=jnp.full(2, 1.03),
        max_substeps=2,
        solve=solve,
    )
    assert calls == 2
    np.testing.assert_array_equal(result.support_available, [True, False])
    np.testing.assert_array_equal(result.result.moments[0][:, 0], [2.0, 1.0])
    np.testing.assert_array_equal(
        result.result.support.stationary.position[0], seeds[0] + 1
    )
    np.testing.assert_array_equal(
        result.result.support.stationary.position[1], seeds[1]
    )
    np.testing.assert_array_equal(result.result.support.reading[0], [2.0])
    assert np.all(np.isnan(result.result.support.reading[1]))
