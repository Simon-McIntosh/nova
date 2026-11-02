"""A batched warm Newton solve reaches what each slice's own solve reaches.

The batched step shares one compiled program across independent slices: the
read takes one decision branch for the whole batch and the step is a ``vmap``
over the lanes. A batch that shared one lane's state, support or factor would
agree with itself and disagree with the solve each lane stands in for, so the
lanes are given distinct perturbations and every lane's converged flux is
compared against the same state solved alone.

The solve runs at the diverted 550-cell certificate state the record measures,
not at a coarser rebuild of it. At 64 cells the fixture's converged moments are
not a fixed point of that mesh's operator — the joint residual there starts
near 13 and the iteration walks away from it — so a 64-cell gate would compare
two solves that are both diverging rather than two that converge.

A lane that has met the residual bound is frozen: its update is zeroed while
the remaining lanes iterate. Without the freeze a full Newton step takes back off
converged solve and the batch two-cycles between 1.6442e-7 and 3.19e-10 instead
of satisfying a whole-batch bound.

The branch clause separates a batch-uniform decision from one taken inside the
map: a batch carrying one stale track must rebuild every lane and report the
rebuild on every lane, which a per-lane conditional does not.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine

LANES = 2
MAX_STEPS = 60
RESIDUAL = 1e-10
EQUALITY = 1e-10
LANE_PERTURBATION = (2e-5, 6e-5)
pytestmark = pytest.mark.timeout(1800)


def _gradient(radius, flux):
    return jnp.zeros_like(radius)


@pytest.fixture(scope="module")
def built():
    bank, cache, roots, convention, policy, profile, quadrature, moments, _analytic = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    return {
        "bank": bank,
        "cache": cache,
        "seeds": seeds,
        "convention": convention,
        "policy": policy,
        "profile": profile,
        "quadrature": quadrature,
        "moments": moments,
    }


def _lanes(built):
    moments = built["moments"]
    stacked = tuple(
        jnp.stack([value * (1.0 + scale) for scale in LANE_PERTURBATION])
        for value in moments
    )
    lane_seeds = jnp.broadcast_to(built["seeds"], (LANES, *jnp.shape(built["seeds"])))
    return stacked, lane_seeds


def _largest_residual(residual):
    return jnp.max(jnp.abs(jnp.concatenate(residual.residual, axis=-1)), axis=-1)


def _solve(built, state, steps):
    """One unbatched solve of a single state to the residual bound."""
    bank, cache = built["bank"], built["cache"]
    seeds, convention, policy = built["seeds"], built["convention"], built["policy"]
    profile, quadrature = built["profile"], built["quadrature"]

    def read(one):
        return support_read.read(bank, cache, one, seeds, seeds, convention, policy)

    def residual(one, support):
        return support_read.step_residual(bank, one, one, support, profile, quadrature)

    for index in range(steps):
        support = read(state)
        size = float(_largest_residual(residual(state, support)))
        if size <= RESIDUAL:
            return state, index
        step = support_read.newton_step(
            bank,
            state,
            support,
            profile,
            _gradient,
            quadrature,
            fresh_geometry=True,
        )
        jax.block_until_ready(step.update)
        state = tuple(
            value + change for value, change in zip(state, step.update, strict=True)
        )
    return state, steps


@pytest.mark.slow
def test_batched_solve_reaches_the_single_solve_flux(built):
    stacked, lane_seeds = _lanes(built)
    bank, cache = built["bank"], built["cache"]
    convention, policy = built["convention"], built["policy"]
    profile, quadrature = built["profile"], built["quadrature"]

    def batched(state, hold):
        return support_read.step_batch(
            bank,
            cache,
            state,
            lane_seeds,
            lane_seeds,
            convention,
            policy,
            profile,
            _gradient,
            quadrature,
            fresh_geometry=True,
            converged=hold,
        )

    read_batch = jax.jit(
        lambda state: support_read.read_batch(
            bank, cache, state, lane_seeds, lane_seeds, convention, policy
        )
    )
    batch = jax.jit(batched)

    def lane_residual(one, one_support):
        return _largest_residual(
            support_read.step_residual(bank, one, one, one_support, profile, quadrature)
        )

    hold = np.zeros(LANES, dtype=bool)
    steps = 0
    trace = []
    while steps < MAX_STEPS:
        support = read_batch(stacked)
        sizes = np.asarray(jax.vmap(lane_residual)(stacked, support))
        trace.append(sizes.tolist())
        hold = hold | (sizes <= RESIDUAL)
        if hold.all():
            break
        correction = batch(stacked, jnp.asarray(hold))
        jax.block_until_ready(correction.step.update)
        stacked = tuple(
            value + change
            for value, change in zip(stacked, correction.step.update, strict=True)
        )
        steps += 1
    print(f"batched trace: {trace}")
    final = np.asarray(jax.vmap(lane_residual)(stacked, read_batch(stacked)))
    print(f"batched residuals: {final.tolist()} after {steps} steps")
    assert final.max() <= RESIDUAL

    separate, separate_steps = [], []
    for lane in range(LANES):
        one = tuple(value[lane] for value in stacked)
        alone, count = _solve(built, one, MAX_STEPS)
        separate.append(alone)
        separate_steps.append(count)
    for lane, alone in enumerate(separate):
        for name, value, reference in zip(
            ("radial", "vertical", "density"),
            (stacked[0][lane], stacked[1][lane], stacked[2][lane]),
            alone,
        ):
            np.testing.assert_allclose(
                np.asarray(value),
                np.asarray(reference),
                rtol=EQUALITY,
                atol=0.0,
                err_msg=f"lane {lane} {name} converged flux",
            )

    # The lanes are genuinely distinct states: a batch that broadcast one
    # lane's state across the axis would return equal fluxes and pass a
    # comparison written against a single shared state.
    assert not np.allclose(np.asarray(stacked[0][0]), np.asarray(stacked[0][1]))

    sample = tuple(value[0] for value in stacked)
    exact = support_read.newton_step(
        bank,
        sample,
        support_read.read(
            bank,
            cache,
            sample,
            built["seeds"],
            built["seeds"],
            convention,
            policy,
        ),
        profile,
        _gradient,
        quadrature,
        fresh_geometry=True,
        condition_diagnostics=True,
    )
    print(
        f"condition at the converged lane: before {float(exact.condition_before):.3e} "
        f"after equilibration {float(exact.condition_after):.3e}; "
        f"batched steps {steps}, single steps {separate_steps}"
    )
    assert np.isfinite(float(exact.condition_before))
    assert float(exact.condition_before) > 0.0


@pytest.mark.slow
def test_one_stale_lane_rebuilds_the_whole_batch(built):
    """The batch takes one reload branch, and every lane reports it."""
    bank, cache = built["bank"], built["cache"]
    convention, policy = built["convention"], built["policy"]
    stacked, lane_seeds = _lanes(built)

    def batch(state, tracking):
        return support_read.read_batch(
            bank,
            cache,
            state,
            lane_seeds,
            lane_seeds,
            convention,
            policy,
            tracking=tracking,
        )

    compiled = jax.jit(batch)
    opened = compiled(stacked, None)
    jax.block_until_ready(opened.valid)
    tracking = jax.vmap(support_read.tracking_of)(opened)

    steady = compiled(stacked, tracking)
    jax.block_until_ready(steady.valid)
    assert not np.asarray(steady.rebuilt).any(), (
        "a steady batch must not rebuild any lane"
    )
    np.testing.assert_array_equal(
        np.asarray(steady.rebuilds), np.asarray(tracking.rebuilds)
    )

    # One lane's tracked cell is stale, so that lane alone asks for a reload
    # and the batch's single decision must take the branch for every lane.
    stale = tracking._replace(
        cells=tracking.cells.at[1].set(jnp.full_like(tracking.cells[1], -1))
    )
    rebuilt = compiled(stacked, stale)
    jax.block_until_ready(rebuilt.valid)
    print(
        f"stale batch rebuilt={np.asarray(rebuilt.rebuilt).tolist()} "
        f"rebuilds={np.asarray(rebuilt.rebuilds).tolist()}"
    )
    assert np.asarray(rebuilt.rebuilt).all(), (
        "one stale lane must rebuild the whole batch"
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt.rebuilds),
        np.asarray(tracking.rebuilds) + 1,
    )
