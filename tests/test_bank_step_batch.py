"""A batched warm Newton step returns what each slice's own step returns.

The batched step shares one compiled program across independent slices: the
read takes one decision branch for the whole batch and the step is a ``vmap``
over the lanes. A batch that shared one lane's state, support or factor would
agree with itself and disagree with the step each lane stands in for, so the
lanes are given distinct perturbations and every lane is compared against the
same state stepped alone.

The branch clause is what separates a batch-uniform decision from a decision
taken inside the map: a batch carrying one stale track must rebuild every lane
and report the rebuild on every lane, which a per-lane conditional does not.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read
from scripts.prototypes.tracked_read.measure import build_kind, cache_for
from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures.measure import analytic_profile

CELLS = 64
pytestmark = pytest.mark.timeout(1800)
POLICY_RADIUS = 0.054806712567833996
LANE_PERTURBATION = (1e-4, -2e-4)


@pytest.fixture(scope="module")
def built():
    case = build_kind("diverted", CELLS, POLICY_RADIUS)
    operands = support_read.SupportOperands.from_rows(
        case["geometry"],
        case["matrix"],
        case["exterior"],
        case["nodes"],
        case["sample_index"],
        bucket=len(case["nodes"]),
    )
    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    quadrature = current.Quadrature(sigma=case["convention"].sigma, segments=64)
    return case, operands, cache_for(operands), profile, quadrature


def _gradient(radius, flux):
    return jnp.zeros_like(radius)


def _perturbed(lane, scale):
    """One lane's distinct state: the converged moments shifted by ``scale``."""
    return tuple(value * (1.0 + scale) for value in lane)


def _stack(lanes):
    return tuple(
        jnp.stack([lane[index] for lane in lanes]) for index in range(len(lanes[0]))
    )


@pytest.mark.slow
def test_batched_step_equals_the_single_step_lane_by_lane(built):
    case, operands, cache, profile, quadrature = built
    convention, policy = case["convention"], case["policy"]
    seconds = case["states"]["converged"]
    seeds = jnp.asarray(case["roots"])

    lanes = [_perturbed(seconds, scale) for scale in LANE_PERTURBATION]
    stacked = _stack(lanes)
    count = jnp.shape(stacked[0])[0]
    lane_seeds = jnp.broadcast_to(seeds, (count, *jnp.shape(seeds)))

    def batch(state, tracking):
        return support_read.step_batch(
            operands,
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
            tracking=tracking,
        )

    compiled = jax.jit(batch)
    batched = compiled(stacked, None)
    jax.block_until_ready(batched.step.update)

    for lane, state in enumerate(lanes):
        alone = support_read.newton_step(
            operands,
            state,
            support_read.read(
                operands, cache, state, seeds, seeds, convention, policy
            ),
            profile,
            _gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )
        jax.block_until_ready(alone.update)
        for name, together, single in zip(
            ("radial", "vertical", "density"), batched.step.update, alone.update
        ):
            np.testing.assert_allclose(
                np.asarray(together[lane]),
                np.asarray(single),
                rtol=1e-12,
                atol=0.0,
                err_msg=f"lane {lane} {name} update",
            )
        np.testing.assert_allclose(
            float(batched.step.linear_residual[lane]),
            float(alone.linear_residual),
            rtol=1e-12,
            atol=0.0,
            err_msg=f"lane {lane} linear residual",
        )

    # The lanes are genuinely distinct states: a batch that broadcast one
    # lane's step across the axis would return equal updates and pass a
    # comparison written against a single shared state.
    radial = np.asarray(batched.step.update[0])
    assert not np.allclose(radial[0], radial[1])


@pytest.mark.slow
def test_one_stale_lane_rebuilds_the_whole_batch(built):
    """The batch takes one reload branch, and every lane reports it."""
    case, operands, cache, profile, quadrature = built
    convention, policy = case["convention"], case["policy"]
    seeds = jnp.asarray(case["roots"])
    lanes = [_perturbed(case["states"]["converged"], scale) for scale in LANE_PERTURBATION]
    stacked = _stack(lanes)
    count = jnp.shape(stacked[0])[0]
    lane_seeds = jnp.broadcast_to(seeds, (count, *jnp.shape(seeds)))

    def batch(state, tracking):
        return support_read.step_batch(
            operands,
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
            tracking=tracking,
        )

    compiled = jax.jit(batch)
    opened = compiled(stacked, None)
    jax.block_until_ready(opened.support.valid)
    tracking = jax.vmap(support_read.tracking_of)(opened.support)

    steady = compiled(stacked, tracking)
    jax.block_until_ready(steady.support.valid)
    assert not np.asarray(steady.support.rebuilt).any(), (
        "a steady batch must not rebuild any lane"
    )
    np.testing.assert_array_equal(
        np.asarray(steady.support.rebuilds), np.asarray(tracking.rebuilds)
    )

    # One lane's tracked cell is stale, so that lane alone asks for a reload
    # and the batch's single decision must take the branch for every lane.
    stale = tracking._replace(
        cells=tracking.cells.at[1].set(jnp.full_like(tracking.cells[1], -1))
    )
    rebuilt = compiled(stacked, stale)
    jax.block_until_ready(rebuilt.support.valid)
    assert np.asarray(rebuilt.support.rebuilt).all(), (
        "one stale lane must rebuild the whole batch"
    )
    np.testing.assert_array_equal(
        np.asarray(rebuilt.support.rebuilds),
        np.asarray(tracking.rebuilds) + 1,
    )