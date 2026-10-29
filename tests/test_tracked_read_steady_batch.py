"""A steady batch reads what each slice would read alone, and rebuilds nothing.

The batched read shares one decision branch across its lanes, so a batch that
shared one lane's reading, seeds or cells would agree with itself and disagree
with the read each lane stands in for. Each lane therefore carries its own
tracking state and is compared against the same state read once.

The steady clause is that no lane asks for a rebuild and the batch executes the
carry branch: a batch that lost the tracking state would rebuild every lane,
which the per-lane flags here refuse even where the numbers would agree. A batch
in which one lane does rebuild must reconstruct every lane correctly, because
the branch is taken once for the whole batch.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import support_read
from scripts.prototypes.tracked_read.measure import build_kind, cache_for

CELLS = 64
# Building the certificate's bank and compiling the batched programs costs
# minutes on the device, which the suite-wide bound reports as a timeout rather
# than a result.
pytestmark = pytest.mark.timeout(1800)
POLICY_RADIUS = 0.054806712567833996
COUNT = 2


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
    return case, operands, cache_for(operands)


def _stack(lanes):
    return tuple(
        jnp.stack([lane[leaf] for lane in lanes]) for leaf in range(len(lanes[0]))
    )


def _batch(operands, cache, case, moments, seeds, tracking):
    convention, policy = case["convention"], case["policy"]
    return support_read.read_batch(
        operands,
        cache,
        moments,
        seeds,
        seeds,
        convention,
        policy,
        tracking=tracking,
    )


def _tiled_tracking(tracking, count):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (count, *jnp.shape(leaf))), tracking
    )


@pytest.mark.slow
def test_batched_steady_read_equals_the_single_read_lane_by_lane(built):
    """Every lane carries its own tracking state and none of them rebuilds."""
    case, operands, cache = built
    convention, policy = case["convention"], case["policy"]
    seeds = jnp.asarray(case["roots"])

    # Distinct states, one per lane: a batch that shared one lane's reading
    # across the axis would agree with itself and disagree with the read each
    # lane is standing in for. One tiled state cannot show that.
    lanes = [case["states"]["converged"], case["states"]["trial"]]
    assert len(lanes) == COUNT
    stacked = _stack(lanes)

    opened = _batch(operands, cache, case, stacked, seeds, None)
    jax.block_until_ready(opened)
    tracking = jax.vmap(support_read.tracking_of)(opened)
    jax.block_until_ready(tracking)

    steady = _batch(operands, cache, case, stacked, seeds, tracking)
    jax.block_until_ready(steady)

    rebuilt_per_lane = np.asarray(steady.rebuilt)
    assert not rebuilt_per_lane.any(), (
        f"a lane rebuilt at a steady state: {rebuilt_per_lane.tolist()}"
    )
    np.testing.assert_array_equal(
        np.asarray(steady.rebuilds), np.asarray(tracking.rebuilds)
    )

    for index, moments in enumerate(lanes):
        single = support_read.read(
            operands,
            cache,
            moments,
            seeds,
            seeds,
            convention,
            policy,
            tracking=tracking[index],
        )
        np.testing.assert_allclose(
            np.asarray(steady.stationary.position)[index],
            np.asarray(single.stationary.position),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.asarray(steady.stationary.status)[index],
            np.asarray(single.stationary.status),
        )
        np.testing.assert_array_equal(
            np.asarray(steady.cells)[index], np.asarray(single.cells)
        )
        np.testing.assert_allclose(
            float(steady.reading.boundary_flux[index]),
            float(single.reading.boundary_flux),
            rtol=1e-12,
            atol=1e-12,
        )
        assert bool(steady.valid[index]) == bool(single.valid)


@pytest.mark.slow
def test_a_batch_with_one_rebuilding_lane_rebuilds_and_still_matches(built):
    """One lane's stale track takes the batch's rebuild branch, and all are right.

    The decision is taken once for the whole batch, so a batch holding a lane
    that must rebuild pays the whole-mesh reload for every lane. That is correct
    only if the reloaded decision is each lane's own, which is what this
    compares against the single reads.
    """
    case, operands, cache = built
    convention, policy = case["convention"], case["policy"]
    seeds = jnp.asarray(case["roots"])
    lanes = [case["states"]["converged"], case["states"]["trial"]]

    opened = _batch(operands, cache, case, _stack(lanes), seeds, None)
    jax.block_until_ready(opened)
    tracking = jax.vmap(support_read.tracking_of)(opened)

    # One lane's cell membership is stale, so that lane alone asks for a
    # rebuild and the batch is forced down the reload branch.
    stale = tracking._replace(
        cells=tracking.cells.at[1].set(jnp.full_like(tracking.cells[1], -1))
    )
    stencil = jax.vmap(
        lambda state, seed: support_read.stationary_stencil(
            operands, cache, state, seed, seed, convention
        )
    )(tuple(stacked for stacked in _stack(lanes)), jnp.stack([seeds, seeds]))
    per_lane = jax.vmap(support_read.rebuild_needed)(stencil, stale)
    jax.block_until_ready(per_lane)
    assert bool(jnp.any(per_lane)), (
        "the mixed batch must hold a lane that asks for a rebuild, "
        f"got {np.asarray(per_lane).tolist()}"
    )

    result = _batch(operands, cache, case, _stack(lanes), seeds, stale)
    jax.block_until_ready(result)
    assert bool(jnp.all(result.rebuilt)), "the batch must take the reload branch"

    for index, moments in enumerate(lanes):
        single = support_read.read(
            operands, cache, moments, seeds, seeds, convention, policy
        )
        np.testing.assert_allclose(
            np.asarray(result.stationary.position)[index],
            np.asarray(single.stationary.position),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            float(result.reading.boundary_flux[index]),
            float(single.reading.boundary_flux),
            rtol=1e-12,
            atol=1e-12,
        )