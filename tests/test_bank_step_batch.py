"""A batched warm Newton solve reaches what each slice's own solve reaches.

The batched step shares one compiled program across independent slices: the
read takes one decision branch for the whole batch and the step is a ``vmap``
over the lanes. A batch that shared one lane's state, support or factor would
agree with itself and disagree with the solve each lane stands in for, so the
lanes are given distinct perturbations and every lane's converged flux is
compared against the same state solved alone.

Equality is asserted at the converged state rather than after one step. A
batched dense solve and a batched tangent are not bit-identical to their
single-lane forms on the device, because the two take different LU paths, so a
single step can differ at the level the factor's condition number allows while
the two solves land on the same flux. The condition estimate is recorded when
the clause runs so that difference is read against the conditioning that
permits it.

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

from nova.equilibrium import support_read
from scripts.prototypes.bank_step_batch.fixture import build_case
from scripts.prototypes.tracked_read.measure import cache_for

CELLS = 64
LANES = 2
MAX_STEPS = 120
RESIDUAL = 1e-10
EQUALITY = 1e-10
pytestmark = pytest.mark.timeout(1800)
POLICY_RADIUS = 0.054806712567833996
LANE_PERTURBATION = (1e-5, 3e-5)


@pytest.fixture(scope="module")
def built():
    case = build_case("diverted", CELLS, POLICY_RADIUS)
    operands = support_read.SupportOperands.from_rows(
        case["geometry"],
        case["matrix"],
        case["exterior"],
        case["nodes"],
        case["sample_index"],
        bucket=len(case["nodes"]),
    )
    return case, operands, cache_for(operands)


def _seeds(case):
    """Seeds offset a fraction of a cell pitch, which is what keeps a read in
    the basin after the moments move by a small multiple."""
    roots = jnp.asarray(case["roots"])
    return roots + 0.03 * jnp.median(case["geometry"].pitch)


def _gradient(radius, flux):
    return jnp.zeros_like(radius)


def _perturbed(lane, scale):
    """One lane's distinct state: the converged moments shifted by ``scale``."""
    return tuple(value * (1.0 + scale) for value in lane)


def _stack(lanes):
    return tuple(
        jnp.stack([lane[index] for lane in lanes]) for index in range(len(lanes[0]))
    )


def _largest_residual(residual):
    return jnp.max(jnp.abs(jnp.concatenate(residual.residual, axis=-1)), axis=-1)


@pytest.mark.slow
def test_batched_solve_reaches_the_single_solve_flux(built):
    case, operands, cache = built
    convention, policy = case["convention"], case["policy"]
    profile, quadrature = case["profile"], case["quadrature"]
    seeds = _seeds(case)

    lanes = [
        _perturbed(case["states"]["converged"], scale) for scale in LANE_PERTURBATION
    ]
    stacked = _stack(lanes)
    lane_seeds = jnp.broadcast_to(seeds, (LANES, *jnp.shape(seeds)))

    def batched_step(state):
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
        )

    def lane_read(state):
        return support_read.read(
            operands, cache, state, seeds, seeds, convention, policy
        )

    def lane_residual(state, support):
        return support_read.step_residual(
            operands, state, state, support, profile, quadrature
        )

    def lane_step(state):
        return support_read.newton_step(
            operands,
            state,
            lane_read(state),
            profile,
            _gradient,
            quadrature,
            fresh_geometry=True,
        )

    batch = jax.jit(batched_step)
    residual = jax.jit(lambda state, support: jax.vmap(lane_residual)(state, support))
    read_batch = jax.jit(
        lambda state: support_read.read_batch(
            operands, cache, state, lane_seeds, lane_seeds, convention, policy
        )
    )
    single_step = jax.jit(lane_step)
    single_residual = jax.jit(lane_residual)

    together = stacked
    batched_support = read_batch(together)
    batched_steps = 0
    while batched_steps < MAX_STEPS:
        sizes = _largest_residual(residual(together, batched_support))
        if float(jnp.max(sizes)) <= RESIDUAL:
            break
        together = tuple(
            value + change
            for value, change in zip(together, batch(together).step.update, strict=True)
        )
        batched_support = read_batch(together)
        batched_steps += 1
    batched_sizes = _largest_residual(residual(together, batched_support))
    jax.block_until_ready(batched_sizes)
    print(f"batched residuals: {np.asarray(batched_sizes).tolist()}")

    separate, separate_steps, separate_sizes = [], [], []
    for lane, state in enumerate(lanes):
        steps = 0
        support = lane_read(state)
        while steps < MAX_STEPS:
            size = float(_largest_residual(single_residual(state, support)))
            if size <= RESIDUAL:
                break
            correction = single_step(state)
            jax.block_until_ready(correction.update)
            state = tuple(
                value + change
                for value, change in zip(state, correction.update, strict=True)
            )
            support = lane_read(state)
            steps += 1
        size = float(_largest_residual(single_residual(state, support)))
        separate.append(state)
        separate_steps.append(steps)
        separate_sizes.append(size)
        print(f"lane {lane} alone residual: {size}")

    for lane, alone in enumerate(separate):
        for name, value, reference in zip(
            ("radial", "vertical", "density"), _lane(together, lane), alone
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
    radial = np.asarray(_lane(together, 0)[0])
    assert not np.allclose(radial, np.asarray(_lane(together, 1)[0]))

    # One step past convergence, with exact condition estimates, records why a
    # single step's flux differs between the batched and single paths at all.
    sample = _lane(together, 0)
    exact = support_read.newton_step(
        operands,
        sample,
        support_read.read(operands, cache, sample, seeds, seeds, convention, policy),
        profile,
        _gradient,
        quadrature,
        fresh_geometry=True,
        condition_diagnostics=True,
    )
    print(
        f"condition at the converged lane: before {float(exact.condition_before):.3e} "
        f"after equilibration {float(exact.condition_after):.3e}; "
        f"batched steps {batched_steps}, single steps {separate_steps}"
    )
    assert np.isfinite(float(exact.condition_before))
    assert float(exact.condition_before) > 0.0


def _lane(stacked, lane):
    return tuple(value[lane] for value in stacked)


@pytest.mark.slow
def test_one_stale_lane_rebuilds_the_whole_batch(built):
    """The batch takes one reload branch, and every lane reports it."""
    case, operands, cache = built
    convention, policy = case["convention"], case["policy"]
    profile, quadrature = case["profile"], case["quadrature"]
    seeds = _seeds(case)
    lanes = [
        _perturbed(case["states"]["converged"], scale) for scale in LANE_PERTURBATION
    ]
    stacked = _stack(lanes)
    lane_seeds = jnp.broadcast_to(seeds, (LANES, *jnp.shape(seeds)))

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
