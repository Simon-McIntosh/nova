"""The batched read compiles once and carries its tracking across slices.

One compiled program serves every state of the batch's shape, so the bank's
operands and the moments are traced arguments of the vmapped read and not
constants folded into it. The clause here is the compile count: a second batch
of the same shape with different moments must reuse the executable the first
one built. A read whose bank is captured as a closure constant instead has to
build a new program whenever the captured value differs, which is what this
check refuses.
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
from scripts.prototypes.tracked_read_batch.measure import batch_program, tiled

CELLS = 64
# Building the certificate's bank and compiling the batch costs minutes on the
# device, which the suite-wide 300 s bound reports as a timeout rather than a
# result.
pytestmark = pytest.mark.timeout(1800)
POLICY_RADIUS = 0.054806712567833996
COUNT = 2


def _built():
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


@pytest.mark.slow
def test_a_batch_read_equals_the_single_read_slice_by_slice():
    """Every lane of the batch must return what one unbatched read returns.

    The batch shares one compiled program across its lanes, so a lane that
    picked up another lane's cells, seeds or decision would agree with the
    batch and disagree with the read it is standing in for. Each lane is
    therefore compared against the same state read once.
    """
    case, operands, cache = _built()
    convention, policy = case["convention"], case["policy"]
    seeds = jnp.asarray(case["roots"])

    # Distinct states, one per lane: a batch that shared one lane's reading
    # across the axis would agree with itself and disagree with the read each
    # lane is standing in for, which one tiled state cannot show.
    lanes = [case["states"]["converged"], case["states"]["trial"]]
    assert len(lanes) == COUNT
    stacked = tuple(
        jnp.stack([lane[leaf] for lane in lanes]) for leaf in range(len(lanes[0]))
    )
    program = batch_program(operands, cache, convention, policy)
    batch = program(stacked, tiled(seeds, COUNT), tiled(seeds, COUNT), None)
    jax.block_until_ready(batch)

    for index, moments in enumerate(lanes):
        single = support_read.read(
            operands, cache, moments, seeds, seeds, convention, policy
        )
        np.testing.assert_allclose(
            np.asarray(batch.stationary.position)[index],
            np.asarray(single.stationary.position),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.asarray(batch.stationary.status)[index],
            np.asarray(single.stationary.status),
        )
        np.testing.assert_allclose(
            float(batch.reading.boundary_flux[index]),
            float(single.reading.boundary_flux),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.asarray(batch.cells)[index], np.asarray(single.cells)
        )


@pytest.mark.slow
def test_one_program_serves_a_second_batch_of_another_state():
    case, operands, cache = _built()
    convention, policy = case["convention"], case["policy"]
    seeds = tiled(jnp.asarray(case["roots"]), COUNT)
    converged = tuple(tiled(moment, COUNT) for moment in case["states"]["converged"])
    trial = tuple(tiled(moment, COUNT) for moment in case["states"]["trial"])

    program = batch_program(operands, cache, convention, policy)
    assert hasattr(program, "_cache_size"), (
        "the batch must hand back one compiled program, not a per-call rebuild"
    )
    first = program(converged, seeds, seeds, None)
    tracking = jax.vmap(support_read.tracking_of)(first)
    settled = program(converged, seeds, seeds, tracking)
    jax.block_until_ready(settled)
    assert not bool(jnp.any(settled.rebuilt))
    built = program._cache_size()

    program(trial, seeds, seeds, tracking)
    assert program._cache_size() == built
