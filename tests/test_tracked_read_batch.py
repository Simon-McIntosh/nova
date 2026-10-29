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
import pytest

from nova.equilibrium import support_read
from scripts.prototypes.tracked_read.measure import build_kind, cache_for
from scripts.prototypes.tracked_read_batch.measure import batched, tiled

CELLS = 128
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
def test_one_program_serves_a_second_batch_of_another_state():
    case, operands, cache = _built()
    convention, policy = case["convention"], case["policy"]
    seeds = tiled(jnp.asarray(case["roots"]), COUNT)
    converged = tuple(tiled(moment, COUNT) for moment in case["states"]["converged"])
    trial = tuple(tiled(moment, COUNT) for moment in case["states"]["trial"])

    program = jax.jit(batched(operands, cache, convention, policy))
    first = program(converged, seeds, seeds, None)
    tracking = jax.vmap(support_read.tracking_of)(first)
    settled = program(converged, seeds, seeds, tracking)
    jax.block_until_ready(settled)
    assert not bool(jnp.any(settled.rebuilt))
    built = program._cache_size()

    program(trial, seeds, seeds, tracking)
    assert program._cache_size() == built
