"""Observe fragment rebuilding for an unchanged carried support."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import numpy as np

from nova.equilibrium import topology
from tests.test_candidate_inspection_batched import small_case

assert jax.config.jax_enable_x64
field, geometry, convention, policy, solved = small_case()
decision = jax.jit(topology.read_at_stationary_points)(
    field, geometry, convention, policy, solved
)
jax.block_until_ready(decision)
assert bool(decision.valid)
calls = []
original = topology.quadratic_cell_fragments


def observed(vertices, count, coefficient, capacity):
    jax.debug.callback(lambda value: calls.append(float(value)), coefficient[0])
    return original(vertices, count, coefficient, capacity)


topology.quadratic_cell_fragments = observed
refresh = jax.jit(topology.refresh_at_stationary_points)
for index in range(2):
    calls.clear()
    actual = refresh(field, geometry, convention, policy, solved, decision)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    np.testing.assert_allclose(actual.fragment_area, decision.fragment_area, atol=1e-13)
    assert len(calls) == len(geometry.centre), (len(calls), len(geometry.centre))
    print(
        f"UNCHANGED_REFRESH pass={index} clipped_cells={len(calls)} "
        f"total_cells={len(geometry.centre)}",
        flush=True,
    )
