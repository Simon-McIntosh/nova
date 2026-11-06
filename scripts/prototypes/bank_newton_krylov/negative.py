"""Remove block preconditioning at the iteration count measured with it."""

import jax.numpy as jnp
import pytest

from nova.equilibrium import support_read


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if "test_diverted_physical_update_matches_double_reference" not in item.name:
        yield
        return
    original = support_read._block_gmres

    def unpreconditioned(matrix, right, inverse, indices, **kwargs):
        _, _, _, count = original(matrix, right, inverse, indices, **kwargs)
        identity = jnp.tile(jnp.eye(3, dtype=matrix.dtype), (right.size // 3, 1))
        return original(matrix, right, identity, indices, max_iterations=count)

    support_read._block_gmres = unpreconditioned
    try:
        yield
    finally:
        support_read._block_gmres = original
