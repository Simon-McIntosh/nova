"""Compare a runtime correction that shares one adaptive primal trace."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from scripts.prototypes.fine_grid_convergence import (
    handoff,
    runtime_solve,
    solve_carrier,
)


def stable_factor(state, operands, scale, profile):
    """Linearize once at the runtime state, then batch only tangent propagation."""
    count = state.size
    width = runtime_solve.DIRECTION_BLOCK
    blocks = (count + width - 1) // width
    _, product = jax.linearize(
        lambda point: runtime_solve.residual(point, operands, scale, profile), state
    )
    columns = jnp.zeros((blocks * width, count), state.dtype)

    def body(block, packed):
        first = block * width
        indices = first + jnp.arange(width)
        directions = (indices[:, None] == jnp.arange(count)[None]).astype(state.dtype)
        products = jax.vmap(product)(directions)
        return jax.lax.dynamic_update_slice(packed, products, (first, 0))

    matrix = jax.lax.fori_loop(0, blocks, body, columns)[:count].T
    return jsp_linalg.lu_factor(matrix)


def main():
    assert jax.config.jax_enable_x64 is True
    handoff.booked_blend = lambda gap: jnp.zeros_like(gap)
    runtime_solve.factor = stable_factor
    print("RUNTIME_LINEARIZE_FACTOR BOOKED_BLEND_ZERO", flush=True)
    solve_carrier.main()


if __name__ == "__main__":
    main()
