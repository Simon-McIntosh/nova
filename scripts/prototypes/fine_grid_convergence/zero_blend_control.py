"""Run the same runtime correction with the booked cut contribution absent."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp

from scripts.prototypes.fine_grid_convergence import handoff, solve_carrier


def main():
    assert jax.config.jax_enable_x64 is True
    handoff.booked_blend = lambda gap: jnp.zeros_like(gap)
    print("BOOKED_BLEND_ZERO", flush=True)
    solve_carrier.main()


if __name__ == "__main__":
    main()
