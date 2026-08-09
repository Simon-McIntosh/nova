"""Fixed-shape morphology shared by accelerator equilibrium kernels."""

import jax.numpy as jnp


def _dilate4(mask: jnp.ndarray) -> jnp.ndarray:
    """Return one false-bordered four-neighbour dilation."""
    up = jnp.zeros_like(mask).at[1:, :].set(mask[:-1, :])
    down = jnp.zeros_like(mask).at[:-1, :].set(mask[1:, :])
    left = jnp.zeros_like(mask).at[:, 1:].set(mask[:, :-1])
    right = jnp.zeros_like(mask).at[:, :-1].set(mask[:, 1:])
    return mask | up | down | left | right
