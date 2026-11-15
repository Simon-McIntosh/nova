"""One compiled packed-sheet program for pair and tile comparisons."""

import jax
import jax.numpy as jnp

from nova.biot.sheet import packed_sheet_field
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64


@jax.jit
def packed_field(*values):
    """Return radial and vertical fields for equally shaped arrays of pairs."""
    return packed_sheet_field(jnp, *values)


def axis_pair(values):
    """Adapt one six-coordinate pair to the shared array program."""
    return jnp.stack(packed_field(*[value[None] for value in values])).ravel()


axis_pair_jacobian = jax.jit(jax.jacfwd(axis_pair))
