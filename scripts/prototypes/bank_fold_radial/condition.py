"""Balanced bank and centroid-bordered Jacobian spectra at one retained state."""

import jax.numpy as jnp

from nova.equilibrium import support_read
from scripts.prototypes.bank_fold_radial.measure import (
    ROW_SCALE,
    SOURCE_SCALE,
    centroid_row,
    linearized_system,
)


def condition(machine, state, seed, field, drive):
    """Return bordered and unpinned smallest singular values on a fresh read."""
    matrix, booked, _, field_column, valid, field_valid = linearized_system(
        machine, state, seed, drive, field
    )
    balanced, row, column = support_read._equilibrate(matrix, state, booked)
    centres = machine.operands.geometry.centre[:, 0]
    radius_row = centroid_row(state, centres)
    bordered = jnp.block(
        [
            [balanced, (field_column * SOURCE_SCALE / row)[:, None]],
            [(radius_row * column * ROW_SCALE)[None, :], jnp.zeros((1, 1))],
        ]
    )
    original = jnp.linalg.svd(balanced, compute_uv=False)
    pinned = jnp.linalg.svd(bordered, compute_uv=False)
    return pinned[-1], original[-1], valid & field_valid
