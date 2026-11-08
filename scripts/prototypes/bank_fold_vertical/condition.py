"""Balanced bank and centroid-bordered Jacobian spectra at one retained state."""

import jax.numpy as jnp

from nova.equilibrium import support_read
from scripts.prototypes.bank_fold_vertical.measure import centroid, shifted
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture
from scripts.prototypes.bank_warm_start import streams


def condition(machine, state, seed, field, drive):
    """Return bordered and unpinned smallest singular values on a fresh read."""
    op, cache = shifted(machine, field)
    profile = streams.scaled_profile(machine.profile, drive)
    support = support_read.read(
        op, cache, state, seed, seed, machine.convention, machine.policy
    )
    residual = support_read.step_residual(
        op, state, state, support, profile, machine.quadrature
    )
    matrix = support_read.step_jacobian(
        op,
        state,
        support,
        profile,
        profile_fixture.gradient,
        machine.quadrature,
        booked=residual.booked,
        fresh_geometry=True,
    )
    balanced, row, column = support_read._equilibrate(matrix, state, residual.booked)
    op_plus, cache_plus = shifted(machine, field + 1e-6)
    support_plus = support_read.read(
        op_plus, cache_plus, state, seed, seed, machine.convention, machine.policy
    )
    residual_plus = support_read.step_residual(
        op_plus, state, state, support_plus, profile, machine.quadrature
    )

    def flat(value):
        return jnp.concatenate(value)

    field_column = (flat(residual_plus.residual) - flat(residual.residual)) / 1e-6
    cells = state[0].size
    total = jnp.sum(state[0])
    height = centroid(state)
    height_row = jnp.concatenate(
        (
            jnp.full(cells, -height / total),
            jnp.zeros(cells),
            jnp.full(cells, 1 / total),
        )
    )
    bordered = jnp.block(
        [
            [balanced, (field_column * 1e-3 / row)[:, None]],
            [(height_row * column * 1e-3)[None, :], jnp.zeros((1, 1))],
        ]
    )
    original = jnp.linalg.svd(balanced, compute_uv=False)
    pinned = jnp.linalg.svd(bordered, compute_uv=False)
    return pinned[-1], original[-1], support.valid & support_plus.valid
