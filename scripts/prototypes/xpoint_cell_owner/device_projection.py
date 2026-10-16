"""Project a fitted quadratic onto a measured saddle jet at fixed shape."""

import jax.numpy as jnp


def saddle_constrained_coefficients(coefficients, centres, pitches, saddle, level):
    """Keep each Bell Hessian and impose one shared saddle value and gradient."""
    local = (saddle - centres) / pitches[:, None]
    x, y = local[:, 0], local[:, 1]
    radial_squared = coefficients[:, 3]
    cross = coefficients[:, 4]
    vertical_squared = coefficients[:, 5]
    radial = -2 * radial_squared * x - cross * y
    vertical = -cross * x - 2 * vertical_squared * y
    constant = level + radial_squared * x * x + cross * x * y + vertical_squared * y * y
    return jnp.stack(
        (constant, radial, vertical, radial_squared, cross, vertical_squared),
        axis=1,
    )
