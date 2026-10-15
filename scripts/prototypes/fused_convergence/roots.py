"""Implicit tangent for simple cubic ray intersections in the prototype."""

import jax
import jax.numpy as jnp

from nova.equilibrium import topology


_ORIGINAL_PARAMETERS = topology._cubic_parameters


@jax.custom_jvp
def cubic_parameters(cubic, quadratic, linear, constant):
    return _ORIGINAL_PARAMETERS(cubic, quadratic, linear, constant)


@cubic_parameters.defjvp
def _cubic_parameters_jvp(primals, tangents):
    cubic, quadratic, linear, constant = primals
    dcubic, dquadratic, dlinear, dconstant = tangents
    roots = _ORIGINAL_PARAMETERS(*primals)
    finite = jnp.isfinite(roots)
    safe_roots = jnp.where(finite, roots, 0.0)
    numerator = (
        dcubic[..., None] * safe_roots**3
        + dquadratic[..., None] * safe_roots**2
        + dlinear[..., None] * safe_roots
        + dconstant[..., None]
    )
    denominator = (
        3 * cubic[..., None] * safe_roots**2
        + 2 * quadratic[..., None] * safe_roots
        + linear[..., None]
    )
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(cubic), jnp.abs(quadratic)),
        jnp.maximum(jnp.abs(linear), jnp.abs(constant)),
    )
    simple = jnp.abs(denominator) > 64 * jnp.finfo(roots.dtype).eps * scale[..., None]
    tangent = jnp.where(
        finite & simple, -numerator / jnp.where(simple, denominator, 1), 0.0
    )
    return roots, tangent


def install():
    """Give the read and booking code the same ray-root tangent."""
    topology._cubic_parameters = cubic_parameters


def restore():
    """Return to the imported ray-root implementation."""
    topology._cubic_parameters = _ORIGINAL_PARAMETERS
