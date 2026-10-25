"""The sampled arc root keeps a finite tangent across its discriminant seam."""

import jax
import jax.numpy as jnp
import numpy as np


def test_arc_root_tangent_is_finite_across_the_discriminant_seam():
    """A weight reparametrisation that crosses the seam keeps a finite direction."""
    from nova.equilibrium.separatrix_clip import _traced_quadratic_arc

    coefficient = jnp.asarray([[-0.25, 0.0, 0.0, 1.0, 0.0, 1.0]])
    tangent = jnp.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    centre = jnp.asarray([[0.0, 0.0]])
    scale = jnp.asarray([[1.0, 1.0]])
    start = jnp.asarray([[-1.0, 0.0]])
    end = jnp.asarray([[1.0, 0.0]])

    def arc(weights):
        return _traced_quadratic_arc(start, end, weights, centre, scale)

    value, direction = jax.jvp(arc, (coefficient,), (tangent,))
    assert bool(np.all(np.isfinite(np.asarray(value))))
    assert bool(np.all(np.isfinite(np.asarray(direction))))