"""Track the analytic certificate's axis under a prescribed vertical field."""

import jax
import jax.numpy as jnp


def shifted_axis(field, seed, slope, pitch, iterations=32):
    """Polish the closed-form field, independently of the Bell jet map."""

    def step(_, point):
        jet = field.evaluate(point)
        gradient = jet.gradient + jnp.array((0.0, slope))
        correction = jnp.linalg.solve(jet.hessian, gradient)
        length = jnp.linalg.norm(correction)
        correction *= jnp.minimum(1.0, pitch / jnp.where(length > 0, length, 1.0))
        return point - correction

    point = jax.lax.fori_loop(0, iterations, step, seed)
    jet = field.evaluate(point)
    residual = jnp.linalg.norm(jet.gradient + jnp.array((0.0, slope)))
    return point, residual
