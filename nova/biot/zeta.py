"""Evaluate the zeta integral for finite-volume Biot methods.

``zeta`` is the one non-closed-form piece of the rectangular/bow conductor
antiderivative: the midpoint quadrature of an arcsinh integrand over the arc
half-angle.  The canonical implementation is pure numpy (imports with numpy
only, so the kernel layer carries no numba/JAX requirement).  A JAX pytree
form, :class:`Zeta`, is provided for the jitted batched-evaluation path and is
only usable when JAX is installed; the numpy ``zeta`` and the JAX ``Zeta``
evaluate the same integrand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


def _arcsinh_beta_1(
    rs: np.ndarray, r: np.ndarray, gamma: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Return the zeta integrand at arc half-angle ``alpha`` (phi = pi - 2 alpha)."""
    phi = np.pi - 2.0 * alpha
    g2 = gamma**2 + r**2 * np.sin(phi) ** 2
    return np.arcsinh((rs - r * np.cos(phi)) / np.sqrt(g2))


def zeta(
    rs: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    number: int = 500,
) -> np.ndarray:
    """Evaluate the zeta integral element-wise via midpoint quadrature.

    Each element integrates the arcsinh integrand over ``[0, alpha_i]`` with a
    per-element panel count scaled to its own limit (``~ |alpha_i| * number``),
    so the resolution follows the arc extent and the working set stays one
    element wide -- important for the large target x source arrays a grid solve
    produces.  Inputs broadcast to a common shape; the result takes the shape
    of ``alpha``.
    """
    shape = np.shape(alpha)
    rs = np.ravel(rs)
    r = np.ravel(r)
    gamma = np.ravel(gamma)
    alpha = np.ravel(alpha)
    result = np.zeros(len(alpha))
    for i in range(len(alpha)):
        if np.isclose(alpha[i], 0.0):
            continue
        num = max(3, int(abs(alpha[i]) * number))
        dalpha = alpha[i] / (num - 1)
        nodes = np.linspace(0.0, alpha[i], num)[:-1] + dalpha / 2.0  # midpoints
        result[i] = abs(dalpha) * np.sum(_arcsinh_beta_1(rs[i], r[i], gamma[i], nodes))
    return result.reshape(shape)


try:  # optional: only importable where JAX is installed
    import jax
    import jax.numpy as jnp

    @jax.tree_util.register_pytree_node_class
    @dataclass
    class Zeta:
        """Evaluate the zeta integral through JAX (jittable, vmap-safe)."""

        rs: np.ndarray | jnp.ndarray = field(repr=False)
        zs: np.ndarray | jnp.ndarray = field(repr=False)
        r: np.ndarray | jnp.ndarray = field(repr=False)
        z: np.ndarray | jnp.ndarray = field(repr=False)
        alpha: np.ndarray | jnp.ndarray = field(repr=False)
        number: int = 300

        def tree_flatten(self):
            """Return flattened pytree structure."""
            children = (self.rs, self.zs, self.r, self.z, self.alpha)
            aux_data = self.number
            return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            """Rebuild instance from pytree variables."""
            return cls(*children, aux_data)

        @cached_property
        @jax.jit
        def gamma(self):
            """Return gamma coefficient."""
            return self.zs - self.z

        @jax.jit
        def g2(self, phi):
            """Return G2 coefficient."""
            return self.gamma**2 + self.r**2 * jnp.sin(phi) ** 2

        @jax.jit
        def arcsinh_beta_1(self, alpha):
            """Return the zeta integrand."""
            phi = jnp.pi - 2 * alpha
            return jnp.arcsinh(
                (self.rs - self.r * jnp.cos(phi)) / jnp.sqrt(self.g2(phi))
            )

        @jax.jit
        def __call__(self):
            """Return the zeta integral."""
            alpha = jnp.linspace(1e-8, self.alpha, self.number)
            return jax.scipy.integrate.trapezoid(
                self.arcsinh_beta_1(alpha), alpha, axis=0
            )

except ModuleNotFoundError:  # numpy-only environment

    class Zeta:  # type: ignore[no-redef]
        """Placeholder when JAX is unavailable; use the numpy ``zeta`` instead."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "nova.biot.zeta.Zeta requires JAX; install the 'jax' extra or "
                "use the numpy zeta() function"
            )
