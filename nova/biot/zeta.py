from dataclasses import dataclass, field

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass
class Zeta:
    """Evaluate zeta function."""

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
        """Return gamma coefficent."""
        return self.zs - self.z

    @jax.jit
    def G2(self, phi):
        """Return G2 coefficent."""
        return self.gamma**2 + self.r**2 * jnp.sin(phi) ** 2

    @jax.jit
    def arcsinh_beta_1(self, alpha):
        """Return zeta intergrand."""
        phi = jnp.pi - 2 * alpha
        return jnp.arcsinh((self.rs - self.r * jnp.cos(phi)) / jnp.sqrt(self.G2(phi)))

    # @jax.jit
    def __call__(self):
        """Return zeta intergral."""
        alpha = jnp.linspace(1e-8, self.alpha, self.number)
        return jax.scipy.integrate.trapezoid(self.arcsinh_beta_1(alpha), alpha, axis=0)


if __name__ == "__main__":

    data = np.ones((3, 4))

    rs = data
    zs = data
    r = data
    z = data
    alpha = data

    zeta = Zeta(rs, zs, r, z, alpha, 12)

    print(zeta().dtype)
