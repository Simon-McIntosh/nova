from dataclasses import dataclass, field

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np
import scipy


@jax.tree_util.register_pytree_node_class
@dataclass
class Zeta:
    """Evaluate zeta function."""

    rs: np.ndarray | jnp.ndarray = field(repr=False)
    zs: np.ndarray | jnp.ndarray = field(repr=False)
    r: np.ndarray | jnp.ndarray = field(repr=False)
    z: np.ndarray | jnp.ndarray = field(repr=False)
    alpha: np.ndarray | jnp.ndarray = field(repr=False)
    number: int = 150

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

    def __call__(self):
        """Return zeta intergral."""
        return scipy.integrate.quad_vec(self.arcsinh_beta_1, 0, self.alpha)

        # alpha = jnp.linspace(0, self.alpha, self.number)
        # return jax.scipy.integrate.trapezoid(
        # self.arcsinh_beta_1(alpha), alpha, axis=0)


if __name__ == "__main__":

    rs = np.linspace(1, 3)
    zs = np.linspace(4, 7)
    r = np.linspace(1, 3)
    z = np.linspace(4, 7)
    alpha = np.linspace(1, 3)

    zeta = Zeta(rs, zs, r, z, alpha, 12)

    print(zeta())
