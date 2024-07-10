"""Evaluate zeta intergral for finite volume biot methods."""

from dataclasses import dataclass, field

from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from nova import njit, prange


@njit(cache=True, fastmath=True, nogil=True)
def arcsinh_beta_1(rs, r, gamma, alpha):
    """Return zeta intergrand."""
    phi = np.pi - 2 * alpha
    G2 = gamma**2 + r**2 * np.sin(phi) ** 2
    return np.arcsinh((rs - r * np.cos(phi)) / np.sqrt(G2))


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def zeta(rs, r, gamma, alpha, number=500):
    """Evaluate zeta function."""
    shape = alpha.shape
    rs = np.ravel(rs)
    r = np.ravel(r)
    gamma = np.ravel(gamma)
    alpha = np.ravel(alpha)
    length = len(alpha)
    result = np.full(length, 0.0)
    for i in prange(length):
        if np.isclose(alpha[i], 0):
            continue
        num = np.max(np.array([3, int(abs(alpha[i]) * number)]))
        dalpha = alpha[i] / (num - 1)
        _alpha = np.linspace(0, alpha[i], num)[:-1] + dalpha / 2
        intergrand = arcsinh_beta_1(rs[i], r[i], gamma[i], _alpha)
        result[i] = abs(dalpha) * np.sum(intergrand)
        # _alpha[0] = dalpha / 2
        # result[i] = abs(dalpha) * intergrand[0] + np.trapz(intergrand[1:], _alpha[1:])
    return result.reshape(shape)


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

    gamma = zs - z
    zeta(rs, r, gamma, alpha)
    # zeta = Zeta(rs, zs, r, z, alpha, 12)

    # print(zeta().dtype)
