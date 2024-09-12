"""Jax backed free-boundary equilibrium solver."""

from dataclasses import dataclass, field
from functools import cached_property

import jax
import jax.numpy as jnp
from scipy.constants import mu_0

from nova.jax.basis import Basis
from nova.jax.tree_util import Pytree
from nova.jax.target import Target
from nova.jax.topology import Topology


@dataclass
@jax.tree_util.register_pytree_node_class
class Plasma(Pytree):
    """Update plasma current."""

    grid: Target
    wall: Target
    p_prime: Basis
    ff_prime: Basis
    source_current: jnp.ndarray = field(repr=False)
    area: jnp.ndarray = field(repr=False)
    polarity: int

    def __post_init__(self):
        """Generate topology instance."""
        self.topology = Topology(self.grid.null, self.wall.null)

    @jax.jit
    def __call__(self, psi: jnp.ndarray):
        """Return total poloidal flux."""
        return self.external + self.internal(psi)

    @jax.jit
    def residual(self, psi):
        """Return poloidal flux residual."""
        return psi - self(psi)

    @jax.jit
    def plasma_current(self, psi):
        """Return plasma current calculated from poloidal flux."""
        psi_norm, ionize = self.topology.update(psi, self.polarity)
        current_density = self.grid.coordinate[:, 0] * self.p_prime(
            psi_norm
        ) + self.ff_prime(psi_norm) / (mu_0 * self.grid.coordinate[:, 0])
        current_density *= -2 * jnp.pi
        plasma_current = current_density * self.area
        return jnp.where(ionize, plasma_current, 0)

    @cached_property
    def external(self):
        """Return external flux map."""
        return jnp.r_[
            self.grid.external(self.source_current),
            self.wall.external(self.source_current),
        ]

    @jax.jit
    def internal(self, psi):
        """Return internal (plasma generated) flux map."""
        plasma_current = self.plasma_current(psi)
        return jnp.r_[
            self.grid.internal(plasma_current), self.wall.internal(plasma_current)
        ]

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (
            self.grid,
            self.wall,
            self.p_prime,
            self.ff_prime,
            self.source_current,
            self.area,
        )
        aux_data = {"polarity": self.polarity}
        return (children, aux_data)
