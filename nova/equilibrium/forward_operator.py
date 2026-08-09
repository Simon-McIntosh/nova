"""Jitted flux operator for the forward free-boundary equilibrium solve.

The eager Newton-Krylov path in :mod:`nova.equilibrium.forward` drives the
plasma component's write-then-read cycle; this module carries the same
free-boundary map as a fixed-shape JAX pytree, so the residual is
differentiable and a batch of slices can be solved under ``vmap``. It is kept
apart from :mod:`nova.equilibrium.forward` so importing the eager solver does
not require the optional jax extra.
"""

from dataclasses import dataclass, field
from functools import cached_property

import jax
import jax.numpy as jnp
from scipy.constants import mu_0

from nova.biot.target import FluxTarget
from nova.equilibrium.topology import Topology
from nova.jax.tree_util import Pytree
from nova.linalg.interpolant import Basis


@dataclass
@jax.tree_util.register_pytree_node_class
class ForwardFluxOperator(Pytree):
    """Map a trial poloidal flux to the flux its plasma current generates.

    The total flux on the concatenated grid and wall targets is the sum of the
    external contribution (all conductors except the plasma) and the internal
    contribution of the plasma filaments, whose current follows from the trial
    flux through the flux functions :math:`p'(\\psi_N)` and
    :math:`FF'(\\psi_N)`. The plasma current is rescaled to the prescribed net
    current, so the free-boundary fixed point conserves :math:`I_p`.

    A root of :meth:`residual` is a free-boundary equilibrium. All fluxes are
    total poloidal fluxes, :math:`\\Phi = 2 \\pi R A_\\phi` in Wb.
    """

    grid: FluxTarget
    wall: FluxTarget
    p_prime: Basis
    ff_prime: Basis
    current: jnp.ndarray = field(repr=False)
    area: jnp.ndarray = field(repr=False)
    plasma_index: int
    polarity: int

    def __post_init__(self):
        """Generate topology instance and split plasma from external current."""
        self.topology = Topology(self.grid.null, self.wall.null)
        self.net_plasma_current = self.current[self.plasma_index]
        self.external_current = self.current.at[self.plasma_index].set(0.0)

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
        plasma_current = jnp.where(ionize, current_density * self.area, 0)
        plasma_current *= self.net_plasma_current / jnp.sum(plasma_current)
        return plasma_current

    @cached_property
    def external(self):
        """Return external flux map."""
        return jnp.r_[
            self.grid.external(self.external_current),
            self.wall.external(self.external_current),
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
            self.current,
            self.area,
        )
        aux_data = {"plasma_index": self.plasma_index, "polarity": self.polarity}
        return (children, aux_data)
