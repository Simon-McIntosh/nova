"""Manage jax EM wall and grid targets."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from nova.jax.null import Null1D, Null2D
from nova.jax.tree_util import Pytree


@dataclass
@jax.tree_util.register_pytree_node_class
class Target(Pytree):
    """Manage EM coupling form external sources and plasma to target."""

    source_target: jnp.ndarray = field(repr=False)
    plasma_target: jnp.ndarray = field(repr=False)
    null: Null1D | Null2D
    source_plasma_index: int

    @property
    def coordinate(self):
        """Return target coordinate."""
        return self.null.coordinate

    @property
    def node_number(self):
        """Return target node number."""
        return self.null.node_number

    @jax.jit
    def external(self, source_current: jnp.ndarray):
        """Return external poloidal flux map."""
        source_current = source_current.at[self.source_plasma_index].set(0.0)
        return self.source_target @ source_current

    @jax.jit
    def internal(self, plasma_current: jnp.ndarray):
        """Return internal (plasma generated) poloidal flux map."""
        return self.plasma_target @ plasma_current

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.source_target, self.plasma_target, self.null)
        aux_data = {"source_plasma_index": self.source_plasma_index}
        return (children, aux_data)
