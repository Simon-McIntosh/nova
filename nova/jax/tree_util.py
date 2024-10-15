"""Manage pytree base class."""

import abc

import jax


@jax.tree_util.register_pytree_node_class
class Pytree(abc.ABC):
    """Jax pytree base class."""

    @abc.abstractmethod
    def tree_flatten(self) -> (tuple, dict):
        """Return flattened pytree."""
        return (), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Return unflattened pytree."""
        return cls(*children, **aux_data)
