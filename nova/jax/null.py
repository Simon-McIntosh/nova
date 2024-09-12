"""Group fieldnull categorization algorithms."""

from dataclasses import dataclass, field
import jax
import jax.numpy as jnp

from nova.jax import select
from nova.jax.tree_util import Pytree


@dataclass
@jax.tree_util.register_pytree_node_class
class NullBase(Pytree):
    """Null pytree base class."""

    coordinate: jnp.ndarray = field(repr=False)

    def __post_init__(self):
        """Calculate node number."""
        self.node_number = self.coordinate.shape[0]

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.coordinate,)
        return (children, {})


@dataclass
@jax.tree_util.register_pytree_node_class
class Null1D(NullBase):
    """Locate and label field nulls on 1D loop."""

    @jax.jit
    def __call__(self, psi, polarity):
        """Return subgrid interpolated field null."""
        return select.wall_flux(
            self.coordinate[:, 0], self.coordinate[:, 1], psi, polarity
        )


@dataclass
@jax.tree_util.register_pytree_node_class
class Null2D(NullBase):
    """Locate and label field nulls on structured and unstructured grids."""

    stencil: jnp.ndarray = field(repr=False)
    coordinate_stencil: jnp.ndarray = field(repr=False)
    maxsize: int = 5

    @jax.jit
    def __call__(self, psi):
        """Return subgrid interpolated field nulls."""
        psi_stencil = psi[self.stencil]
        number, cluster = self.categorize(psi_stencil)
        return jax.vmap(self.interpolate, (0, 0))(number, cluster)

    @staticmethod
    @jax.jit
    def zero_cross_count(number, patch_array):
        """Count the total number of sign changes when traversing vertex patch.

            - 0: minima / maxima point
            - 2: regular point
            - 4: saddle point

        From On detecting all saddle points in 2D images, A. Kuijper
        """
        o_point_number, x_point_number = number

        def zero_cross(carry, value):
            """Increment zero crossing counter and update state."""
            count, sign, center = carry
            _sign = value > center
            sign_change = _sign != sign
            count += sign_change
            sign = jnp.where(sign_change, _sign, sign)
            return (count, sign, center), None

        center = patch_array[0]
        sign = patch_array[-1] > center
        count = jax.lax.scan(zero_cross, (0, sign, center), patch_array[1:])[0][0]
        o_point_number += count == 0
        x_point_number += count == 4
        return (o_point_number, x_point_number), count

    @jax.jit
    def categorize(self, psi_stencil):
        """Categorize points in 1d hexagonal grid."""
        number, count = jax.lax.scan(self.zero_cross_count, (0, 0), psi_stencil)

        def cluster(_, null_type):
            index = jnp.where((count == null_type), size=self.maxsize)[0]
            return (
                _,
                jnp.c_[
                    self.coordinate_stencil[index], psi_stencil[index, :, jnp.newaxis]
                ],
            )

        return jnp.array(number), jax.lax.scan(cluster, (), jnp.array([0, 4]))[1]

    @jax.jit
    def interpolate(self, number, cluster):
        """Interpolate subnull from cluster data."""

        def subnull(carry, cluster):
            carry += 1
            return carry, jnp.where(
                carry <= number, select.subnull(cluster.T), jnp.nan * jnp.ones(4)
            )

        return jax.lax.scan(subnull, 0, cluster)[1]

    def tree_flatten(self):
        """Return flattened pytree."""
        children, aux_data = super().tree_flatten()
        children += (self.stencil, self.coordinate_stencil)
        aux_data |= {"maxsize": self.maxsize}
        return (children, aux_data)
