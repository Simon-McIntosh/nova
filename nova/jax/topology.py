"""Extract plasma topology from flux map."""

from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

from nova.jax.null import Null1D, Null2D
from nova.jax.tree_util import Pytree


@dataclass
@jax.tree_util.register_pytree_node_class
class Topology(Pytree):
    """Manage plasma topology."""

    grid: Null2D
    wall: Null1D

    @jax.jit
    def x_point_index(self, vmap_x, polarity, o_psi):
        """Return index of primary x-point."""
        x_psi = vmap_x[:, 2]
        return jnp.nanargmax(polarity * (x_psi - o_psi))

    @jax.jit
    def x_point_data(self, vmap_x, polarity, o_psi):
        """Return primary x-point data."""
        index = self.x_point_index(vmap_x, polarity, o_psi)
        return vmap_x[index]

    @jax.jit
    def x_point(self, psi_grid, polarity):
        """Return primary x-point position."""
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        return self.x_point_data(vmap_x, polarity, data_o[2])[:2]

    @jax.jit
    def x_psi(self, psi_grid, polarity):
        """Return primary x-point flux."""
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        return self.x_point_data(vmap_x, polarity, data_o[2])[2]

    @jax.jit
    def o_point_index(self, vmap_o, polarity):
        """Return primary o-point index."""
        o_psi = vmap_o[:, 2]
        return jnp.nanargmax(polarity * o_psi)

    @jax.jit
    def o_point_data(self, vmap_o, polarity):
        """Return primary o-point data."""
        index = self.o_point_index(vmap_o, polarity)
        return vmap_o[index]

    @jax.jit
    def o_point(self, psi_grid, polarity):
        """Return primary o-point position."""
        vmap_o = self.grid(psi_grid)[0]
        return self.o_point_data(vmap_o, polarity)[:2]

    @jax.jit
    def o_psi(self, psi_grid, polarity):
        """Return primary o-point flux."""
        vmap_o = self.grid(psi_grid)[0]
        return self.o_point_data(vmap_o, polarity)[2]

    @jax.jit
    def w_point(self, psi_wall, polarity):
        """Return w_point position."""
        return self.wall(psi_wall, polarity)[:2]

    @jax.jit
    def w_psi(self, psi_wall, polarity):
        """Return wall-point flux."""
        return self.wall(psi_wall, polarity)[2]

    @jax.jit
    def boundary(self, data_o, vmap_x, data_w, polarity):
        """Return boundary data structure."""
        # x-point vertical bounds
        x_heights = vmap_x[:, 1]
        x_height_min = jnp.nanmin(x_heights)
        x_height_max = jnp.nanmax(x_heights)
        # select grid x-point
        data_x = self.x_point_data(vmap_x, polarity, data_o[2])
        # o-point and w-point heights
        o_height = data_o[1]
        w_height = data_w[1]
        # adjust x-point bounds
        x_height_min = jnp.where(x_height_min > o_height, -jnp.inf, x_height_min)
        x_height_min = jnp.where(x_height_max > o_height, jnp.inf, x_height_max)
        # asses plasma operational mode
        mode_index = jax.lax.cond(
            polarity < 0,
            jnp.nanargmin,
            jnp.nanargmax,
            jnp.r_[data_x[2], data_w[2]],
        )
        return jnp.where(
            (w_height < x_height_min) | (w_height > x_height_max),
            data_x,
            jnp.c_[data_x, data_w][:, mode_index],
        )

    @jax.jit
    def psi_mask(self, polarity, psi_grid, psi_boundary):
        """Return plasma filament psi-mask."""
        return jax.lax.cond(
            polarity > 0, jnp.greater_equal, jnp.less, psi_grid, psi_boundary
        )

    @jax.jit
    def x_mask(self, data_o, vmap_x):
        """Return plasma filament x-point mask."""
        mask = jnp.ones(self.grid.node_number, dtype=bool)

        @jax.jit
        def update_mask(mask, data_x):
            mask = jax.lax.select(
                mask & jnp.isfinite(data_x[0]),
                jax.lax.cond(
                    data_x[1] < data_o[1],
                    jnp.greater,
                    jnp.less,
                    self.grid.coordinate[:, 1],
                    data_x[1],
                ),
                mask,
            )
            return mask, None

        return jax.lax.scan(update_mask, mask, vmap_x)[0]

    @partial(jax.jit, static_argnums=3)
    def psi_lcfs(self, psi_axis, psi_boundary, psi_norm=0.999):
        """Return poloidal flux at last closed flux surface."""
        return psi_norm * (psi_boundary - psi_axis) + psi_axis

    @jax.jit
    def normalize(self, psi_axis, psi_boundary, psi_grid):
        """Return normalized flux."""
        return (psi_grid - psi_axis) / (psi_boundary - psi_axis)

    @jax.jit
    def ionize(self, data_o, vmap_x, polarity, psi_grid, psi_lcfs):
        """Return ionization mask."""
        return self.x_mask(data_o, vmap_x) & self.psi_mask(polarity, psi_grid, psi_lcfs)

    @jax.jit
    def update(self, psi, polarity):
        """Return normalized poloidal flux and ionization mask."""
        # split poloidal flux map into grid and wall zones
        psi_grid = jax.lax.dynamic_slice_in_dim(psi, 0, self.grid.node_number)
        psi_wall = jax.lax.dynamic_slice_in_dim(
            psi, self.grid.node_number, self.wall.node_number
        )
        # calculate flux map topology
        vmap_o, vmap_x = self.grid(psi_grid)
        data_o = self.o_point_data(vmap_o, polarity)
        data_w = self.wall(psi_wall, polarity)
        data_b = self.boundary(data_o, vmap_x, data_w, polarity)
        # normalize psi grid."""
        psi_norm = self.normalize(data_o[2], data_b[2], psi_grid)
        psi_lcfs = self.psi_lcfs(data_o[2], data_b[2])
        ionize = self.ionize(data_o, vmap_x, polarity, psi_grid, psi_lcfs)
        return psi_norm, ionize

    def tree_flatten(self):
        """Return flattened pytree."""
        children = (self.grid, self.wall)
        aux_data = {}
        return (children, aux_data)
