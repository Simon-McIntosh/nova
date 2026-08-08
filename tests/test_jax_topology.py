"""Batched-evaluation contract for the jax plasma-topology stack."""

import numpy as np
import pytest

from nova.geometry.hexstencil import hex_stencil
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.jax.null import Null1D, Null2D
    from nova.jax.topology import Topology


def _structured_grid(nx, nz, xlim=(0.5, 1.5), zlim=(-0.6, 0.6)):
    """Return flat coordinates, hexagonal stencil and stencil geometry."""
    x = np.linspace(*xlim, nx)
    z = np.linspace(*zlim, nz)
    x2d, z2d = np.meshgrid(x, z, indexing="ij")
    coordinate = np.c_[x2d.ravel(), z2d.ravel()]
    stencil = hex_stencil((nx, nz))
    return coordinate, stencil, coordinate[stencil]


def _flux_field(coordinate, xo=1.0, zo=0.0, xs=1.0, zs=-0.4, amp=1.0):
    """Return an analytic flux map with an o-point (max) and a saddle."""
    x, z = coordinate[:, 0], coordinate[:, 1]
    return -amp * ((x - xo) ** 2 + (z - zo) ** 2) + 0.3 * (
        (x - xs) ** 2 - (z - zs) ** 2
    )


@pytest.fixture(scope="module")
def topology():
    """Return a jax Topology on a synthetic structured grid + wall loop."""
    coordinate, stencil, coordinate_stencil = _structured_grid(40, 40)
    grid = Null2D(
        jnp.asarray(coordinate),
        jnp.asarray(stencil),
        jnp.asarray(coordinate_stencil),
        maxsize=5,
    )
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    wall_xy = np.c_[1.0 + 0.45 * np.cos(theta), 0.45 * np.sin(theta)]
    return Topology(grid, Null1D(jnp.asarray(wall_xy))), coordinate, wall_xy


def _flux_stack(coordinate, wall_xy, scales):
    """Return a (batch, node) flux map stack scaled per slice."""
    psi_grid = _flux_field(coordinate)
    psi_wall = _flux_field(wall_xy)
    psi = np.r_[psi_grid, psi_wall]
    return jnp.asarray(np.stack([psi * s for s in scales]))


def test_update_resolves_finite_topology(topology):
    """A single update returns a finite normalised map and a plasma region."""
    topo, coordinate, wall_xy = topology
    psi = _flux_stack(coordinate, wall_xy, [1.0])[0]
    psi_norm, ionize = topo.update(psi, 1)
    assert np.all(np.isfinite(np.asarray(psi_norm)))
    assert int(np.sum(np.asarray(ionize))) > 0


def test_batched_update_matches_per_slice(topology):
    """update_batch over a leading axis equals per-slice update calls."""
    topo, coordinate, wall_xy = topology
    scales = [1.0, 1.1, 0.9, 1.2, 0.75]
    psi_batch = _flux_stack(coordinate, wall_xy, scales)
    polarity = 1

    batch_norm, batch_ionize = topo.update_batch(psi_batch, polarity)
    assert batch_norm.shape[0] == len(scales)

    for i in range(len(scales)):
        norm_i, ionize_i = topo.update(psi_batch[i], polarity)
        assert np.allclose(
            np.asarray(batch_norm[i]), np.asarray(norm_i), equal_nan=True
        )
        assert np.array_equal(np.asarray(batch_ionize[i]), np.asarray(ionize_i))


def test_batched_primary_points_match_per_slice(topology):
    """Primary o-/x-point queries vmap to the same values as per-slice."""
    topo, coordinate, wall_xy = topology
    scales = [1.0, 1.15, 0.85]
    psi_batch = _flux_stack(coordinate, wall_xy, scales)
    polarity = 1
    psi_grid = jax.vmap(lambda p: topo.split_flux_map(p)[0])(psi_batch)

    batch_o = jax.vmap(topo.o_point, in_axes=(0, None))(psi_grid, polarity)
    for i in range(len(scales)):
        o_i = topo.o_point(psi_grid[i], polarity)
        assert np.allclose(np.asarray(batch_o[i]), np.asarray(o_i), equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
