"""Containment regression coverage for the solve-path X-point read."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.topology import Topology
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


def _topology():
    configure_dtypes()
    radius = np.linspace(0.5, 1.5, 17)
    height = np.linspace(-0.5, 0.5, 17)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    grid = Null2D.from_coordinates(coordinate, hex_stencil((17, 17)), maxsize=3)
    wall = Null1D(
        jnp.asarray(
            [[0.6, -0.3], [1.4, -0.3], [1.4, 0.3], [0.6, 0.3]],
            dtype=jnp.float64,
        )
    )
    return Topology(grid, wall)


def test_read_qualification_prefers_the_contained_saddle_before_flux_ranking():
    """An external higher-flux saddle cannot select the solve-path X-point."""
    topology = _topology()
    candidates = jnp.asarray(
        [
            [1.0, -0.1, 0.5],
            [1.0, -0.4, 0.9],
            [jnp.nan, jnp.nan, jnp.nan],
        ],
        dtype=jnp.float64,
    )
    axis = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float64)
    wall = jnp.asarray([1.0, -0.2, 0.7], dtype=jnp.float64)

    contained = topology.contained_x_candidates(candidates)
    selected = topology.x_point_data(candidates, 1.0, axis[2])
    boundary = topology.boundary(axis, candidates, wall, 1.0)

    np.testing.assert_array_equal(np.asarray(contained), [True, False, False])
    np.testing.assert_allclose(np.asarray(selected), np.asarray(candidates[0]))
    np.testing.assert_allclose(np.asarray(boundary), np.asarray(candidates[0]))
