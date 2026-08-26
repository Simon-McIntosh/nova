"""Single-null hex-flood geometries with a nearby secondary saddle."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.topology import Topology, TopologyClass
from nova.geometry.hexstencil import hex_stencil


@dataclass(frozen=True)
class SingleNullGeometry:
    """Analytic primary and secondary null arrangement on one vertical side."""

    name: str
    direction: float

    @property
    def primary_height(self) -> float:
        """Return the primary saddle height relative to the magnetic axis."""
        return self.direction * 0.35

    @property
    def secondary_height(self) -> float:
        """Return the secondary saddle height relative to the magnetic axis."""
        return self.direction * 0.70

    def potential(self, vertical):
        """Return a closed-form vertical potential with two nearby barriers."""
        directed = self.direction * vertical
        roots = (0.0, 0.35, 0.52, 0.70, 0.98)
        derivative = np.poly1d(roots, r=True)
        return 100.0 * np.polyint(derivative)(directed)

    def flux(self, radial, vertical):
        """Return the manufactured poloidal flux field."""
        return radial**2 + self.potential(vertical)


GEOMETRIES = (
    SingleNullGeometry("single-null-up", 1.0),
    SingleNullGeometry("single-null-down", -1.0),
)
GRID_SIZES = (33, 41, 49)


def _topology_and_flux(geometry: SingleNullGeometry, size: int):
    """Construct the production topology reader over one manufactured field."""
    local_radius = np.linspace(-1.25, 1.25, size)
    vertical = np.linspace(-1.25, 1.25, size)
    radius = local_radius + 1.7
    radius_grid, height_grid = np.meshgrid(radius, vertical, indexing="ij")
    local_radius_grid = radius_grid - 1.7
    coordinate = np.c_[radius_grid.ravel(), height_grid.ravel()]

    angle = 2.0 * np.pi * np.arange(64) / 64
    wall = np.c_[1.7 + 1.15 * np.cos(angle), 1.22 * np.sin(angle)]
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil((size, size)), maxsize=5),
        Null1D(jnp.asarray(wall)),
    )
    grid_flux = geometry.flux(local_radius_grid, height_grid)
    wall_flux = geometry.flux(wall[:, 0] - 1.7, wall[:, 1])
    psi = jnp.asarray(np.r_[grid_flux.ravel(), wall_flux])
    inside = jnp.ones(coordinate.shape[0], dtype=bool)
    return topology, psi, inside, local_radius_grid, height_grid


@pytest.mark.parametrize("size", GRID_SIZES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.name)
def test_single_null_secondary_shadows_remain_outside_lcfs(geometry, size):
    """Both null shadows stay private without cutting into the analytic core."""
    topology, psi, inside, local_radius, height = _topology_and_flux(geometry, size)
    primary_level = float(geometry.flux(0.0, geometry.primary_height))
    secondary_level = float(geometry.flux(0.0, geometry.secondary_height))
    assert 0.0 < secondary_level - primary_level < 0.01 * primary_level

    def read(field):
        return topology.read_with_connectivity(
            field,
            -1,
            inside,
            int(TopologyClass.DIVERTED),
        )

    eager_masks, eager_state, _eager_connected = read(psi)
    compiled_masks, compiled_state, _compiled_connected = jax.jit(read)(psi)
    eager_labels = np.asarray(eager_masks.label).reshape((size, size))
    compiled_labels = np.asarray(compiled_masks.label).reshape((size, size))
    np.testing.assert_array_equal(eager_labels, compiled_labels)

    spacing = 2.5 / (size - 1)
    expected_primary = np.array([1.7, geometry.primary_height])
    expected_secondary = np.array([1.7, geometry.secondary_height])
    np.testing.assert_allclose(
        np.asarray(eager_state.x_point), expected_primary, rtol=0.0, atol=spacing
    )
    np.testing.assert_allclose(
        np.asarray(compiled_state.x_point), expected_primary, rtol=0.0, atol=spacing
    )
    grid_flux = topology.split_flux_map(psi)[0]
    _o_candidates, x_candidates = topology.grid(grid_flux)
    finite_x = np.asarray(x_candidates)[np.isfinite(np.asarray(x_candidates)[:, 0]), :2]
    assert np.min(np.linalg.norm(finite_x - expected_primary, axis=1)) <= spacing
    assert np.min(np.linalg.norm(finite_x - expected_secondary, axis=1)) <= spacing

    directed_height = geometry.direction * height
    closed = geometry.flux(local_radius, height) <= primary_level
    analytic_core = closed & (directed_height < 0.35)
    primary_shadow = closed & (directed_height > 0.35) & (directed_height < 0.70)
    secondary_shadow = closed & (directed_height > 0.70)
    private = eager_labels == int(PlasmaDomain.PRIVATE_FLUX)

    assert np.any(private & primary_shadow)
    assert np.any(private & secondary_shadow)
    assert np.all(~private | primary_shadow | secondary_shadow)
    assert not np.any(private & analytic_core)
