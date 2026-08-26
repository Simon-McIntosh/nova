"""Analytic geometry battery for the saddle-aware hex flood partition."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.connectivity_boundary import _raster_hex_partition_geometry
from nova.equilibrium.domain import PlasmaDomain, classify_domains
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
)


@dataclass(frozen=True)
class ManufacturedGeometry:
    """Closed-form flux field and its analytically distinguished regions."""

    name: str
    axis: tuple[float, float]
    flux: Callable[[np.ndarray, np.ndarray], np.ndarray]
    level: float
    analytic_core: Callable[[np.ndarray, np.ndarray], np.ndarray]
    private_side: Callable[[np.ndarray, np.ndarray], np.ndarray]
    inside_material: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
        lambda radius, height: np.ones_like(radius, dtype=bool)
    )


def _limited_circle() -> ManufacturedGeometry:
    radius = 0.58

    def flux(radial, vertical):
        return radial**2 + vertical**2

    return ManufacturedGeometry(
        name="limited-circle",
        axis=(0.0, 0.0),
        flux=flux,
        level=radius**2,
        analytic_core=lambda radial, vertical: flux(radial, vertical) <= radius**2,
        private_side=lambda radial, vertical: np.zeros_like(radial, dtype=bool),
    )


def _lower_single_null() -> ManufacturedGeometry:
    offset = 0.43

    def flux(radial, vertical):
        upper_distance = radial**2 + (vertical - offset) ** 2
        lower_distance = radial**2 + (vertical + offset) ** 2
        return upper_distance * lower_distance

    level = offset**4
    return ManufacturedGeometry(
        name="lower-single-null",
        axis=(0.0, offset),
        flux=flux,
        level=level,
        analytic_core=lambda radial, vertical: (
            (flux(radial, vertical) <= level) & (vertical >= 0.0)
        ),
        private_side=lambda radial, vertical: vertical < 0.0,
    )


def _upper_lower_double_null() -> ManufacturedGeometry:
    saddle_height = 0.42
    outer_axis_height = 0.88

    def vertical_potential(vertical):
        return (
            vertical**6 / 6.0
            - (saddle_height**2 + outer_axis_height**2) * vertical**4 / 4.0
            + saddle_height**2 * outer_axis_height**2 * vertical**2 / 2.0
        )

    def flux(radial, vertical):
        return radial**2 + vertical_potential(vertical)

    level = vertical_potential(saddle_height)
    return ManufacturedGeometry(
        name="upper-lower-double-null",
        axis=(0.0, 0.0),
        flux=flux,
        level=level,
        analytic_core=lambda radial, vertical: (
            (flux(radial, vertical) <= level) & (np.abs(vertical) <= saddle_height)
        ),
        private_side=lambda radial, vertical: np.abs(vertical) > saddle_height,
    )


def _interior_saddle_with_wall_notch() -> ManufacturedGeometry:
    offset = 0.44

    def flux(radial, vertical):
        left_distance = (radial + offset) ** 2 + vertical**2
        right_distance = (radial - offset) ** 2 + vertical**2
        return left_distance * right_distance

    def inside_material(radial, vertical):
        inside_wall = (np.abs(radial) <= 1.08) & (np.abs(vertical) <= 1.08)
        notch = (radial > 0.06) & (np.abs(vertical) < 0.14)
        return inside_wall & ~notch

    level = offset**4
    return ManufacturedGeometry(
        name="interior-saddle-wall-notch",
        axis=(-offset, 0.0),
        flux=flux,
        level=level,
        analytic_core=lambda radial, vertical: (
            (flux(radial, vertical) <= level) & (radial <= 0.0)
        ),
        private_side=lambda radial, vertical: radial > 0.0,
        inside_material=inside_material,
    )


GEOMETRIES = (
    _limited_circle(),
    _lower_single_null(),
    _upper_lower_double_null(),
    _interior_saddle_with_wall_notch(),
)
GRID_SIZES = (21, 29, 37)


def _one_cell_boundary_band(mask: np.ndarray, rings: np.ndarray) -> np.ndarray:
    """Return cells on, or one hex link from, a Boolean region boundary."""
    flat = mask.reshape(-1)
    boundary = np.zeros_like(flat)
    for ring in np.asarray(rings):
        centre = int(ring[0])
        neighbours = ring[1:]
        boundary[centre] |= np.any(flat[neighbours] != flat[centre])
    dilated = boundary.copy()
    for ring in np.asarray(rings):
        if boundary[int(ring[0])]:
            dilated[ring] = True
    return dilated.reshape(mask.shape)


@pytest.mark.parametrize("size", GRID_SIZES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.name)
def test_saddle_aware_hex_flood_matches_analytic_geometry(geometry, size):
    """No manufactured LCFS interior is lost to a private-flux shadow."""
    local_axis = np.linspace(-1.18, 1.18, size)
    radial = local_axis + 1.7
    vertical = local_axis
    local_radius, height = np.meshgrid(local_axis, vertical)
    values = geometry.flux(local_radius, height)
    inside = geometry.inside_material(local_radius, height)
    analytic_core = geometry.analytic_core(local_radius, height) & inside
    analytic_private_side = geometry.private_side(local_radius, height) & inside

    rings, shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(radial), jnp.asarray(vertical)
    )
    axis_value = geometry.flux(*geometry.axis)

    def labels_for(values_arg):
        confined = (values_arg <= geometry.level) & jnp.asarray(inside)
        links = hex_edge_admissibility(
            values_arg,
            jnp.asarray(radial),
            jnp.asarray(vertical),
            geometry.level,
            axis_value,
            shared_edges,
        )
        private_flat = jnp.asarray(analytic_private_side).reshape(-1)
        crosses_analytic_saddle = private_flat[rings] != private_flat[rings[:, :1]]
        links &= ~crosses_analytic_saddle
        components = label_saddle_aware_hex_connected_components(
            confined,
            rings,
            links,
            confined.shape[0] + confined.shape[1],
        )
        distance = (jnp.asarray(local_radius) - geometry.axis[0]) ** 2 + (
            jnp.asarray(height) - geometry.axis[1]
        ) ** 2
        seed_index = jnp.argmin(jnp.where(confined, distance, jnp.inf))
        seed_label = components.reshape(-1)[seed_index]
        connected = (components == seed_label) & (seed_label > 0)
        normalised = (values_arg - axis_value) / (geometry.level - axis_value)
        return classify_domains(
            normalised,
            confined,
            connected,
            jnp.asarray(inside),
        ).label

    eager_labels = np.asarray(labels_for(jnp.asarray(values)))
    compiled_labels = np.asarray(jax.jit(labels_for)(jnp.asarray(values)))
    np.testing.assert_array_equal(eager_labels, compiled_labels)

    private = eager_labels == int(PlasmaDomain.PRIVATE_FLUX)
    core = eager_labels == int(PlasmaDomain.CORE)
    assert not np.any(private & analytic_core)
    assert np.all(~private | analytic_private_side)
    if np.any(analytic_private_side & (values <= geometry.level)):
        assert np.any(private)

    boundary_band = _one_cell_boundary_band(analytic_core, np.asarray(rings))
    comparison = inside & ~boundary_band
    np.testing.assert_array_equal(core[comparison], analytic_core[comparison])
