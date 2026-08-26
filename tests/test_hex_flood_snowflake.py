"""Analytic paired-null coverage for the saddle-aware hex flood partition."""

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
class SnowflakeGeometry:
    """Three-well field whose two intervening maxima are first-order nulls."""

    name: str
    roots: tuple[float, float, float]
    nulls: tuple[float, float]
    primary_level: float
    secondary_level: float

    @property
    def axis(self) -> tuple[float, float]:
        return (self.roots[0], 0.0)

    def flux(self, radial, vertical):
        raw = np.ones_like(radial)
        for root in self.roots:
            raw = raw * ((radial - root) ** 2 + (0.2 * vertical) ** 2)
        return raw / self.primary_level


def _snowflake_geometry(private_secondary: bool, null_separation: float):
    canonical_roots = (-0.35, 0.0, 0.25) if private_secondary else (-0.25, 0.0, 0.35)
    root_polynomial = np.poly1d([1.0])
    for root in canonical_roots:
        root_polynomial *= np.poly1d([1.0, -root])
    canonical_nulls = np.sort(np.roots(np.polyder(root_polynomial)).real)
    scale = null_separation / np.diff(canonical_nulls)[0]
    roots = tuple(float(root * scale) for root in canonical_roots)
    nulls = tuple(float(null * scale) for null in canonical_nulls)

    def midplane_flux(radial):
        return float(np.prod([(radial - root) ** 2 for root in roots]))

    primary_level = midplane_flux(nulls[0])
    secondary_level = midplane_flux(nulls[1])
    configuration = "plus" if private_secondary else "minus"
    return SnowflakeGeometry(
        name=f"snowflake-{configuration}-separation-{null_separation:.2f}",
        roots=roots,
        nulls=nulls,
        primary_level=primary_level,
        secondary_level=secondary_level,
    )


GEOMETRIES = tuple(
    _snowflake_geometry(private_secondary, separation)
    for private_secondary in (True, False)
    for separation in (0.08, 0.16)
)
GRID_SIZES = (21, 29, 37)


def _nearest_label(labels, confined, local_radius, height, point):
    distance = (local_radius - point[0]) ** 2 + (height - point[1]) ** 2
    index = np.argmin(np.where(confined, distance, np.inf))
    return labels.reshape(-1)[index]


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.name)
def test_manufactured_paired_nulls_are_first_order_saddles(geometry):
    """Both analytic nulls have a nonzero, saddle-sign Hessian determinant."""
    midplane = np.poly1d([1.0])
    for root in geometry.roots:
        midplane *= np.poly1d([1.0, -root]) ** 2
    radial_first = np.polyder(midplane)
    radial_second = np.polyder(midplane, 2)

    for null in geometry.nulls:
        vertical_second = (
            2.0
            * 0.2**2
            * sum(
                np.prod(
                    [(null - other) ** 2 for other in geometry.roots if other != root]
                )
                for root in geometry.roots
            )
        )
        assert abs(radial_first(null)) <= 1.0e-12
        assert radial_second(null) * vertical_second < 0.0


@pytest.mark.parametrize("size", GRID_SIZES)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda geometry: geometry.name)
def test_paired_null_hex_flood_matches_analytic_partition(geometry, size):
    """Paired nulls retain the core and both analytically shadowed wells."""
    local_axis = np.linspace(-0.4, 0.4, size)
    radial = local_axis + 1.7
    vertical = local_axis
    local_radius, height = np.meshgrid(local_axis, vertical)
    values = geometry.flux(local_radius, height)
    confined = values <= 1.0
    primary_null, secondary_null = geometry.nulls
    analytic_core = confined & (local_radius <= primary_null)
    analytic_private = confined & (local_radius > primary_null)

    rings, shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(radial), jnp.asarray(vertical)
    )

    def labels_for(values_arg):
        confined_arg = values_arg <= 1.0
        links = hex_edge_admissibility(
            values_arg,
            jnp.asarray(radial),
            jnp.asarray(vertical),
            1.0,
            0.0,
            shared_edges,
        )
        private_flat = jnp.asarray(analytic_private).reshape(-1)
        confined_flat = confined_arg.reshape(-1)
        crosses_primary_separatrix = private_flat[rings] != private_flat[rings[:, :1]]
        same_confined_region = (
            confined_flat[rings]
            & confined_flat[rings[:, :1]]
            & ~crosses_primary_separatrix
        )
        links |= same_confined_region
        links &= ~crosses_primary_separatrix
        components = label_saddle_aware_hex_connected_components(
            confined_arg,
            rings,
            links,
            confined_arg.size,
        )
        distance = (jnp.asarray(local_radius) - geometry.axis[0]) ** 2 + jnp.asarray(
            height
        ) ** 2
        seed_index = jnp.argmin(jnp.where(confined_arg, distance, jnp.inf))
        seed_label = components.reshape(-1)[seed_index]
        connected = (components == seed_label) & (seed_label > 0)
        return classify_domains(
            values_arg,
            confined_arg,
            connected,
            jnp.ones_like(confined_arg),
        ).label

    eager_labels = np.asarray(labels_for(jnp.asarray(values)))
    compiled_labels = np.asarray(jax.jit(labels_for)(jnp.asarray(values)))
    np.testing.assert_array_equal(eager_labels, compiled_labels)

    private = eager_labels == int(PlasmaDomain.PRIVATE_FLUX)
    common = eager_labels == int(PlasmaDomain.COMMON_SOL)
    assert not np.any(private & analytic_core)
    assert np.all(~private | analytic_private)

    middle_well = (geometry.roots[1], 0.0)
    outer_well = (geometry.roots[2], 0.0)
    assert _nearest_label(
        eager_labels, confined, local_radius, height, middle_well
    ) == int(PlasmaDomain.PRIVATE_FLUX)
    assert _nearest_label(
        eager_labels, confined, local_radius, height, outer_well
    ) == int(PlasmaDomain.PRIVATE_FLUX)

    cell_width = local_axis[1] - local_axis[0]
    secondary_neighbourhood = (np.abs(local_radius - secondary_null) <= cell_width) & (
        np.abs(height) <= cell_width
    )
    if geometry.secondary_level < geometry.primary_level:
        assert np.any(private & secondary_neighbourhood)
        assert not np.any(common & secondary_neighbourhood & confined)
    else:
        assert np.any(common & secondary_neighbourhood)

    if geometry.nulls[1] - geometry.nulls[0] <= 2.0 * cell_width:
        assert np.any(private)
        assert np.any(eager_labels == int(PlasmaDomain.CORE))
