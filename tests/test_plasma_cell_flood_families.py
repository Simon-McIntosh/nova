"""Manufactured saddle partitions evaluated on physical hex-cell edges."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.cell_partition import (
    cell_partition_geometry,
    missing_link_mask,
)
from nova.equilibrium.domain import PlasmaDomain, classify_domains
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
)
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.geometry.hexstencil import hex_stencil
from tests.test_hex_flood_coil_nulls import STACK
from tests.test_hex_flood_geometries import GEOMETRIES as BASE_GEOMETRIES
from tests.test_hex_flood_sn_secondary import GEOMETRIES as SECONDARY_GEOMETRIES
from tests.test_hex_flood_snowflake import GEOMETRIES as SNOWFLAKE_GEOMETRIES


@dataclass(frozen=True)
class Carrier:
    coordinate: np.ndarray
    stencil: np.ndarray
    polygons: tuple[np.ndarray, ...]
    area: np.ndarray
    rings: np.ndarray
    shared_edges: np.ndarray


def _carrier(shape=(21, 35), pitch=0.08, centre=(0.0, 0.0)) -> Carrier:
    column, row = np.indices(shape)
    coordinate = np.c_[
        pitch * (column.ravel() + 0.5 * row.ravel()),
        pitch * np.sqrt(3.0) / 2.0 * row.ravel(),
    ]
    coordinate -= np.mean(coordinate, axis=0)
    coordinate += np.asarray(centre)
    angle = np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    offset = pitch / np.sqrt(3.0) * np.c_[np.cos(angle), np.sin(angle)]
    polygons = tuple(point + offset for point in coordinate)
    stencil = hex_stencil(shape)
    rings, edges = cell_partition_geometry(coordinate, stencil, polygons)
    area = np.full(len(coordinate), np.sqrt(3.0) * pitch**2 / 2.0, dtype=np.float64)
    return Carrier(
        coordinate,
        stencil,
        polygons,
        area,
        np.asarray(rings),
        np.asarray(edges),
    )


def _edge_samples(carrier: Carrier, flux: Callable) -> np.ndarray:
    parameter = np.asarray((0.0, 0.5, 1.0))
    points = carrier.shared_edges[..., :1, :] + parameter[None, None, :, None] * (
        carrier.shared_edges[..., 1:, :] - carrier.shared_edges[..., :1, :]
    )
    return np.asarray(flux(points[..., 0], points[..., 1]))


def _partition(
    carrier: Carrier,
    flux: Callable,
    level: float,
    axis: tuple[float, float],
    inside: np.ndarray,
    private_side: np.ndarray | None = None,
    preserve_confined_side: bool = False,
) -> np.ndarray:
    radius, height = carrier.coordinate.T
    values = np.asarray(flux(radius, height))
    axis_value = float(flux(*axis))
    confined = (values >= level if axis_value >= level else values <= level) & inside
    links = np.asarray(
        hex_edge_admissibility(
            jnp.asarray(values),
            jnp.asarray(radius),
            jnp.asarray(height),
            level,
            axis_value,
            jnp.asarray(carrier.shared_edges),
            edge_values=jnp.asarray(_edge_samples(carrier, flux)),
        )
    ).copy()
    links &= ~missing_link_mask(carrier.rings)
    if private_side is not None:
        private = np.asarray(private_side, dtype=bool)
        crosses = private[carrier.rings] != private[carrier.rings[:, :1]]
        if preserve_confined_side:
            ring_confined = confined[carrier.rings]
            links |= ring_confined & ring_confined[:, :1] & ~crosses
        links &= ~crosses
    components = np.asarray(
        label_saddle_aware_hex_connected_components(
            jnp.asarray(confined),
            jnp.asarray(carrier.rings),
            jnp.asarray(links),
            confined.size,
        )
    )
    distance = (radius - axis[0]) ** 2 + (height - axis[1]) ** 2
    seed = int(np.argmin(np.where(confined, distance, np.inf)))
    seed_label = components[seed]
    connected = (components == seed_label) & (seed_label > 0)
    normalised = (values - axis_value) / (level - axis_value)
    return np.asarray(
        classify_domains(
            jnp.asarray(normalised),
            jnp.asarray(confined),
            jnp.asarray(connected),
            jnp.asarray(inside),
        ).label
    )


def _boundary_band(mask: np.ndarray, rings: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    for ring in rings:
        centre = int(ring[0])
        boundary[centre] = np.any(mask[ring[1:]] != mask[centre])
    band = boundary.copy()
    for ring in rings:
        if boundary[int(ring[0])]:
            band[ring] = True
    return band


@pytest.mark.parametrize(
    "geometry", BASE_GEOMETRIES, ids=lambda geometry: geometry.name
)
def test_base_geometries_match_their_analytic_hex_partition(geometry):
    """Match the analytic regions outside the raster-authority boundary band.

    The double-null carrier has exactly two discrepant centroids, at
    ``(0, ±0.41569219381653)`` m. Both are one pitch from its analytic
    separatrix and none lie outside the one-ring band tolerated by the raster
    hex-flood tests.
    """
    carrier = _carrier()
    radius, height = carrier.coordinate.T
    inside = geometry.inside_material(radius, height)
    analytic_core = geometry.analytic_core(radius, height) & inside
    private_side = geometry.private_side(radius, height) & inside
    labels = _partition(
        carrier,
        geometry.flux,
        geometry.level,
        geometry.axis,
        inside,
        private_side,
    )

    core = labels == int(PlasmaDomain.CORE)
    private = labels == int(PlasmaDomain.PRIVATE_FLUX)
    boundary_band = _boundary_band(analytic_core, carrier.rings)
    mismatch = core != analytic_core
    comparison = inside & ~boundary_band
    np.testing.assert_array_equal(core[comparison], analytic_core[comparison])
    assert not np.any(mismatch & ~boundary_band)
    if geometry.name == "upper-lower-double-null":
        assert np.count_nonzero(mismatch) == 2
    assert not np.any(private & analytic_core & ~boundary_band)
    assert np.all(~private | private_side | boundary_band)


@pytest.mark.parametrize(
    "geometry", SECONDARY_GEOMETRIES, ids=lambda geometry: geometry.name
)
def test_secondary_saddles_remain_in_the_private_hex_partition(geometry):
    carrier = _carrier(shape=(19, 39), pitch=0.07)
    radius, height = carrier.coordinate.T
    level = float(geometry.flux(0.0, geometry.primary_height))
    directed = geometry.direction * height
    private_side = directed > abs(geometry.primary_height)
    labels = _partition(
        carrier,
        geometry.flux,
        level,
        (0.0, 0.0),
        np.ones(len(radius), dtype=bool),
        private_side,
    )
    values = geometry.flux(radius, height)
    core = (values <= level) & ~private_side
    primary_shadow = (values <= level) & ((directed > 0.35) & (directed < 0.70))
    secondary_shadow = (values <= level) & (directed > 0.70)
    private = labels == int(PlasmaDomain.PRIVATE_FLUX)

    assert not np.any(private & core)
    assert np.any(private & primary_shadow)
    assert np.any(private & secondary_shadow)


@pytest.mark.parametrize(
    "geometry", SNOWFLAKE_GEOMETRIES, ids=lambda geometry: geometry.name
)
def test_paired_nulls_match_their_analytic_hex_partition(geometry):
    carrier = _carrier(shape=(19, 27), pitch=0.035)
    radius, height = carrier.coordinate.T
    values = geometry.flux(radius, height)
    primary = geometry.nulls[0]
    analytic_private = (values <= 1.0) & (radius > primary)
    labels = _partition(
        carrier,
        geometry.flux,
        1.0,
        geometry.axis,
        np.ones(len(radius), dtype=bool),
        analytic_private,
        preserve_confined_side=True,
    )
    private = labels == int(PlasmaDomain.PRIVATE_FLUX)
    analytic_core = (values <= 1.0) & (radius <= primary)

    assert not np.any(private & analytic_core)
    assert np.all(~private | analytic_private)
    for well in geometry.roots[1:]:
        distance = (radius - well) ** 2 + height**2
        index = int(np.argmin(np.where(values <= 1.0, distance, np.inf)))
        assert labels[index] == int(PlasmaDomain.PRIVATE_FLUX)


def _zero_profile(psi_norm):
    return jnp.zeros_like(psi_norm)


def _coil_partition(carrier: Carrier, include_coils: bool) -> np.ndarray:
    def flux(radius, height):
        shape = np.broadcast_shapes(np.shape(radius), np.shape(height))
        coordinate = np.column_stack(
            (
                np.broadcast_to(radius, shape).ravel(),
                np.broadcast_to(height, shape).ravel(),
            )
        )
        return STACK.flux(coordinate, include_coils=include_coils).reshape(shape)

    wall_angle = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)
    wall_coordinate = np.c_[
        1.0 + 0.46 * np.cos(wall_angle),
        0.82 * np.sin(wall_angle),
    ]
    values = flux(carrier.coordinate[:, 0], carrier.coordinate[:, 1])
    wall_values = flux(wall_coordinate[:, 0], wall_coordinate[:, 1])
    moment_geometry = MomentGeometry.from_cells(
        StencilMesh(carrier.coordinate, carrier.stencil, carrier.area),
        carrier.polygons,
    )
    inside = ((carrier.coordinate[:, 0] - 1.0) / 0.44) ** 2 + (
        carrier.coordinate[:, 1] / 0.78
    ) ** 2 <= 1.0
    operator = ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((len(values), 1)),
            plasma_target=jnp.zeros((len(values), 1)),
            null=Null2D.from_coordinates(
                carrier.coordinate, carrier.stencil, maxsize=12
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall_values), 1)),
            plasma_target=jnp.zeros((len(wall_values), 1)),
            null=Null1D(jnp.asarray(wall_coordinate)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(carrier.area),
        polarity=1,
        inside_material=jnp.asarray(inside),
        moment_geometry=moment_geometry,
        use_linear_moments=False,
    )
    np.testing.assert_array_equal(
        np.asarray(operator._fixed_design_topology.connectivity_rings), carrier.rings
    )
    np.testing.assert_allclose(
        np.asarray(operator._fixed_design_topology.connectivity_shared_edges),
        carrier.shared_edges,
        rtol=0.0,
        atol=1.0e-14,
    )
    physical = jnp.asarray(np.r_[values, wall_values])
    labels = np.asarray(operator.current_domain_masks(physical).label)
    return labels


def test_external_coil_nulls_do_not_capture_the_hex_carrier_partition():
    carrier = _carrier(shape=(21, 33), pitch=0.07, centre=(1.3, 0.0))
    plasma_labels = _coil_partition(carrier, include_coils=False)
    combined_labels = _coil_partition(carrier, include_coils=True)
    plasma_boundary = _boundary_band(plasma_labels, carrier.rings)
    combined_boundary = _boundary_band(combined_labels, carrier.rings)
    inside = ((carrier.coordinate[:, 0] - 1.0) / 0.44) ** 2 + (
        carrier.coordinate[:, 1] / 0.78
    ) ** 2 <= 1.0
    comparison = inside & ~plasma_boundary & ~combined_boundary

    np.testing.assert_array_equal(
        combined_labels[comparison], plasma_labels[comparison]
    )
    stable_core = (plasma_labels == int(PlasmaDomain.CORE)) & ~plasma_boundary & inside
    coil_shadow = (combined_labels == int(PlasmaDomain.PRIVATE_FLUX)) & stable_core
    assert not np.any(coil_shadow)
