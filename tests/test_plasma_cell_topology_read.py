"""Parity checks for topology labels read on the cell carrier."""

import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _raster_hex_partition_geometry,
)
from nova.equilibrium.domain import classify_domains
from nova.equilibrium.flux_surface_connectivity import (
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
)
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _zero_profile(psi_norm):
    return jnp.zeros_like(psi_norm)


def _raster_operator(shape: tuple[int, int]) -> ForwardFluxOperator:
    radius = np.linspace(1.05, 2.35, shape[0])
    height = np.linspace(-0.72, 0.72, shape[1])
    lattice = FluxLattice(radius, height)
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    wall_coordinate = np.c_[
        1.7 + 0.64 * np.cos(wall_angle),
        0.68 * np.sin(wall_angle),
    ]
    node_count = lattice.node_count
    return ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.zeros((node_count, 1)),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((len(wall_coordinate), 1)),
            plasma_target=jnp.zeros((len(wall_coordinate), 1)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
    )


def _limited_flux(radius, height):
    return (radius - 1.7) ** 2 + height**2


def _single_null_flux(radius, height):
    local = radius - 1.7
    offset = 0.31
    return (local**2 + (height - offset) ** 2) * (local**2 + (height + offset) ** 2)


def _raster_label_oracle(operator, physical, state):
    """Build expected labels solely from the retained raster primitives."""
    radial = np.unique(np.asarray(operator.grid.coordinate)[:, 0])
    vertical = np.unique(np.asarray(operator.grid.coordinate)[:, 1])
    shape = (radial.size, vertical.size)
    flux = jnp.asarray(physical[: operator.grid.node_number]).reshape(shape).T
    inside = jnp.asarray(operator.inside_material).reshape(shape).T
    rings, shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(radial), jnp.asarray(vertical)
    )

    polarity = float(operator.polarity)
    closed = (
        flux >= state.boundary_flux if polarity > 0.0 else flux < state.boundary_flux
    )
    inside_flux = jnp.where(closed & inside, flux, jnp.nan)
    inward = _PRE_SADDLE_OFFSET_FRACTION * (
        jnp.nanmax(inside_flux) - jnp.nanmin(inside_flux)
    )
    direction = jnp.where(state.axis_flux >= state.boundary_flux, 1.0, -1.0)
    component_flux = state.boundary_flux + direction * inward
    component_flux = jnp.where(
        state.boundary_is_xpoint, component_flux, state.boundary_flux
    )
    confined = closed & inside
    exact_links = hex_edge_admissibility(
        flux,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        state.boundary_flux,
        state.axis_flux,
        shared_edges,
    )
    inward_links = hex_edge_admissibility(
        flux,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        component_flux,
        state.axis_flux,
        shared_edges,
    )
    coordinate = (
        jnp.asarray(operator.grid.coordinate)
        .reshape(shape + (2,))
        .transpose((1, 0, 2))
        .reshape((-1, 2))
    )
    centre = coordinate[rings[:, :1]]
    neighbour = coordinate[rings]
    edge_pitch = jnp.linalg.norm(neighbour - centre, axis=-1)
    edge_midpoint = jnp.mean(shared_edges, axis=-2)
    saddle_distance = jnp.linalg.norm(edge_midpoint - state.x_point, axis=-1)
    saddle_neighbourhood = state.boundary_is_xpoint & (
        saddle_distance <= 3.0 * edge_pitch
    )
    links = exact_links & (inward_links | ~saddle_neighbourhood)
    labels = label_saddle_aware_hex_connected_components(
        confined, rings, links, confined.size
    )
    radius_grid, height_grid = jnp.meshgrid(jnp.asarray(radial), jnp.asarray(vertical))
    distance2 = (radius_grid - state.axis[0]) ** 2 + (height_grid - state.axis[1]) ** 2
    seed_index = jnp.argmin(jnp.where(confined, distance2, jnp.inf))
    seed_label = labels.reshape(-1)[seed_index]
    connected = (labels == seed_label) & (seed_label > 0)
    psi_norm = (flux - state.axis_flux) / (state.boundary_flux - state.axis_flux)
    masks = classify_domains(psi_norm, closed, connected, inside)
    return masks.label.T.reshape(-1)


def test_raster_fixture_labels_are_bitwise_equal_to_independent_oracle():
    """Carrier-capable labels exactly match an independent raster primitive read."""
    configure_dtypes()
    compared = 0
    differing_count = 0
    for flux in (_limited_flux, _single_null_flux):
        operator = _raster_operator((17, 19))
        grid_coordinate = np.asarray(operator.grid.coordinate)
        wall_coordinate = np.asarray(operator.wall.coordinate)
        physical = jnp.asarray(
            np.r_[
                flux(grid_coordinate[:, 0], grid_coordinate[:, 1]),
                flux(wall_coordinate[:, 0], wall_coordinate[:, 1]),
            ]
        )

        actual, state, _connected, admitted = operator._fixed_design_read(physical)
        expected = _raster_label_oracle(operator, physical, state)

        assert bool(admitted)
        differing = np.flatnonzero(np.asarray(actual.label) != np.asarray(expected))
        compared += actual.label.size
        differing_count += differing.size
        assert differing.size == 0
        np.testing.assert_array_equal(actual.label, expected)
    print(f"raster_oracle_compared={compared} differing={differing_count}")
    assert compared == 646
    assert differing_count == 0


def test_carrier_shadow_operands_do_not_request_tensor_axes():
    """Wall shadowing reads carrier labels and nulls without a raster reshape."""
    configure_dtypes()
    operator = _raster_operator((13, 15))
    operator._fixed_design_topology.connectivity_radius = jnp.empty(0)
    operator._fixed_design_topology.connectivity_height = jnp.empty(0)
    coordinate = np.asarray(operator.grid.coordinate)
    wall = np.asarray(operator.wall.coordinate)
    state = jnp.asarray(
        np.r_[
            _limited_flux(coordinate[:, 0], coordinate[:, 1]),
            _limited_flux(wall[:, 0], wall[:, 1]),
        ]
    )

    flood_shadow, wall_shadow = operator.residual_shadow_components(state)
    qualification = operator._fixed_design_topology.read_qualification(
        state, operator.polarity, operator.inside_material
    )

    assert flood_shadow.shape == (operator.grid.node_number,)
    assert wall_shadow.shape == (operator.wall.node_number,)
    assert flood_shadow.dtype == jnp.bool_
    assert wall_shadow.dtype == jnp.bool_
    assert qualification.polish_receipt["fit_attempted"].tolist() == [True, True]
    assert qualification.polish_receipt["fit_iterations"].tolist() == [1, 1]
    assert np.all(np.isfinite(np.asarray(qualification.polish_receipt["fit_residual"])))
