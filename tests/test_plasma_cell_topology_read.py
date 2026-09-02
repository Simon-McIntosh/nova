"""Parity checks for topology labels read on the cell carrier."""

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import ForwardFluxOperator, axis_cell_seed
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


@pytest.mark.parametrize(
    "flux",
    (_limited_flux, _single_null_flux),
    ids=("limited", "single-null"),
)
def test_raster_fixture_labels_are_bitwise_equal_to_retained_read(flux):
    """Every fixture has zero cell-label differences from the raster reader."""
    configure_dtypes()
    operator = _raster_operator((17, 19))
    grid_coordinate = np.asarray(operator.grid.coordinate)
    wall_coordinate = np.asarray(operator.wall.coordinate)
    physical = jnp.asarray(
        np.r_[
            flux(grid_coordinate[:, 0], grid_coordinate[:, 1]),
            flux(wall_coordinate[:, 0], wall_coordinate[:, 1]),
        ]
    )

    actual, _state, _connected, admitted = operator._fixed_design_read(physical)
    initial = operator.topology.read_qualification(
        physical, operator.polarity, operator.inside_material
    )
    _seed, material = axis_cell_seed(
        operator.grid.coordinate, initial.state.axis, operator.inside_material
    )
    retained = operator.topology.read_qualification(
        physical, operator.polarity, material
    )

    assert bool(admitted)
    differing = np.flatnonzero(
        np.asarray(actual.label) != np.asarray(retained.masks.label)
    )
    assert differing.size == 0
    np.testing.assert_array_equal(actual.label, retained.masks.label)


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

    assert flood_shadow.shape == (operator.grid.node_number,)
    assert wall_shadow.shape == (operator.wall.node_number,)
    assert flood_shadow.dtype == jnp.bool_
    assert wall_shadow.dtype == jnp.bool_
