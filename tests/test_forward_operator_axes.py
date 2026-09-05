"""Structured-axis and batching requirements for forward topology reads."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import fit_tensor_spline


def _operator(coordinate: np.ndarray) -> ForwardFluxOperator:
    """Build a minimal forward operator on the supplied grid coordinates."""
    radial = np.linspace(0.5, 1.5, 9)
    vertical = np.linspace(-0.5, 0.5, 9)
    lattice = FluxLattice(radial, vertical)
    angle = 2.0 * np.pi * np.arange(64) / 64
    wall_coordinate = np.c_[1.0 + 0.7 * np.cos(angle), 0.7 * np.sin(angle)]
    node_count = lattice.node_count
    grid = FluxTarget(
        source_target=jnp.zeros((node_count, 1)),
        plasma_target=jnp.zeros((node_count, 1)),
        null=Null2D.from_coordinates(coordinate, hex_stencil(lattice.shape)),
    )
    wall = FluxTarget(
        source_target=jnp.zeros((len(wall_coordinate), 1)),
        plasma_target=jnp.zeros((len(wall_coordinate), 1)),
        null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
    )

    def zero(psi_norm):
        return jnp.zeros_like(psi_norm)

    return ForwardFluxOperator(
        grid=grid,
        wall=wall,
        source=ForwardSource(core=DomainProfile(p_prime=zero, ff_prime=zero)),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
    )


def _flux_field(coordinate: np.ndarray) -> np.ndarray:
    """Return a smooth field with one magnetic axis and one saddle."""
    radial = coordinate[:, 0] - 1.0
    height = coordinate[:, 1]
    separation = 0.22
    return -(radial**3 / 3.0 - separation**2 * radial + height**2)


# The class margin is read on the fitted tensor spline, so the wall operand no
# longer comes from the analytic wall values.  Evaluate the fixture wall flux on
# the same spline and hold it below the separatrix: the old ``+0.2`` offset put
# the inboard wall pole above the axis value, which moved the boundary selection
# onto the wall and emptied the axis-closed region on the spline read.
_WALL_SEPARATRIX_OFFSET = 0.2


def test_non_tensor_product_grid_is_refused_without_authored_cell_polygons() -> None:
    """The carrier-cell flux read needs authored polygons, so the tensor-product
    requirement gates construction before any margin access is reachable."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.5, 1.5, 9), np.linspace(-0.5, 0.5, 9))
    coordinate = np.asarray(lattice.coordinate).copy()
    coordinate[40, 0] += 1.0e-3

    with pytest.raises(ValueError, match="cell polygons must have at least"):
        _operator(coordinate)


def test_tensor_product_grid_class_margin_is_read_on_the_spline() -> None:
    """The structured-grid class margin comes from the fitted tensor spline."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.5, 1.5, 9), np.linspace(-0.5, 0.5, 9))
    operator = _operator(lattice.coordinate)
    grid_flux = _flux_field(lattice.coordinate)
    wall_coordinate = np.asarray(operator.wall.coordinate)
    surface = fit_tensor_spline(
        jnp.asarray(lattice.radius, dtype=jnp.float64),
        jnp.asarray(lattice.height, dtype=jnp.float64),
        jnp.asarray(grid_flux.reshape((9, 9)).T, dtype=jnp.float64),
    )
    wall_flux = (
        np.asarray(
            surface(
                jnp.asarray(wall_coordinate[:, 0], dtype=jnp.float64),
                jnp.asarray(wall_coordinate[:, 1], dtype=jnp.float64),
            )
        )
        - _WALL_SEPARATRIX_OFFSET
    )
    state = jnp.asarray(np.r_[grid_flux, wall_flux])

    radius, height, shape = operator.connectivity_grid_axes()
    np.testing.assert_array_equal(radius, lattice.radius)
    np.testing.assert_array_equal(height, lattice.height)
    assert shape == lattice.shape

    _masks, topology = operator.read(state)

    assert float(topology.class_margin) == pytest.approx(
        -0.31887492472951007, rel=1.0e-12, abs=1.0e-12
    )
    assert float(operator.topology_margin(state)) == pytest.approx(
        float(topology.class_margin), rel=0.0, abs=0.0
    )

    _pinned_masks, pinned = operator.read(state, TopologyClass.DIVERTED)
    assert float(pinned.class_margin) < 0.0
    assert not bool(pinned.diverted)


def test_forward_topology_state_is_a_vmap_output_pytree() -> None:
    """A batched forward read returns every topology array and its margin."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.5, 1.5, 9), np.linspace(-0.5, 0.5, 9))
    operator = _operator(lattice.coordinate)
    grid_flux = _flux_field(lattice.coordinate)
    wall_flux = _flux_field(np.asarray(operator.wall.coordinate)) + 0.2
    state = jnp.asarray(np.r_[grid_flux, wall_flux])

    topology = jax.vmap(lambda flux: operator.read(flux)[1])(jnp.stack((state, state)))

    assert topology.axis.shape == (2, 2)
    assert topology.diverted.shape == (2,)
    assert topology.class_determinate.shape == (2,)
    assert topology.class_margin.shape == (2,)
    np.testing.assert_array_equal(topology.axis[0], topology.axis[1])
    np.testing.assert_array_equal(topology.diverted[0], topology.diverted[1])
    np.testing.assert_array_equal(
        topology.class_determinate[0], topology.class_determinate[1]
    )
    np.testing.assert_array_equal(topology.class_margin[0], topology.class_margin[1])
