"""Sign-aware topology reads for equilibria with reversed plasma current."""

import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.equilibrium.stencil_nulls import magnetic_axis_subgrid
from nova.equilibrium.topology import Topology
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _solovev_fixture(size=81):
    """Return a smooth axisymmetric extremum and its rectangular coordinates."""
    radial = np.linspace(0.7, 1.3, size)
    vertical = np.linspace(-0.3, 0.3, size)
    radius, height = np.meshgrid(radial, vertical, indexing="ij")
    distance = (radius - 1.0) ** 2 + 1.4 * height**2
    return -distance, radial, vertical


def test_reversed_current_axis_keeps_position_and_flips_flux_extremum():
    """The reversed-current axis is the minimum at the same physical point."""
    configure_dtypes()
    positive, radial, vertical = _solovev_fixture()
    inside = np.ones_like(positive, dtype=bool)

    forward = magnetic_axis_subgrid(positive, radial, vertical, inside, polarity=1)
    reversed_current = magnetic_axis_subgrid(
        -positive, radial, vertical, inside, polarity=-1
    )

    np.testing.assert_allclose(
        [forward["r"], forward["z"]],
        [reversed_current["r"], reversed_current["z"]],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert bool(forward["found"])
    assert bool(reversed_current["found"])
    assert float(forward["ntype"]) == 1.0
    assert float(reversed_current["ntype"]) == -1.0
    np.testing.assert_allclose(
        float(reversed_current["psi"]), -float(forward["psi"]), rtol=0.0, atol=1.0e-12
    )


def test_topology_selection_mask_and_flood_follow_the_same_current_sign():
    """Axis-side selection, masks and component floods reverse together."""
    configure_dtypes()
    radial = np.linspace(0.5, 1.5, 17)
    vertical = np.linspace(-0.5, 0.5, 17)
    radius, height = np.meshgrid(radial, vertical, indexing="ij")
    coordinate = np.c_[radius.ravel(), height.ravel()]
    topology = Topology(
        Null2D.from_coordinates(coordinate, hex_stencil((17, 17)), maxsize=5),
        Null1D(np.asarray([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])),
    )

    values = np.asarray([-0.2, 0.0, 0.2])
    np.testing.assert_array_equal(
        np.asarray(topology.psi_mask(1, values, 0.0)), [False, True, True]
    )
    np.testing.assert_array_equal(
        np.asarray(topology.psi_mask(-1, values, 0.0)), [True, False, False]
    )

    positive_flux = 1.0 - 4.0 * ((radius - 1.0) ** 2 + height**2)
    reversed_flux = -positive_flux
    inside = np.ones(coordinate.shape[0], dtype=bool)
    positive_closed = topology.psi_mask(1, positive_flux.ravel(), 0.2)
    reversed_closed = topology.psi_mask(-1, reversed_flux.ravel(), -0.2)
    positive_component = topology.axis_component(
        positive_flux.ravel(),
        0.2,
        1.0,
        np.asarray([1.0, 0.0]),
        positive_closed,
        inside,
        polarity=1,
    )
    reversed_component = topology.axis_component(
        reversed_flux.ravel(),
        -0.2,
        -1.0,
        np.asarray([1.0, 0.0]),
        reversed_closed,
        inside,
        polarity=-1,
    )
    np.testing.assert_array_equal(
        np.asarray(positive_component), np.asarray(reversed_component)
    )

    candidates = np.asarray([[1.0, -0.3, 0.8, 0.0], [1.0, 0.3, 0.5, 0.0]], dtype=float)
    positive = topology.x_point_index(candidates, 1, 0.0)
    reversed_current = topology.x_point_index(candidates, -1, 0.0)
    assert int(positive) == 0
    assert int(reversed_current) == 1
