"""Cold current-moment seeds on rectangular operators without cell polygons."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import CellCurrentMoments
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


def _zero_profile(value):
    return jnp.zeros_like(value)


def _raster_profile() -> ForwardProfile:
    """Build a small structured operator with no polygon moment geometry."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.6, 1.4, 9), np.linspace(-0.4, 0.4, 9))
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    wall_coordinate = np.c_[
        1.0 + 0.5 * np.cos(wall_angle),
        0.5 * np.sin(wall_angle),
    ]
    node_count = lattice.node_count
    wall_count = len(wall_coordinate)
    operator = ForwardFluxOperator(
        grid=FluxTarget(
            source_target=jnp.zeros((node_count, 1)),
            plasma_target=jnp.eye(node_count),
            null=Null2D.from_coordinates(
                lattice.coordinate, hex_stencil(lattice.shape), maxsize=5
            ),
        ),
        wall=FluxTarget(
            source_target=jnp.zeros((wall_count, 1)),
            plasma_target=jnp.zeros((wall_count, node_count)),
            null=Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        use_linear_moments=False,
    )
    assert operator.moment_geometry is None
    return ForwardProfile(operator=operator, lattice=lattice)


def test_geometry_free_coupling_accepts_only_zero_first_moments() -> None:
    """A raster carrier needs no inversion for its uniform-disc seed."""
    operator = _raster_profile().operator
    current = jnp.asarray([2.0, -1.0, 4.0])
    zero = jnp.zeros_like(current)

    coefficients = operator.coupling_current_moments(
        CellCurrentMoments(current, zero, zero)
    )

    np.testing.assert_array_equal(coefficients.cell_current, current)
    np.testing.assert_array_equal(coefficients.radial_moment, zero)
    np.testing.assert_array_equal(coefficients.vertical_moment, zero)

    with pytest.raises(ValueError, match="moment geometry.*nonzero first"):
        operator.coupling_current_moments(
            CellCurrentMoments(current, zero.at[1].set(0.25), zero)
        )


def test_cold_seed_portfolio_builds_on_geometry_free_raster() -> None:
    """The neutral cold branches compose the external and zeroth-moment images."""
    profile = _raster_profile()

    portfolio = profile.cold_seed_portfolio(12_000.0, (1.0, 0.0))

    branches = portfolio.branches
    assert branches.flux.shape == (2, profile.operator.node_number)
    np.testing.assert_array_equal(branches.flux[0], branches.flux[1])
    np.testing.assert_array_equal(branches.plasma_current, [12_000.0, 12_000.0])
    assert np.all(np.asarray(branches.supported_cells) > 0)


def test_geometry_coupling_is_bit_identical_to_the_existing_formula() -> None:
    """The polygon-backed operator retains its exact coefficient calculation."""
    lattice = FluxLattice(np.linspace(0.6, 1.4, 9), np.linspace(-0.4, 0.4, 9))
    node_count = lattice.node_count
    wall_angle = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    profile = ForwardProfile.from_lattice(
        lattice,
        ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=np.zeros(1),
        source_to_grid=np.zeros((node_count, 1)),
        plasma_to_grid=np.zeros((node_count, node_count)),
        source_to_wall=np.zeros((wall_angle.size, 1)),
        plasma_to_wall=np.zeros((wall_angle.size, node_count)),
        wall_coordinate=np.c_[
            1.0 + 0.5 * np.cos(wall_angle),
            0.5 * np.sin(wall_angle),
        ],
        cubic_cell_average=False,
    )
    current = jnp.linspace(1.0, 2.0, node_count)
    radial_moment = jnp.linspace(-0.2, 0.3, node_count)
    vertical_moment = jnp.linspace(0.4, -0.1, node_count)
    moments = CellCurrentMoments(current, radial_moment, vertical_moment)
    second = jnp.asarray(profile.operator.moment_geometry.second_moment)
    determinant = second[:, 0] * second[:, 1] - second[:, 2] ** 2
    expected = CellCurrentMoments(
        current,
        (second[:, 1] * radial_moment - second[:, 2] * vertical_moment) / determinant,
        (second[:, 0] * vertical_moment - second[:, 2] * radial_moment) / determinant,
    )

    actual = profile.operator.coupling_current_moments(moments)

    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
