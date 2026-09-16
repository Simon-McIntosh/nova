"""Axis-cell ownership at the forward source/topology boundary."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import axis_cell_seed
from nova.equilibrium.source import PolynomialFluxFunction


def _lattice() -> FluxLattice:
    """Return a compact valid tensor grid with one unambiguous centre cell."""
    return FluxLattice(
        np.linspace(0.7, 1.3, 7),
        np.linspace(-0.3, 0.3, 7),
    )


def test_continuous_axis_admits_only_its_owning_cell() -> None:
    """A centre-excluded wall cell remains an occupiable axis seed."""
    lattice = _lattice()
    material = np.ones(lattice.node_count, dtype=bool)
    owner = 3 * lattice.height.size + 3
    material[owner] = False

    seed, repaired = axis_cell_seed(
        lattice.coordinate,
        jnp.asarray([1.04, 0.03]),
        material,
    )

    np.testing.assert_array_equal(np.flatnonzero(np.asarray(seed)), [owner])
    assert bool(repaired[owner])
    np.testing.assert_array_equal(
        np.asarray(repaired) & ~material,
        np.asarray(seed),
    )


def test_axis_cell_ownership_is_jit_and_batch_safe() -> None:
    """Axis motion changes one fixed-shape seed without host selection."""
    lattice = _lattice()
    material = jnp.ones(lattice.node_count, dtype=bool)
    axes = jnp.asarray([[0.71, -0.29], [1.29, 0.28]])

    seeds, repaired = jax.jit(jax.vmap(axis_cell_seed, in_axes=(None, 0, None)))(
        jnp.asarray(lattice.coordinate), axes, material
    )

    np.testing.assert_array_equal(np.sum(np.asarray(seeds), axis=1), [1, 1])
    assert np.asarray(repaired).all()


def test_polynomial_flux_function_traces_coefficients_and_normalisation() -> None:
    """A fixed evaluator exposes every varying profile value as a pytree leaf."""
    coefficients = jnp.asarray([2.0, -3.0, 0.5])
    normalisation = jnp.asarray(4.0)
    profile = PolynomialFluxFunction(coefficients, normalisation)
    coordinate = jnp.asarray([0.0, 0.5, 1.0])

    leaves = jax.tree_util.tree_leaves(profile)
    assert len(leaves) == 2
    np.testing.assert_array_equal(leaves[0], coefficients)
    np.testing.assert_array_equal(leaves[1], normalisation)
    np.testing.assert_array_equal(
        profile(coordinate),
        4.0 * (2.0 - 3.0 * coordinate + 0.5 * coordinate**2),
    )
    tangent = jax.jacfwd(lambda value: PolynomialFluxFunction(value)(0.25))(
        coefficients
    )
    np.testing.assert_array_equal(tangent, jnp.asarray([1.0, 0.25, 0.0625]))
