"""Axis-cell ownership at the forward source/topology boundary."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward_operator import axis_cell_seed
from nova.equilibrium.source import (
    DomainProfile,
    PolynomialFluxFunction,
    project_domain_profile,
    project_flux_function,
)
from nova.jax.config import configure_dtypes


@pytest.fixture(autouse=True)
def _extended_precision():
    """Enable x64 before any array is built.

    The projection's residuals and condition numbers are stated in double
    precision, and the default single-precision backend would silently change
    the measurement rather than fail: coefficients captured in single
    precision still solve to a plausible-looking fit.
    """
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    yield


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


def _synthetic_flux_function(coordinate):
    """A smooth analytic gradient pair on the declared normalised flux."""
    value = 3.0 + 2.0 * coordinate - 1.5 * coordinate**3 + 0.7 * coordinate**5
    return value


def test_flux_function_projection_recovers_a_synthetic_profile() -> None:
    """The projected polynomial reproduces the sampled function it was given.

    The fit carries its shape in dimensionless coefficients and its SI
    magnitude in the normalisation leaf, so the round trip is asserted on the
    evaluation and the split is asserted on the leaves: a projection that
    folded the scale into the coefficients would still pass the evaluation and
    fail the split, and the split is what leaves the physical amplitude free
    for a compensator to move.
    """
    knots = np.linspace(0.0, 1.0, 65)
    projection = project_flux_function(_synthetic_flux_function, knots)
    receipt = projection.receipt()
    assert receipt["basis"] == "monomial"
    assert receipt["tolerance_met"] is True
    assert receipt["relative_residual"] < 1.0e-3
    assert receipt["condition_number"] == receipt["monomial_condition_number"]

    fitted = np.asarray(projection.function(jnp.asarray(knots)))
    reference = _synthetic_flux_function(knots)
    residual = np.max(np.abs(fitted - reference)) / np.max(np.abs(reference))
    assert residual < 1.0e-3
    assert projection.si_scale == float(np.max(np.abs(reference)))
    assert np.max(np.abs(np.asarray(projection.function.coefficients))) <= 1.0


def test_flux_function_projection_anchors_the_ends_past_the_power_basis() -> None:
    """A high order is fitted in the exact-end-value basis, not raw monomials.

    The two endpoint samples are reproduced exactly and the design actually
    solved is far better conditioned than the monomial design of the same
    order, so the reformulation is what makes the higher order usable rather
    than a relabelling of the same matrix.
    """
    knots = np.linspace(0.0, 1.0, 65)
    reference = _synthetic_flux_function(knots)
    projection = project_flux_function(
        _synthetic_flux_function, knots, order=8, maximum_order=8
    )
    receipt = projection.receipt()
    assert receipt["basis"] == "exact-end-value"
    assert receipt["order"] == 8
    assert receipt["condition_number"] < receipt["monomial_condition_number"] / 5.0
    endpoints = np.asarray(projection.function(jnp.asarray([0.0, 1.0])))
    # The anchoring is exact by construction; the slack is the roundoff of
    # evaluating the re-expanded monomials at the upper end, ten orders below
    # the fit residual the same projection reports.
    np.testing.assert_allclose(
        endpoints, [reference[0], reference[-1]], rtol=0.0, atol=1.0e-9
    )


def test_flux_function_projection_refuses_ends_it_cannot_anchor() -> None:
    """A high-order fit off the unit interval is refused, not silently worse."""
    knots = np.linspace(0.2, 1.0, 33)
    with pytest.raises(ValueError, match="span"):
        project_flux_function(_synthetic_flux_function, knots, order=8, maximum_order=8)


def test_bank_row_projected_profile_round_trips() -> None:
    """One bank row's extracted gradients survive the polynomial projection.

    The samples are the stored extraction itself, so the residual reported
    here is the projection error on the profiles the Shafranov row is stated
    against rather than on a synthetic stand-in.
    """
    import zarr

    from benchmarks.efit_forward_parity_slice import _profile_function
    from nova.imas.mast_solve_inputs import SHOT_STORE

    shot, row = 21978, 35
    store = Path(SHOT_STORE) / f"{shot}.zarr/efm"
    if not store.exists():
        pytest.skip(f"the extracted profile store {store} is not mounted here")
    group = zarr.open_group(str(store), mode="r")
    psi_norm = np.asarray(group["psi_norm"], dtype=np.float64)
    assert psi_norm.shape == (65,)
    np.testing.assert_allclose(psi_norm, np.linspace(0.0, 1.0, 65))
    total_flux_factor = 2.0 * np.pi
    gradients = {
        "p_prime": -np.asarray(group["pprime"][row], dtype=np.float64)
        / total_flux_factor,
        "ff_prime": -np.asarray(group["ffprime"][row], dtype=np.float64)
        / total_flux_factor,
    }
    core = DomainProfile(
        p_prime=_profile_function(psi_norm, gradients["p_prime"]),
        ff_prime=_profile_function(psi_norm, gradients["ff_prime"]),
    )
    projection = project_domain_profile(core, psi_norm)
    evaluation = np.linspace(0.0, 1.0, 257)
    for name, fit in (
        ("p_prime", projection.p_prime),
        ("ff_prime", projection.ff_prime),
    ):
        reference = np.asarray(
            getattr(core, name)(jnp.asarray(evaluation)), dtype=np.float64
        )
        fitted = np.asarray(fit.function(jnp.asarray(evaluation)), dtype=np.float64)
        scale = float(np.max(np.abs(reference)))
        print(
            f"BANK-PROJECTION {name} order={fit.order} basis={fit.basis} "
            f"relative_residual={fit.relative_residual:.6g} "
            f"condition_number={fit.condition_number:.6g} "
            f"maximum_relative_residual="
            f"{float(np.max(np.abs(fitted - reference))) / scale:.6g}"
        )
        assert float(np.max(np.abs(fitted - reference))) / scale < 1.0e-2
        assert fit.condition_number < 1.0e4
