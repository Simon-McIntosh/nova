"""Edge consistency between the analytic state's flux functions and its map.

The reference is the exact lower-single-null Solov'ev state
(:class:`~nova.equilibrium.analytic_single_null.CerfonFreidbergSingleNull`),
whose strong-form source is known in closed form.  Its flux functions are the
two Solov'ev constants, so the profile current is exact and the only error left
is the structured-grid Grad--Shafranov stencil's own discretisation, which must
vanish at the stencil's second order.
"""

from __future__ import annotations

import jax
import numpy as np
from scipy.constants import mu_0

from nova.jax.config import configure_dtypes

from nova.equilibrium.analytic_single_null import (
    CerfonFreidbergSingleNull,
    cerfon_freidberg_single_null,
)
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.edge_consistency import (
    EDGE_BAND_RELATIVE_THRESHOLD,
    EDGE_BAND_WIDTH,
    EdgeConsistency,
    edge_consistency,
    grid_current_density,
)

# The production Grad--Shafranov read assembles its stencil in jnp, so the
# working precision is set before any array is built: at single precision the
# cancellation inside Delta* leaves a round-off floor above the second-order
# residual measured here.
configure_dtypes()
assert jax.config.jax_enable_x64 is True

RADIAL_BOUNDS = (1.08, 2.30)
VERTICAL_BOUNDS = (-1.10, 1.00)
IDS_GRID_SIZE = 65
RESOLUTIONS = (33, 65, 129)


def _solovev_flux_functions(
    state: CerfonFreidbergSingleNull,
) -> tuple[float, float]:
    """Return the exact Solov'ev ``p'`` and ``FF'`` in nova's COCOS 17 chain.

    The strong-form source is ``Delta*Phi = 4 pi^2 (mu_0 R^2 p' + FF')`` and the
    analytic state's per-radian source is a constant plus a term in ``R^2``, so
    matching the two reads the two constants off directly.
    """

    radius = state.major_radius
    scale = state.flux_scale_per_radian_wb
    p_prime = (
        scale * (1.0 - state.source_parameter) / (TOTAL_FLUX_FACTOR * radius**4 * mu_0)
    )
    ff_prime = scale * state.source_parameter / (TOTAL_FLUX_FACTOR * radius**2)
    return p_prime, ff_prime


def _sample(
    state: CerfonFreidbergSingleNull, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the uniform grid and the total poloidal flux on it."""

    grid_r = np.linspace(*RADIAL_BOUNDS, size)
    grid_z = np.linspace(*VERTICAL_BOUNDS, size)
    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    points = np.column_stack((mesh_r.ravel(), mesh_z.ravel()))
    flux = (TOTAL_FLUX_FACTOR * state.flux(points)).reshape(mesh_r.shape)
    return grid_r, grid_z, flux


def _band_mask(state: CerfonFreidbergSingleNull, flux: np.ndarray) -> np.ndarray:
    """Return the edge-band cells a flux map defines, independently of the measure."""

    psi_axis = TOTAL_FLUX_FACTOR * state.axis_flux
    psi_boundary = TOTAL_FLUX_FACTOR * float(state.flux(state.x_point[None, :])[0])
    psi_norm = (flux - psi_axis) / (psi_boundary - psi_axis)
    return (psi_norm >= 1.0 - EDGE_BAND_WIDTH) & (psi_norm <= 1.0)


def _measure(
    state: CerfonFreidbergSingleNull,
    size: int,
    *,
    plasma_ff_prime_scale: float = 1.0,
    band_ff_prime_scale: float = 1.0,
    band_width: float = EDGE_BAND_WIDTH,
) -> EdgeConsistency:
    """Measure one analytic state, optionally perturbing FF' in the band only."""

    grid_r, grid_z, flux = _sample(state, size)
    p_prime, ff_prime = _solovev_flux_functions(state)
    psi_axis = TOTAL_FLUX_FACTOR * state.axis_flux
    psi_boundary = TOTAL_FLUX_FACTOR * float(state.flux(state.x_point[None, :])[0])

    def p_prime_of(_psi_norm: np.ndarray) -> np.ndarray:
        return np.full_like(flux, p_prime)

    def ff_prime_of(psi_norm: np.ndarray) -> np.ndarray:
        value = np.full_like(psi_norm, ff_prime * plasma_ff_prime_scale)
        edge = psi_norm >= 1.0 - band_width
        return np.where(edge, value * band_ff_prime_scale, value)

    return edge_consistency(
        flux,
        grid_r,
        grid_z,
        psi_axis,
        psi_boundary,
        p_prime_of,
        ff_prime_of,
        band_width=band_width,
    )


def test_grid_read_recovers_a_constant_current_density():
    """A flux whose exact Delta* is a constant current is read exactly."""

    grid_r = np.linspace(1.0, 2.0, 7)
    grid_z = np.linspace(-1.0, 1.0, 7)
    mesh_r, _mesh_z = np.meshgrid(grid_r, grid_z)
    current_density = 7.0
    flux = -TOTAL_FLUX_FACTOR * mu_0 * current_density * mesh_r**3 / 3.0

    observed = grid_current_density(flux, grid_r, grid_z)

    radial = grid_r[2:-2][np.newaxis, :]
    radial_step = grid_r[1] - grid_r[0]
    expected = current_density * (1.0 - radial_step**2 / (3.0 * radial**2))

    np.testing.assert_allclose(
        observed[2:-2, 2:-2],
        np.broadcast_to(expected, observed[2:-2, 2:-2].shape),
        rtol=1.0e-12,
    )
    assert np.count_nonzero(np.isfinite(observed)) == 9


def test_consistent_analytic_state_sits_under_the_fixed_threshold():
    """The exact state is discretised, not inconsistent, at the IDS grid."""

    result = _measure(cerfon_freidberg_single_null(), IDS_GRID_SIZE)

    assert result.band_cell_count > 100
    assert result.scored_cell_count > result.band_cell_count
    assert result.band_relative < EDGE_BAND_RELATIVE_THRESHOLD
    assert result.whole_slice_relative < result.band_relative


def test_whole_slice_form_is_the_profile_minus_grid_read_rms():
    """The first form reproduces the parity chain's figure from first principles."""

    state = cerfon_freidberg_single_null()
    size = 65
    grid_r, grid_z, flux = _sample(state, size)
    p_prime, ff_prime = _solovev_flux_functions(state)
    psi_axis = TOTAL_FLUX_FACTOR * state.axis_flux
    psi_boundary = TOTAL_FLUX_FACTOR * float(state.flux(state.x_point[None, :])[0])
    psi_norm = (flux - psi_axis) / (psi_boundary - psi_axis)

    mesh_r, _mesh_z = np.meshgrid(grid_r, grid_z)
    grid_current = grid_current_density(flux, grid_r, grid_z)
    profile_current = -TOTAL_FLUX_FACTOR * (
        mesh_r * p_prime + ff_prime / (mu_0 * mesh_r)
    )
    scored = np.isfinite(grid_current) & (psi_norm >= 0.0) & (psi_norm <= 1.0)
    expected = np.sqrt(
        np.mean(
            (
                (profile_current[scored] - grid_current[scored])
                / np.max(np.abs(grid_current[scored]))
            )
            ** 2
        )
    )

    result = _measure(state, size)

    assert result.whole_slice_relative == expected


def test_one_percent_ff_prime_perturbation_is_detected():
    """A one per cent FF' error over the whole plasma clears the threshold."""

    state = cerfon_freidberg_single_null()
    consistent = _measure(state, IDS_GRID_SIZE)
    perturbed = _measure(state, IDS_GRID_SIZE, plasma_ff_prime_scale=1.01)

    assert perturbed.band_relative > EDGE_BAND_RELATIVE_THRESHOLD
    assert perturbed.band_relative > 2.0 * consistent.band_relative


def test_edge_band_ff_prime_perturbation_is_reported_inconsistent():
    """Negative control: FF' perturbed only in the band, flux map untouched.

    The flux map is the same array in both runs, so the grid read is unchanged
    and every cell of the difference comes from the profile current the
    perturbed flux function drives.
    """

    state = cerfon_freidberg_single_null()
    grid_r, grid_z, flux = _sample(state, IDS_GRID_SIZE)
    consistent = _measure(state, IDS_GRID_SIZE)
    perturbed = _measure(state, IDS_GRID_SIZE, band_ff_prime_scale=1.01)

    _grid_r, _grid_z, perturbed_flux = _sample(state, IDS_GRID_SIZE)
    np.testing.assert_array_equal(flux, perturbed_flux)

    assert perturbed.band_relative > EDGE_BAND_RELATIVE_THRESHOLD
    band_mask = _band_mask(state, flux)
    assert np.array_equal(band_mask, _band_mask(state, perturbed_flux))
    valid = np.isfinite(grid_current_density(flux, grid_r, grid_z))
    band_cells = int(np.count_nonzero(band_mask & valid))
    assert consistent.band_cell_count == band_cells
    assert perturbed.band_cell_count == band_cells


def test_band_residual_falls_at_the_stencil_second_order():
    """Refining the grid drives the band residual down at order two."""

    state = cerfon_freidberg_single_null()
    spacings = np.array([1.0 / (size - 1) for size in RESOLUTIONS], dtype=float)
    band_relative = np.array(
        [_measure(state, size).band_relative for size in RESOLUTIONS]
    )
    whole_slice = np.array(
        [_measure(state, size).whole_slice_relative for size in RESOLUTIONS]
    )

    assert np.all(np.diff(band_relative) < 0.0)
    assert np.all(np.diff(whole_slice) < 0.0)
    band_order = float(np.polyfit(np.log(spacings), np.log(band_relative), 1)[0])
    whole_order = float(np.polyfit(np.log(spacings), np.log(whole_slice), 1)[0])
    assert 1.7 < band_order < 2.3
    assert 1.7 < whole_order < 2.3
