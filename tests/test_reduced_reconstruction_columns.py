"""The reduced reconstructions' own IMAS-equilibrium conversions.

Each conversion writes the field set the estimator scoreboard reads: a closed
boundary outline, the global quantities, the flux profile and the reconstructed
map on the grid the reconstruction used. These tests pin the conventions the
benchmark depends on -- the map's axis ordering, the vacuum toroidal field and
the vacuum toroidal-field closure -- on a synthetic quadratic flux whose axis is
known, so no reconstruction has to run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from nova.equilibrium.harmonic import harmonic_equilibrium
from nova.equilibrium.moment import moment_equilibrium
from nova.equilibrium.profile import ProfileResult, profile_equilibrium

GRID_R = np.linspace(5.6, 6.8, 17)
GRID_Z = np.linspace(-1.2, 1.2, 25)
AXIS_R = 6.2
SPAN = 40.0

RADIAL, VERTICAL = np.meshgrid(GRID_R, GRID_Z)
INSIDE = (np.abs(VERTICAL) < 1.0) & (np.abs(RADIAL - AXIS_R) < 0.5)
LIMITER_R = np.array([5.2, 5.2, 7.2, 7.2, 5.2])
LIMITER_Z = np.array([-2.0, 2.0, 2.0, -2.0, -2.0])


def flux_field(span: float = SPAN) -> np.ndarray:
    """Return a quadratic flux on the (vertical, radial) grid with a known axis."""

    return -span * ((RADIAL - AXIS_R) ** 2 + VERTICAL**2)


def ring(radius: float = 0.3) -> np.ndarray:
    """Return the closed boundary ring of a circle about the axis."""

    angle = np.linspace(0.0, 2.0 * np.pi, 33)
    return np.column_stack([AXIS_R + radius * np.cos(angle), radius * np.sin(angle)])


def assert_common_state(equilibrium, flux2d: np.ndarray) -> None:
    """Assert the field set every conversion writes for the scoreboard."""

    frame = equilibrium.time_slice[0]
    stored = frame.profiles_2d[0]
    np.testing.assert_allclose(stored.grid.dim1, GRID_R)
    np.testing.assert_allclose(stored.grid.dim2, GRID_Z)
    np.testing.assert_allclose(stored.psi, flux2d.T)
    np.testing.assert_allclose(
        frame.boundary.outline.r[0], frame.boundary.outline.r[-1], atol=1e-12
    )
    np.testing.assert_allclose(
        frame.boundary.outline.z[0], frame.boundary.outline.z[-1], atol=1e-12
    )
    assert frame.profiles_1d.psi.size == GRID_R.size
    np.testing.assert_allclose(frame.profiles_1d.f_df_dpsi, 0.0)
    assert equilibrium.vacuum_toroidal_field.r0 == pytest.approx(6.2)
    assert equilibrium.vacuum_toroidal_field.b0[0] == pytest.approx(-5.3)


def test_the_moment_conversion_writes_its_states_own_map():
    flux = flux_field()
    reconstruction = SimpleNamespace(
        ring=ring(),
        flux=flux,
        centroid_r=AXIS_R,
        centroid_z=0.0,
        cell_current=np.array([1.0e6, -1.0e6]),
        axis_flux=float(flux.min()),
        boundary_flux=-SPAN * 0.3**2,
    )
    equilibrium = moment_equilibrium(
        reconstruction,
        GRID_R,
        GRID_Z,
        INSIDE,
        vacuum_b0=5.3,
        reference_radius=6.2,
    )
    assert_common_state(equilibrium, flux)
    frame = equilibrium.time_slice[0]
    assert frame.global_quantities.ip == pytest.approx(0.0)
    assert frame.global_quantities.psi_axis == pytest.approx(flux.min())
    assert frame.global_quantities.psi_boundary == pytest.approx(-SPAN * 0.3**2)


def test_the_profile_conversion_reads_the_solve_ordering():
    flux2d = flux_field()
    result = ProfileResult(
        flux=flux2d.reshape(-1),
        cell_current=np.array([2.0e6, -1.5e6]),
        coefficients=np.zeros(2),
        residual=np.zeros(()),
        axis=np.array([AXIS_R, 0.0]),
        boundary_flux=-SPAN * 0.3**2,
        core_weight=np.zeros(1),
        axis_state=np.zeros((), dtype=int),
        axis_confidence=np.zeros(()),
        axis_candidate_count=np.zeros((), dtype=int),
        axis_overflow=np.zeros((), dtype=bool),
    )
    equilibrium = profile_equilibrium(
        result,
        GRID_R,
        GRID_Z,
        INSIDE,
        limiter_r=LIMITER_R,
        limiter_z=LIMITER_Z,
        vacuum_b0=5.3,
        reference_radius=6.2,
    )
    assert_common_state(equilibrium, flux2d)
    frame = equilibrium.time_slice[0]
    assert frame.global_quantities.ip == pytest.approx(0.5e6)
    assert frame.global_quantities.magnetic_axis.r == pytest.approx(AXIS_R)


def test_the_harmonic_conversion_writes_the_total_flux():
    inversion = SimpleNamespace(
        config=SimpleNamespace(pole_r=AXIS_R, pole_z=0.0),
        flux_on_grid=lambda grid_r, grid_z: flux_field(),
    )
    equilibrium = harmonic_equilibrium(
        inversion,
        GRID_R,
        GRID_Z,
        INSIDE,
        vacuum_flux=np.zeros_like(flux_field()),
        limiter_r=LIMITER_R,
        limiter_z=LIMITER_Z,
        plasma_current=-1.0e6,
        vacuum_b0=5.3,
        reference_radius=6.2,
    )
    assert_common_state(equilibrium, flux_field())
    frame = equilibrium.time_slice[0]
    assert frame.global_quantities.ip == pytest.approx(-1.0e6)
