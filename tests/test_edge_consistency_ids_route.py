"""The edge-consistency measure through the equilibrium IDS route.

The analytic lower-single-null state has three exactly known answers: the
current its flux map implies integrates to its own plasma current, the current
its stored flux functions drive integrates to the same value, and the band
residual between the two sits below the fixed threshold because the only error
left is the structured-grid stencil's own.  Writing that state into an
equilibrium IDS in a scratch database at DD 3, in the COCOS 11 sense a DD3
write carries, and reading it back through the translated route the ITER
sources take is what shows that the route -- not only the measure -- can see an
edge inconsistency.

The negative control is the same state with FF' perturbed inside the edge band
only: the flux map written is byte-identical, the band mask is identical cell
for cell, and the measure must report it inconsistent beyond the threshold.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import imas
import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR, toroidal_current_density
from nova.imas.edge_consistency import (
    EDGE_BAND_RELATIVE_THRESHOLD,
    EDGE_BAND_WIDTH,
    edge_consistency,
    grid_current_density,
)
from nova.imas.plasma_source import PlasmaSource
from nova.jax.config import configure_dtypes

configure_dtypes()

DD_VERSION = "3.39.0"
RADIAL_BOUNDS = (1.08, 2.30)
VERTICAL_BOUNDS = (-1.10, 1.00)
STORED_POINTS = 65
PROFILE_SAMPLES = 81
VACUUM_FIELD = 5.3

#: The two scalings that carry the translated read into the measure's operands.
#: The measure works in total poloidal flux ``Phi`` with derivatives against
#: ``Phi``; the reader returns the DD3 flux per radian and its derivatives.
MEASURE_FLUX_SCALE = 1.0
MEASURE_DERIVATIVE_SCALE = 1.0


def _stored_axis(bounds: tuple[float, float], points: int) -> np.ndarray:
    """Return a non-uniform axis over the bounds, clustered toward both ends."""

    unit = np.linspace(0.0, 1.0, points)
    warped = unit - 0.5 * np.sin(2.0 * np.pi * unit) / (2.0 * np.pi)
    return bounds[0] + (bounds[1] - bounds[0]) * warped


def _flux_functions(state):
    """Return the exact Solov'ev ``p'`` and ``FF'`` in nova's COCOS 17 chain."""

    radius = state.major_radius
    scale = state.flux_scale_per_radian_wb
    p_prime = (
        scale * (1.0 - state.source_parameter) / (TOTAL_FLUX_FACTOR * radius**4 * mu_0)
    )
    ff_prime = scale * state.source_parameter / (TOTAL_FLUX_FACTOR * radius**2)
    return p_prime, ff_prime


def _sample(state, grid_r, grid_z):
    """Return the total poloidal flux in Wb on a ``(Z, R)`` mesh."""

    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    points = np.column_stack((mesh_r.ravel(), mesh_z.ravel()))
    flux = (TOTAL_FLUX_FACTOR * state.flux(points)).reshape(mesh_r.shape)
    return mesh_r, mesh_z, flux


def _flux_bounds(state) -> tuple[float, float]:
    """Return the state's total poloidal flux at the axis and at the X-point."""

    psi_axis = TOTAL_FLUX_FACTOR * state.axis_flux
    psi_boundary = TOTAL_FLUX_FACTOR * float(state.flux(state.x_point[None, :])[0])
    return psi_axis, psi_boundary


def _write_analytic_state(directory, band_scale: float = 1.0) -> None:
    """Write the analytic state into a scratch equilibrium IDS at DD 3.

    The write carries the COCOS 11 sense a DD3 reader expects: every psi-like
    row and every flux-derivative row is negated, which is the factor the
    translator applies in the other direction on the way back in.
    """

    state = cerfon_freidberg_single_null()
    grid_r = _stored_axis(RADIAL_BOUNDS, STORED_POINTS)
    grid_z = _stored_axis(VERTICAL_BOUNDS, STORED_POINTS)
    mesh_r, _mesh_z, flux = _sample(state, grid_r, grid_z)
    psi_axis, psi_boundary = _flux_bounds(state)
    p_prime, ff_prime = _flux_functions(state)

    fine_r = np.linspace(*RADIAL_BOUNDS, 257)
    fine_z = np.linspace(*VERTICAL_BOUNDS, 257)
    fine_radius, _mz, fine_flux = _sample(state, fine_r, fine_z)
    fine_norm = (fine_flux - psi_axis) / (psi_boundary - psi_axis)
    fine_area = np.gradient(fine_z)[:, None] * np.gradient(fine_r)[None, :]
    fine_current = toroidal_current_density(
        fine_radius,
        np.full_like(fine_radius, p_prime),
        np.full_like(fine_radius, ff_prime),
    )
    inside = (fine_norm >= 0.0) & (fine_norm <= 1.0)
    plasma_current = float(np.sum(np.where(inside, fine_current, 0.0) * fine_area))

    samples = np.linspace(psi_axis, psi_boundary, PROFILE_SAMPLES)
    sample_norm = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
    edge = sample_norm >= 1.0 - EDGE_BAND_WIDTH
    scaled = np.where(edge, ff_prime * band_scale, ff_prime)

    ids = imas.ids_factory.IDSFactory(DD_VERSION).new("equilibrium")
    ids.ids_properties.homogeneous_time = 1
    ids.ids_properties.version_put.data_dictionary = DD_VERSION
    ids.time = np.array([0.0])
    ids.vacuum_toroidal_field.b0 = np.array([VACUUM_FIELD * np.sign(plasma_current)])
    time_slice = ids.time_slice
    time_slice.resize(1)
    entry = time_slice[0]
    entry.time = 0.0
    global_quantities = entry.global_quantities
    global_quantities.ip = plasma_current
    global_quantities.psi_axis = -psi_axis
    global_quantities.psi_boundary = -psi_boundary
    global_quantities.q_95 = 3.0
    global_quantities.magnetic_axis.r = float(state.magnetic_axis[0])
    global_quantities.magnetic_axis.z = float(state.magnetic_axis[1])
    profile = entry.profiles_1d
    profile.psi = -samples
    profile.dpressure_dpsi = np.full((PROFILE_SAMPLES,), -p_prime)
    profile.f_df_dpsi = np.full((PROFILE_SAMPLES,), -scaled)
    profiles_2d = entry.profiles_2d
    profiles_2d.resize(1)
    profiles_2d[0].grid.dim1 = grid_r
    profiles_2d[0].grid.dim2 = grid_z
    profiles_2d[0].psi = -flux.T
    separatrix = entry.boundary_separatrix
    separatrix.outline.r = np.array(
        [RADIAL_BOUNDS[0], RADIAL_BOUNDS[1], RADIAL_BOUNDS[1]]
    )
    separatrix.outline.z = np.array(
        [VERTICAL_BOUNDS[0], VERTICAL_BOUNDS[0], VERTICAL_BOUNDS[1]]
    )
    separatrix.x_point.resize(1)
    separatrix.x_point[0].r = float(state.x_point[0])
    separatrix.x_point[0].z = float(state.x_point[1])
    entry.boundary.outline.r = np.array(separatrix.outline.r)
    entry.boundary.outline.z = np.array(separatrix.outline.z)

    active = imas.ids_factory.IDSFactory(DD_VERSION).new("pf_active")
    active.ids_properties.homogeneous_time = 1
    active.time = np.full((1,), 0.0)
    active.coil.resize(1)
    active.coil[0].name = "analytic"
    active.coil[0].current.data = np.array([0.0])

    database = imas.DBEntry(f"imas:hdf5?path={directory}", "w", dd_version=DD_VERSION)
    try:
        database.put(ids)
        database.put(active)
    finally:
        database.close()


def _read_back(directory) -> object:
    """Read the scratch IDS through the same route the ITER sources take."""

    database = imas.DBEntry(f"imas:hdf5?path={directory}", "r", dd_version=DD_VERSION)
    try:
        equilibrium = database.get("equilibrium", lazy=False, autoconvert=False)
        active = database.get("pf_active", lazy=False, autoconvert=False)
        source = PlasmaSource.from_imas(
            equilibrium, active, provenance="analytic", indices=[0]
        )
    finally:
        database.close()
    return source.slices[0]


def _measure(plasma_slice, points: int | None = None) -> dict:
    """Measure one read slice over the uniform grid the stencil requires."""

    stored_r = np.asarray(plasma_slice.radius, dtype=float)
    stored_z = np.asarray(plasma_slice.height, dtype=float)
    stored_psi = np.asarray(plasma_slice.psi, dtype=float)
    radial_points = int(stored_r.size) if points is None else points
    vertical_points = int(stored_z.size) if points is None else points
    grid_r = np.linspace(float(stored_r[0]), float(stored_r[-1]), radial_points)
    grid_z = np.linspace(float(stored_z[0]), float(stored_z[-1]), vertical_points)
    # A piecewise-linear resample has a discontinuous second derivative at every
    # stored node, and the stencil differentiates exactly that, so the residual
    # grows under refinement instead of falling.  A cubic spline is C2 and puts
    # the only remaining error back on the stencil.
    spline = RectBivariateSpline(stored_r, stored_z, stored_psi, kx=3, ky=3)
    resampled = spline(grid_r, grid_z)
    flux = MEASURE_FLUX_SCALE * resampled.T

    psi_axis = float(plasma_slice.psi_axis)
    psi_boundary = float(plasma_slice.psi_boundary)
    psi_norm = (flux - psi_axis) / (psi_boundary - psi_axis)
    inside = (psi_norm >= 0.0) & (psi_norm <= 1.0)

    samples = np.asarray(plasma_slice.psi_samples, dtype=float)
    coordinate = (samples - psi_axis) / (psi_boundary - psi_axis)
    order = np.argsort(coordinate)
    coordinate = coordinate[order]
    pressure_samples = np.asarray(plasma_slice.p_prime, dtype=float)[order]
    field_samples = np.asarray(plasma_slice.ff_prime, dtype=float)[order]

    def p_prime_of(normalised):
        return MEASURE_DERIVATIVE_SCALE * np.interp(
            normalised, coordinate, pressure_samples
        )

    def ff_prime_of(normalised):
        return MEASURE_DERIVATIVE_SCALE * np.interp(
            normalised, coordinate, field_samples
        )

    result = edge_consistency(
        flux,
        grid_r,
        grid_z,
        psi_axis,
        psi_boundary,
        p_prime_of,
        ff_prime_of,
        inside=inside,
    )

    radius_grid, _height_grid = np.meshgrid(grid_r, grid_z)
    profile_current = toroidal_current_density(
        radius_grid, p_prime_of(psi_norm), ff_prime_of(psi_norm)
    )
    grid_current = grid_current_density(flux, grid_r, grid_z)
    cell_area = np.gradient(grid_z)[:, None] * np.gradient(grid_r)[None, :]
    plasma_current = float(plasma_slice.plasma_current)

    def integrate(field):
        good = inside & np.isfinite(field)
        return float(np.sum(np.where(good, field, 0.0) * cell_area))

    return {
        "band_relative": result.band_relative,
        "band_absolute": result.band_absolute,
        "whole_slice_relative": result.whole_slice_relative,
        "band_cell_count": result.band_cell_count,
        "plasma_current": plasma_current,
        "grid_current_ratio": integrate(grid_current) / plasma_current,
        "profile_current_ratio": integrate(profile_current) / plasma_current,
    }


@pytest.fixture(scope="module")
def consistent_state(tmp_path_factory):
    """The unperturbed analytic state read back through the IDS route."""

    directory = tmp_path_factory.mktemp("analytic-equilibrium-ids")
    _write_analytic_state(directory)
    return _measure(_read_back(directory))


@pytest.fixture(scope="module")
def perturbed_state(tmp_path_factory):
    """The same state with FF' perturbed inside the edge band only."""

    directory = tmp_path_factory.mktemp("analytic-equilibrium-ids-perturbed")
    _write_analytic_state(directory, band_scale=1.01)
    return _measure(_read_back(directory))


def test_the_ids_route_hands_the_measure_a_conforming_state(consistent_state):
    """The analytic state survives the round trip at all three exact answers."""

    assert consistent_state["grid_current_ratio"] == pytest.approx(1.0, rel=0.05)
    assert consistent_state["profile_current_ratio"] == pytest.approx(1.0, rel=0.05)
    assert consistent_state["band_relative"] < EDGE_BAND_RELATIVE_THRESHOLD


def test_the_ids_route_reports_an_edge_band_perturbation(perturbed_state):
    """Negative control: the band-only FF' perturbation is refused."""

    assert perturbed_state["band_relative"] > EDGE_BAND_RELATIVE_THRESHOLD
