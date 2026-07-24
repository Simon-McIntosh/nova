"""Contract for the source-free toroidal-harmonic annulus reconstruction.

Three claims stand without any machine data. The basis is *structurally*
source-free: every column drives the numerical Grad-Shafranov operator to the
finite-difference floor, so no current is committed to the annulus. The read is
*exact* where its premise holds: a field that is a harmonic combination is
recovered from its noise-free sensor signature, and the exact exterior flux of
current filaments inside the pole is reproduced at held-out annulus points. And
its *gauge* behaves: the circulation tie pins the net current, the gradient form
ignores an absolute flux shift, and the invalid near-pole interior is masked
rather than read.
"""

import numpy as np
import pytest

from nova.biot.greens import greens_psi
from nova.equilibrium.harmonic import (
    HarmonicConfig,
    ReconstructHarmonic,
    annulus_penalty_rows,
    annulus_points,
    circulation_row,
    grad_shafranov_residual,
    harmonic_columns,
    harmonic_labels,
    huber_weights,
    mode_penalty,
    ring_legendre_p1,
    toroidal_coordinates,
)
from nova.equilibrium.measurement import Magnetics, SliceMeasurement

POLE = (0.9, 0.0)


def config(order=3, **kwargs):
    """Return a harmonic configuration about the reference pole."""
    return HarmonicConfig(pole_r=POLE[0], pole_z=POLE[1], order=order, **kwargs)


def sensor_ring(n_sensor=48, radius=0.7, seed=0):
    """Return flux loops and field probes on a ring around the plasma."""
    rng = np.random.default_rng(seed)
    angle = np.linspace(0.0, 2.0 * np.pi, n_sensor, endpoint=False)
    flux_loop = np.zeros(n_sensor, bool)
    flux_loop[::2] = True
    return Magnetics(
        r=POLE[0] + radius * np.cos(angle),
        z=POLE[1] + radius * np.sin(angle),
        angle=np.where(flux_loop, 0.0, rng.uniform(0.0, 180.0, size=n_sensor)),
        flux_loop=flux_loop,
    )


def measurement(signature, plasma_current=1.0e5, mask=None):
    """Return a noise-free measurement carrying a given plasma signature."""
    n_sensor = signature.size
    return SliceMeasurement(
        measured=signature,
        vacuum=np.zeros(n_sensor),
        mask=np.ones(n_sensor, bool) if mask is None else mask,
        scale=np.ones(n_sensor),
        plasma_current=plasma_current,
    )


# --- the toroidal coordinate system and its ring functions ------------------


def test_toroidal_transform_round_trips():
    """The inverse transform inverts the forward map exactly."""
    eta = np.array([0.3, 0.8, 1.5, 2.5])
    theta = np.array([0.2, 1.1, -2.0, 3.0])
    a = POLE[0]
    denominator = np.cosh(eta) - np.cos(theta)
    r = a * np.sinh(eta) / denominator
    z = POLE[1] + a * np.sin(theta) / denominator
    cosh_eta, cos_theta, sin_theta, angle, prefactor = toroidal_coordinates(
        r, z, *POLE
    )
    np.testing.assert_allclose(cosh_eta, np.cosh(eta), rtol=1e-10)
    np.testing.assert_allclose(cos_theta, np.cos(theta), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sin_theta, np.sin(theta), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.cos(angle), np.cos(theta), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prefactor, denominator, rtol=1e-10)


def test_the_focal_ring_and_the_far_field_are_the_coordinate_limits():
    """cosh eta diverges on the focal ring and returns to one far from it."""
    on_ring = toroidal_coordinates(np.array([POLE[0]]), np.array([POLE[1]]), *POLE)[0]
    far = toroidal_coordinates(np.array([400.0]), np.array([0.0]), *POLE)[0]
    assert on_ring[0] > 50.0
    assert np.isclose(far[0], 1.0, atol=1e-4)


def test_the_ring_functions_vanish_at_the_far_field_limit():
    """The radial ladder is finite, and vanishes as the far field is approached.

    ``x = cosh eta`` runs from 1 — the symmetry axis and spatial infinity — up to
    infinity on the focal ring, so vanishing at ``x = 1`` is the exterior
    regularity that makes this the physical radial set: a field from sources
    inside the pole cannot survive out to infinity. The quantitative correctness
    of the ladder is pinned by the homogeneous-operator and filament-recovery
    tests below.
    """
    x = np.array([1.0 + 1e-9, 1.05, 1.4, 3.0, 12.0, 1.0e4])
    ladder = ring_legendre_p1(4, x)
    assert ladder.shape == (5, x.size)
    assert np.all(np.isfinite(ladder))
    for n in range(5):
        magnitude = np.abs(ladder[n])
        assert magnitude[0] < 1e-3  # regular at spatial infinity
        assert magnitude[0] < magnitude[1:].max()


# --- structurally source-free ----------------------------------------------


def test_every_column_solves_the_homogeneous_grad_shafranov_operator():
    """No column commits current to the annulus, to the finite-difference floor."""
    grid_r = np.linspace(0.3, 1.9, 80)
    grid_z = np.linspace(-1.2, 1.2, 90)
    radius, height = np.meshgrid(grid_r, grid_z)
    columns, _labels = harmonic_columns(radius.ravel(), height.ravel(), config())
    away = (
        np.hypot(radius.ravel() - POLE[0], height.ravel() - POLE[1]) > 0.35
    ).reshape(height.shape)
    for index in range(columns.shape[1]):
        psi = columns[:, index].reshape(height.shape)
        residual = grad_shafranov_residual(psi, grid_r, grid_z)
        interior = away[1:-1, 1:-1] & np.isfinite(residual[1:-1, 1:-1])
        # scale by the field's own second-derivative magnitude so a flat column
        # cannot pass by being small
        scale = np.nanmax(np.abs(psi)) / (grid_r[1] - grid_r[0]) ** 2 + 1e-30
        relative = np.nanmax(np.abs(residual[1:-1, 1:-1][interior])) / scale
        assert relative < 1e-3, f"column {index}: relative residual {relative:.2e}"


# --- exactness where the premise holds --------------------------------------


def test_recovers_a_source_free_field_exactly():
    """A field that is a harmonic combination is recovered to numerical precision."""
    read = ReconstructHarmonic(sensor_ring(), config())
    rng = np.random.default_rng(1)
    truth = rng.standard_normal(read.sensor_matrix.shape[1])
    signature = read.sensor_matrix @ truth
    inversion = read.fit(measurement(signature))
    assert inversion.misfit < 1e-12
    np.testing.assert_allclose(
        read.sensor_matrix @ inversion.coefficients, signature, rtol=1e-6, atol=1e-9
    )
    assert inversion.labels == harmonic_labels(3)


def test_filament_flux_is_reproduced_at_held_out_annulus_points():
    """Current inside the pole has an exact exterior flux the basis reproduces.

    This is what selects the exterior-regular radial set: a field produced by
    sources inside the pole and observed outward must stay finite out to
    infinity, so it generalises from one annulus ring to a further one.
    """
    filaments = [(0.85, 0.05, 1.0), (0.95, -0.1, 0.7), (0.80, 0.15, 0.5)]
    angle = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)

    def flux(radius):
        r = POLE[0] + radius * np.cos(angle)
        z = POLE[1] + radius * np.sin(angle)
        total = np.zeros(angle.size)
        for filament_r, filament_z, current in filaments:
            total += current * greens_psi(r, z, filament_r, filament_z)
        return r, z, total

    r_fit, z_fit, flux_fit = flux(0.55)
    r_test, z_test, flux_test = flux(0.70)
    fit_columns, _ = harmonic_columns(r_fit, z_fit, config(order=5))
    column_norm = np.linalg.norm(fit_columns, axis=0)
    scaled = np.linalg.lstsq(fit_columns / column_norm, flux_fit, rcond=None)[0]
    coefficients = scaled / column_norm
    test_columns, _ = harmonic_columns(r_test, z_test, config(order=5))
    error = np.sqrt(np.mean((test_columns @ coefficients - flux_test) ** 2))
    assert error / np.sqrt(np.mean(flux_test**2)) < 1e-3


def test_untrusted_rows_do_not_enter_the_fit():
    """Corrupting masked-out rows must not move the coefficients or the flux map."""
    read = ReconstructHarmonic(sensor_ring(seed=2), config(order=2))
    rng = np.random.default_rng(3)
    signature = read.sensor_matrix @ rng.standard_normal(read.sensor_matrix.shape[1])
    mask = np.ones(signature.size, bool)
    mask[::4] = False

    clean = read.fit(measurement(signature.copy(), mask=mask))
    dirty = read.fit(
        measurement(signature + np.where(mask, 0.0, 1e3), mask=mask)
    )
    np.testing.assert_allclose(
        clean.coefficients, dirty.coefficients, rtol=1e-6, atol=1e-9
    )

    grid_r = np.linspace(0.2, 1.9, 33)
    grid_z = np.linspace(-1.0, 1.0, 41)
    psi = clean.flux_on_grid(grid_r, grid_z)
    assert psi.shape == (grid_z.size, grid_r.size)
    assert np.all(np.isfinite(psi))


def test_the_graded_ridge_damps_the_high_modes_hardest():
    """The Sobolev penalty rises with degree and leaves the low-order shape alone."""
    penalty = mode_penalty(3, 2.0)
    assert penalty.size == 2 * 3 + 1
    assert penalty[0] == 1.0
    assert penalty[1] == penalty[2] == 4.0  # degree 1, cosine and sine
    assert penalty[-1] > penalty[1]
    assert np.all(mode_penalty(3, 0.0) == 1.0)


# --- gauge ------------------------------------------------------------------


def test_the_flux_gradient_matches_a_finite_difference_of_the_flux():
    """The reported gradient is the gradient of the reported flux map."""
    read = ReconstructHarmonic(sensor_ring(), config(order=2))
    coefficients = np.arange(1.0, read.sensor_matrix.shape[1] + 1.0)
    # a raster in the annulus, well clear of the pole where the basis diverges
    grid_r = np.linspace(1.15, 1.65, 41)
    grid_z = np.linspace(0.35, 0.85, 37)
    dpsi_dr, dpsi_dz = read.grad_flux_on_grid(coefficients, grid_r, grid_z)
    psi = read.flux_on_grid(coefficients, grid_r, grid_z)
    reference_dz, reference_dr = np.gradient(psi, grid_z, grid_r)
    interior = (slice(2, -2), slice(2, -2))
    # the raster difference is the coarse estimate here, so the tolerance is set
    # by its truncation error rather than by the analytic gradient
    tolerance = 0.01 * np.max(np.abs(reference_dr[interior]))
    np.testing.assert_allclose(
        dpsi_dr[interior], reference_dr[interior], rtol=0.02, atol=tolerance
    )
    np.testing.assert_allclose(
        dpsi_dz[interior], reference_dz[interior], rtol=0.02, atol=tolerance
    )


def test_the_circulation_row_is_path_independent():
    """The gauge tie is the same on any loop enclosing the pole.

    The columns are source-free everywhere except the focal ring, so the
    circulation depends only on the current the loop encloses, not on the path.
    The sine columns are odd about the pole height and enclose no net current, so
    the tie acts purely on the cosine family.
    """
    harmonic = config(order=3)
    near = circulation_row(harmonic, 0.30)
    far = circulation_row(harmonic, 0.45)
    np.testing.assert_allclose(near, far, rtol=1e-3, atol=1e-9)
    labels = harmonic_labels(harmonic.order)
    sine = [index for index, label in enumerate(labels) if label.endswith("s")]
    cosine = [index for index in range(len(labels)) if index not in sine]
    assert np.max(np.abs(near[sine])) < 1e-6 * np.max(np.abs(near[cosine]))


def test_the_circulation_anchor_pins_the_net_current():
    """With the anchor on, the fitted circulation matches mu_0 times Ip."""
    from scipy.constants import mu_0

    magnetics = sensor_ring(seed=7)
    anchored = ReconstructHarmonic(
        magnetics, config(ip_anchor=True, ip_anchor_weight=50.0)
    )
    free = ReconstructHarmonic(magnetics, config(ip_anchor=False))
    # a signature the basis cannot represent exactly, so the DC is not already fixed
    filament = greens_psi(magnetics.r, magnetics.z, 0.88, 0.02)
    signature = magnetics.project(filament, 0.0 * filament, 0.0 * filament) * 3.0e5
    plasma_current = 3.0e5
    slice_data = measurement(signature, plasma_current=plasma_current)

    row = circulation_row(anchored.config, 0.35)
    target = mu_0 * plasma_current
    anchored_error = abs(row @ anchored.fit(slice_data).coefficients - target)
    free_error = abs(row @ free.fit(slice_data).coefficients - target)
    assert anchored_error < 0.05 * abs(target)
    assert anchored_error < free_error


def test_the_anchor_off_leaves_the_sensor_only_fit_untouched():
    """Disabling the anchor gives exactly the sensor-only solve."""
    magnetics = sensor_ring(seed=4)
    read = ReconstructHarmonic(magnetics, config(ip_anchor=False))
    signature = read.sensor_matrix @ np.linspace(1.0, 2.0, read.sensor_matrix.shape[1])
    slice_data = measurement(signature)
    assert read.gauge_anchor(slice_data) is None


def test_the_invalid_interior_is_filled_with_a_confined_plateau():
    """The near-pole disc becomes one confined-side value; the annulus is intact."""
    read = ReconstructHarmonic(sensor_ring(), config())
    grid_r = np.linspace(0.2, 1.9, 40)
    grid_z = np.linspace(-1.2, 1.2, 50)
    radius, height = np.meshgrid(grid_r, grid_z)
    distance = np.hypot(radius - POLE[0], height - POLE[1])
    psi = 1.0 / (distance + 0.05)  # diverges positive toward the pole
    masked = read.mask_invalid_interior(
        psi, grid_r, grid_z, 0.25, axis=(0.96, 0.0)
    )
    inside = distance < 0.25
    np.testing.assert_allclose(masked[~inside], psi[~inside])
    assert np.unique(np.round(masked[inside], 6)).size == 1
    annulus = psi[~inside]
    assert masked[inside].flat[0] > np.median(annulus) + 3.0 * np.std(annulus)


# --- the annulus soft prior -------------------------------------------------


class Grid:
    """Minimal grid carrying the limiter mask the annulus selection reads."""

    def __init__(self, inside_limiter):
        self.inside_limiter = inside_limiter


def test_the_annulus_is_inside_the_limiter_and_outside_the_confined_region():
    flux = np.array([[3.0, 2.0, 1.0], [2.5, 1.5, 0.5]])
    grid = Grid(np.array([[True, True, True], [True, True, False]]))
    index = annulus_points(grid, flux=flux, axis_flux=3.0, boundary_flux=2.0)
    # confined = flux deeper than 2.0 toward 3.0, i.e. flux > 2.0
    np.testing.assert_array_equal(index, np.array([1, 2, 4]))


def test_the_annulus_selection_is_agnostic_to_the_flux_sign_convention():
    """Flipping the sign of every flux reference selects the same annulus."""
    flux = np.array([3.0, 2.0, 1.0, 0.5])
    grid = Grid(np.ones(4, bool))
    rising = annulus_points(grid, flux=flux, axis_flux=3.0, boundary_flux=2.0)
    falling = annulus_points(grid, flux=-flux, axis_flux=-3.0, boundary_flux=-2.0)
    np.testing.assert_array_equal(rising, falling)


def test_the_gradient_form_ignores_an_absolute_flux_shift():
    """A constant added to the target leaves the gradient-form rows unchanged."""
    basis = np.arange(12.0).reshape(6, 2)
    fixed = np.linspace(0.0, 1.0, 6)
    target = np.linspace(1.0, 2.0, 6)
    rows = dict(form="grad-psi", basis=basis, n_profile=2, n_passive=0, weight=2.0)
    design, data = annulus_penalty_rows(fixed=fixed, target=target, **rows)
    shifted_design, shifted_data = annulus_penalty_rows(
        fixed=fixed + 5.0, target=target + 5.0, **rows
    )
    np.testing.assert_allclose(design, shifted_design)
    np.testing.assert_allclose(data, shifted_data)


def test_the_absolute_form_offset_absorbs_a_constant_shift():
    """With the free offset the least-squares residual is shift-invariant."""
    basis = np.linspace(0.0, 1.0, 6)[:, None]
    fixed = np.zeros(6)
    target = np.linspace(1.0, 2.0, 6)
    rows = dict(
        form="abs-psi",
        basis=basis,
        n_profile=1,
        n_passive=0,
        weight=1.0,
        gauge_offset=True,
    )

    def residual(shift):
        design, data = annulus_penalty_rows(
            fixed=fixed, target=target + shift, **rows
        )
        solution = np.linalg.lstsq(design, data, rcond=None)[0]
        return np.linalg.norm(design @ solution - data), solution[0]

    plain, coefficient = residual(0.0)
    shifted, shifted_coefficient = residual(7.0)
    assert abs(plain - shifted) < 1e-8
    assert abs(coefficient - shifted_coefficient) < 1e-8


def test_the_prior_weight_scales_with_the_reads_uncertainty():
    """A less certain read pushes the interior solve proportionally less."""
    basis = np.ones((4, 1))
    design, data = annulus_penalty_rows(
        form="grad-psi",
        basis=basis,
        fixed=np.zeros(4),
        target=np.ones(4),
        n_profile=1,
        n_passive=0,
        weight=1.0,
    )
    loose_design, loose_data = annulus_penalty_rows(
        form="grad-psi",
        basis=basis,
        fixed=np.zeros(4),
        target=np.ones(4),
        n_profile=1,
        n_passive=0,
        weight=1.0,
        uncertainty=4.0,
    )
    np.testing.assert_allclose(loose_design, design / 4.0)
    np.testing.assert_allclose(loose_data, data / 4.0)


def outlying_residual(n_bulk=40, seed=0):
    """Return a residual proxy: a bulk with a well-defined spread, plus one wild
    point."""
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(0.0, 1.0, n_bulk), 100.0]


def test_an_outlying_annulus_point_is_down_weighted():
    """One wild mismatch is Huber-clipped instead of dragging the solve."""
    weight = huber_weights(outlying_residual(), 3.0)
    assert np.median(weight[:-1]) == 1.0
    assert weight[-1] < 0.2


def test_a_degenerate_spread_leaves_the_weights_uniform():
    """With no measurable bulk spread there is no scale to clip against."""
    assert np.all(huber_weights(np.r_[np.zeros(9), 100.0], 3.0) == 1.0)


def test_the_robust_weights_are_invariant_to_a_uniform_shift():
    residual = outlying_residual()
    np.testing.assert_allclose(
        huber_weights(residual, 3.0), huber_weights(residual + 12.0, 3.0)
    )


def test_an_unknown_penalty_form_is_rejected():
    with pytest.raises(ValueError):
        annulus_penalty_rows(
            form="mean-projection",
            fixed=np.zeros(3),
            target=np.zeros(3),
            n_profile=0,
            n_passive=0,
            weight=1.0,
        )
