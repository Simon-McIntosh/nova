"""Analytic contract for axisymmetric force-balance diagnostics."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot.greens import MU0, greens_bz_br
from nova.equilibrium import (
    DECAY_INDEX_WINDOW,
    decay_index,
    shafranov_vertical_field,
    shafranov_vertical_field_elongated,
)
from nova.equilibrium.analytic_single_null import (
    CerfonFreidbergSingleNull,
    cerfon_freidberg_single_null,
)
from nova.equilibrium.diagnostics import (
    beta_p_plus_half_internal_inductance,
    shafranov_contour_integrals,
)
from nova.jax.config import configure_dtypes

#: Extended precision is enabled lazily by the executable paths, so a module
#: that constructs arrays first would silently run in single precision and the
#: traced integrals would carry a relative error near 2**-20.
configure_dtypes()
assert jax.config.jax_enable_x64 is True


def test_circular_required_field_matches_signed_ring_balance():
    """A positive toroidal current needs an inward, negative vertical field."""
    field = shafranov_vertical_field(1.0e6, 1.7, 0.5, 1.25)
    expected = -1.0e-7 * 1.0e6 / 1.7 * (math.log(27.2) - 0.25)
    assert field == pytest.approx(expected, rel=1e-12)
    assert field == pytest.approx(-0.17960101, rel=1e-6)


def test_circular_required_field_reverses_with_current():
    positive = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    negative = shafranov_vertical_field(-6.0e5, 0.9, 0.55, 1.0)
    assert positive < 0.0
    assert negative == pytest.approx(-positive, rel=1e-12)


def test_circular_required_field_scales_with_pressure_and_inductance():
    lower = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    higher = shafranov_vertical_field(6.0e5, 0.9, 0.55, 2.0)
    expected_difference = -1.0e-7 * 6.0e5 / 0.9
    assert higher - lower == pytest.approx(expected_difference, rel=1e-12)


@pytest.mark.parametrize(
    ("major_radius", "minor_radius"),
    [(0.0, 0.5), (-0.9, 0.5), (0.9, 0.0), (0.9, -0.5), (0.9, 7.2)],
)
def test_circular_required_field_returns_nan_for_degenerate_geometry(
    major_radius,
    minor_radius,
):
    assert math.isnan(
        shafranov_vertical_field(
            6.0e5,
            major_radius,
            minor_radius,
            1.0,
        )
    )


@pytest.mark.parametrize(
    ("plasma_current", "major_radius", "minor_radius", "profile_term"),
    [
        (float("nan"), 0.9, 0.5, 1.0),
        (float("inf"), 0.9, 0.5, 1.0),
        (6.0e5, float("nan"), 0.5, 1.0),
        (6.0e5, float("inf"), 0.5, 1.0),
        (6.0e5, 0.9, float("nan"), 1.0),
        (6.0e5, 0.9, float("inf"), 1.0),
        (6.0e5, 0.9, 0.5, float("nan")),
        (6.0e5, 0.9, 0.5, float("inf")),
    ],
)
def test_circular_required_field_returns_nan_without_warnings_for_nonfinite_input(
    plasma_current,
    major_radius,
    minor_radius,
    profile_term,
):
    with np.errstate(all="raise"):
        field = shafranov_vertical_field(
            plasma_current,
            major_radius,
            minor_radius,
            profile_term,
        )
    assert math.isnan(field)


def test_elongated_required_field_matches_area_equivalent_radius():
    field = shafranov_vertical_field_elongated(
        1.0e6,
        1.7,
        0.5,
        1.69,
        1.25,
    )
    expected = -1.0e-7 * 1.0e6 / 1.7 * (math.log(8.0 * 1.7 / (0.5 * 1.3)) - 0.25)
    assert field == pytest.approx(expected, rel=1e-12)


def test_elongated_required_field_reduces_to_circular_at_unit_elongation():
    circular = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elongated = shafranov_vertical_field_elongated(
        6.0e5,
        0.9,
        0.55,
        1.0,
        1.0,
    )
    assert elongated == pytest.approx(circular, rel=1e-14)


def test_elongation_softens_required_field_by_half_logarithm():
    circular = shafranov_vertical_field(6.0e5, 0.9, 0.55, 1.0)
    elongated = shafranov_vertical_field_elongated(
        6.0e5,
        0.9,
        0.55,
        1.8,
        1.0,
    )
    assert abs(elongated) < abs(circular)
    expected_difference = -1.0e-7 * 6.0e5 / 0.9 * (0.5 * math.log(1.8))
    assert circular - elongated == pytest.approx(expected_difference, rel=1e-10)


@pytest.mark.parametrize("elongation", [0.0, -1.0, float("nan"), float("inf")])
def test_elongated_required_field_returns_nan_for_invalid_elongation(elongation):
    assert math.isnan(
        shafranov_vertical_field_elongated(
            6.0e5,
            0.9,
            0.5,
            elongation,
            1.0,
        )
    )


@pytest.mark.parametrize(
    ("plasma_current", "major_radius", "minor_radius", "elongation", "profile_term"),
    [
        (float("nan"), 0.9, 0.5, 1.5, 1.0),
        (float("inf"), 0.9, 0.5, 1.5, 1.0),
        (6.0e5, float("nan"), 0.5, 1.5, 1.0),
        (6.0e5, float("inf"), 0.5, 1.5, 1.0),
        (6.0e5, 0.9, float("nan"), 1.5, 1.0),
        (6.0e5, 0.9, float("inf"), 1.5, 1.0),
        (6.0e5, 0.9, 0.5, float("nan"), 1.0),
        (6.0e5, 0.9, 0.5, float("inf"), 1.0),
        (6.0e5, 0.9, 0.5, 1.5, float("nan")),
        (6.0e5, 0.9, 0.5, 1.5, float("inf")),
    ],
)
def test_elongated_required_field_returns_nan_without_warnings_for_nonfinite_input(
    plasma_current,
    major_radius,
    minor_radius,
    elongation,
    profile_term,
):
    with np.errstate(all="raise"):
        field = shafranov_vertical_field_elongated(
            plasma_current,
            major_radius,
            minor_radius,
            elongation,
            profile_term,
        )
    assert math.isnan(field)


@pytest.mark.parametrize("exponent", [0.0, 1.0, 1.4, 3.0])
def test_decay_index_recovers_power_law_exponent(exponent):
    """A field proportional to radius^-exponent has that decay index."""
    radius = np.linspace(0.4, 1.4, 2001)
    vertical_field = -0.3 * radius**-exponent
    index = decay_index(radius, vertical_field)
    np.testing.assert_allclose(index[10:-10], exponent, atol=5e-3)


def test_decay_index_supports_strictly_decreasing_radius():
    radius = np.linspace(1.4, 0.4, 2001)
    vertical_field = -0.3 * radius**-1.4
    index = decay_index(radius, vertical_field)
    np.testing.assert_allclose(index[10:-10], 1.4, atol=5e-3)


def test_decay_index_returns_nan_at_field_reversal():
    radius = np.linspace(0.4, 1.4, 101)
    vertical_field = radius - 0.9
    index = decay_index(radius, vertical_field)
    assert np.isnan(index[np.argmin(np.abs(vertical_field))])


@pytest.mark.parametrize("fill_value", [float("nan"), float("inf"), -float("inf")])
def test_decay_index_returns_nan_without_warnings_for_nonfinite_field(fill_value):
    radius = np.linspace(0.4, 1.4, 7)
    vertical_field = np.full(radius.shape, fill_value)
    with np.errstate(all="raise"):
        index = decay_index(radius, vertical_field)
    assert np.isnan(index).all()


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), -float("inf")])
def test_decay_index_propagates_an_isolated_nonfinite_sample_locally(sample):
    radius = np.linspace(0.4, 1.4, 7)
    vertical_field = -0.3 * radius**-1.4
    vertical_field[3] = sample
    index = decay_index(radius, vertical_field)
    assert np.isfinite(index[[0, 1, 5, 6]]).all()
    assert np.isnan(index[2:5]).all()


def test_decay_index_rejects_undersized_and_unequal_inputs():
    with pytest.raises(ValueError, match="at least two samples"):
        decay_index(np.array([0.9]), np.array([0.2]))
    with pytest.raises(ValueError, match="equal shapes"):
        decay_index(np.array([0.8, 1.0]), np.array([0.2]))


@pytest.mark.parametrize(
    ("radius", "vertical_field"),
    [
        (np.ones((2, 2)), np.ones((2, 2))),
        (np.array([0.8, 1.0]), np.ones((1, 2))),
    ],
)
def test_decay_index_rejects_non_vector_inputs(radius, vertical_field):
    with pytest.raises(ValueError, match="one-dimensional"):
        decay_index(radius, vertical_field)


@pytest.mark.parametrize(
    "radius",
    [
        np.array([0.8, 0.8, 1.0]),
        np.array([0.8, 1.0, 0.9]),
        np.array([0.8, np.nan, 1.0]),
        np.array([0.8, np.inf, 1.0]),
    ],
)
def test_decay_index_rejects_nonfinite_or_nonmonotonic_radius(radius):
    match = "finite" if not np.isfinite(radius).all() else "strictly monotonic"
    with pytest.raises(ValueError, match=match):
        decay_index(radius, np.full(radius.shape, -0.2))


def test_decay_index_approaches_dipole_limit_for_symmetric_distant_loops():
    """The equatorial far field of a symmetric loop pair decays as radius^-3."""
    radius = np.linspace(1.0, 2.0, 401)
    height = np.zeros_like(radius)
    upper_field, _ = greens_bz_br(radius, height, 0.1, 0.05)
    lower_field, _ = greens_bz_br(radius, height, 0.1, -0.05)
    vertical_field = upper_field + lower_field
    assert np.all(vertical_field < 0.0)
    index = decay_index(radius, vertical_field)
    np.testing.assert_allclose(index[20:-20], 3.0, atol=0.05)


def test_decay_index_window_is_the_open_rigid_displacement_interval():
    assert DECAY_INDEX_WINDOW == (0.0, 1.5)


# ---------------------------------------------------------------------------
# Shafranov virial integrals on the analytic single-null certificate rows
# ---------------------------------------------------------------------------

#: The contour/volume identity residuals are quadrature error on the grid, not
#: an identity residual, so the gate is a bound rather than an equality.  The
#: value sits above the coarsest grid the gate admits (measured 2.43e-4 of the
#: vertical moment at resolution 321) and far below the order-units size a wrong
#: sign or factor produces; the fitted order against the volume side is checked
#: separately in ``test_shafranov_contour_integral_error_falls_with_resolution``.
CONTOUR_RESIDUAL_BOUND = 5.0e-4


@dataclass(frozen=True)
class _AnalyticRow:
    """One closed-form single-null row sampled onto a common representation."""

    contour: np.ndarray
    radial_field: np.ndarray
    vertical_field: np.ndarray
    toroidal_field: np.ndarray
    volume: float
    major_radius: float
    pressure_integral: float
    poloidal_field_integral: float
    toroidal_field_integral: float
    plasma_current: float
    volume_sides: tuple[float, float, float]
    enclosed_area: float
    perimeter: float


def _analytic_row(case, sampling: int, resolution: int, boundary_f: float = 1.0):
    """Sample the analytic row: contour fields, and the volume integrals."""
    scale = case.flux_scale_per_radian_wb
    radius_0 = case.major_radius

    def flux(r, z):
        return scale * case._dimensionless_flux(r / radius_0, z / radius_0)

    def fields(r, z):
        gradient = case.gradient(np.stack((r, z), axis=-1))
        gradient_r = gradient[..., 0]
        gradient_z = gradient[..., 1]
        radial = -gradient_z / r
        vertical = gradient_r / r
        field_function = np.sqrt(
            np.maximum(boundary_f**2 + 4.0 * case.field_coefficient * flux(r, z), 0.0)
        )
        toroidal = field_function / r
        pressure = 4.0 * case.pressure_coefficient / MU0 * flux(r, z)
        return radial, vertical, toroidal, pressure

    contour = np.asarray(case.separatrix(sampling), dtype=np.float64)
    lower_r, upper_r = contour[:, 0].min(), contour[:, 0].max()
    lower_z, upper_z = contour[:, 1].min(), contour[:, 1].max()
    grid_r = np.linspace(lower_r, upper_r, resolution)
    grid_z = np.linspace(lower_z, upper_z, resolution)
    rr, zz = np.meshgrid(grid_r, grid_z, indexing="ij")
    inside = flux(rr, zz) > 0.0
    radial, vertical, toroidal, pressure = fields(rr, zz)
    total_squared = radial**2 + vertical**2 + toroidal**2
    spacing = (grid_r[1] - grid_r[0]) * (grid_z[1] - grid_z[0])
    weight = np.where(inside, 2.0 * np.pi * rr * spacing, 0.0)
    volume = weight.sum()
    source = case.grad_shafranov_source(np.stack((rr, zz), axis=-1))
    plasma_current = np.sum(np.where(inside, -source / (MU0 * rr), 0.0)) * spacing
    poloidal_field_integral = np.sum((radial**2 + vertical**2) * weight)
    return _AnalyticRow(
        contour=contour,
        radial_field=fields(contour[:, 0], contour[:, 1])[0],
        vertical_field=fields(contour[:, 0], contour[:, 1])[1],
        toroidal_field=fields(contour[:, 0], contour[:, 1])[2],
        volume=volume,
        major_radius=np.sum(rr * weight) / volume,
        pressure_integral=np.sum(pressure * weight),
        poloidal_field_integral=poloidal_field_integral,
        toroidal_field_integral=np.sum(toroidal**2 * weight),
        plasma_current=plasma_current,
        volume_sides=(
            np.sum((-(total_squared) / (2.0 * MU0) - 3.0 * pressure) * weight),
            np.sum((-(vertical**2) / MU0 - 2.0 * pressure) * weight),
            np.sum(
                (
                    ((toroidal**2 - radial**2 - vertical**2) / (2.0 * MU0) - pressure)
                    / rr
                )
                * weight
            ),
        ),
        enclosed_area=0.5
        * abs(
            np.sum(
                contour[:, 0] * np.roll(contour[:, 1], -1)
                - np.roll(contour[:, 0], -1) * contour[:, 1]
            )
        ),
        perimeter=float(
            np.sum(np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1))
        ),
    )


def _padded(array: np.ndarray, size: int) -> np.ndarray:
    """Return ``array`` in a length-``size`` exact-zero-padded buffer.

    The trailing axis is carried through so one helper pads both the ``(N, 2)``
    contour and a length-``N`` field component; padded rows are exact zeros,
    which is what the traced integral requires past ``count``.
    """
    array = np.asarray(array, dtype=np.float64)
    padded = np.zeros((size, *array.shape[1:]), dtype=np.float64)
    padded[: array.shape[0]] = array
    return padded


def _contour_integrals(row: _AnalyticRow, padding: int = 7):
    """Evaluate the traced integral on a padded copy of the row's boundary."""
    count = row.contour.shape[0]
    size = count + padding
    return shafranov_contour_integrals(
        jnp.asarray(_padded(row.contour, size)),
        jnp.asarray(_padded(row.radial_field, size)),
        jnp.asarray(_padded(row.vertical_field, size)),
        jnp.asarray(_padded(row.toroidal_field, size)),
        count,
    )


def _compact_case() -> CerfonFreidbergSingleNull:
    """Return a Solovev row shaped like a compact spherical tokamak.

    MAST-scale aspect and shaping (major radius 0.9 m, elongation 1.8,
    triangularity 0.4), so the gate exercises the shaping terms that the
    conventional-aspect-ratio reference leaves small.
    """
    return CerfonFreidbergSingleNull(
        major_radius=0.90,
        inverse_aspect_ratio=0.60,
        elongation=1.8,
        triangularity=0.40,
    )


@pytest.mark.parametrize("resolution", (321, 641, 1281))
def test_shafranov_contour_integrals_equal_their_volume_side(resolution):
    """Each contour integral reproduces the volume integral the identity gives."""
    case = cerfon_freidberg_single_null()
    row = _analytic_row(case, sampling=2001, resolution=resolution)
    count = row.contour.shape[0]
    size = count + 7
    integrals = shafranov_contour_integrals(
        jnp.asarray(_padded(row.contour, size)),
        jnp.asarray(_padded(row.radial_field, size)),
        jnp.asarray(_padded(row.vertical_field, size)),
        jnp.asarray(_padded(row.toroidal_field, size)),
        count,
    )
    vertical_side, radial_side, stress_side = row.volume_sides
    residuals = np.array(
        [
            abs(float(integrals.vertical_moment) - vertical_side) / abs(vertical_side),
            abs(float(integrals.radial_moment) - radial_side) / abs(radial_side),
            abs(float(integrals.toroidal_stress) - stress_side) / abs(stress_side),
        ]
    )
    assert residuals.max() < CONTOUR_RESIDUAL_BOUND, (
        f"resolution {resolution}: contour/volume residuals {residuals}"
    )
    assert float(integrals.enclosed_area) == pytest.approx(row.enclosed_area, rel=1e-6)
    assert float(integrals.perimeter) == pytest.approx(row.perimeter, rel=1e-6)


def test_shafranov_contour_integral_error_falls_with_resolution():
    """The contour/volume mismatch is quadrature error and shrinks with grid size."""
    case = cerfon_freidberg_single_null()
    errors = []
    resolutions = (161, 321, 641)
    for resolution in resolutions:
        row = _analytic_row(case, sampling=2001, resolution=resolution)
        count = row.contour.shape[0]
        size = count + 7
        integrals = shafranov_contour_integrals(
            jnp.asarray(_padded(row.contour, size)),
            jnp.asarray(_padded(row.radial_field, size)),
            jnp.asarray(_padded(row.vertical_field, size)),
            jnp.asarray(_padded(row.toroidal_field, size)),
            count,
        )
        errors.append(
            abs(float(integrals.vertical_moment) - row.volume_sides[0])
            / abs(row.volume_sides[0])
        )
    assert errors[-1] < errors[0], f"errors did not fall: {errors}"
    order = math.log(errors[0] / errors[-1]) / math.log(
        resolutions[-1] / resolutions[0]
    )
    assert order > 0.5, f"fitted order {order:.3f} from errors {errors}"


def test_beta_p_plus_half_internal_inductance_matches_the_volume_definition():
    """The assembly reproduces the volume combination as the grid refines.

    The combination is the difference of two nearly equal terms, twice the
    vertical moment and the toroidal volume, so its residual carries the
    volume-side quadrature error amplified by that cancellation.  The gate is
    therefore the fitted order of a falling error rather than a fixed relative
    tolerance, with a positive control showing the combination breaks outright
    when the toroidal volume is dropped.
    """
    resolutions = (201, 401, 801)
    errors = []
    for resolution in resolutions:
        row = _analytic_row(
            cerfon_freidberg_single_null(), sampling=2001, resolution=resolution
        )
        integrals = _contour_integrals(row)
        combined = float(
            beta_p_plus_half_internal_inductance(
                integrals,
                row.pressure_integral,
                row.toroidal_field_integral,
                row.major_radius,
                row.plasma_current,
            )
        )
        exact = 4.0 * row.pressure_integral / (
            MU0 * row.major_radius * row.plasma_current**2
        ) + row.poloidal_field_integral / (
            MU0**2 * row.plasma_current**2 * row.major_radius
        )
        errors.append(abs(combined - exact) / abs(exact))
    assert errors[-1] < errors[0], (
        f"the combination error did not fall with resolution {resolutions}: {errors}"
    )
    order = math.log(errors[0] / errors[-1]) / math.log(
        resolutions[-1] / resolutions[0]
    )
    assert order > 0.0, (
        f"fitted order {order:.3f} from errors {errors} at resolutions {resolutions}"
    )
    dropped = float(
        beta_p_plus_half_internal_inductance(
            integrals,
            row.pressure_integral,
            row.toroidal_field_integral * 0.0,
            row.major_radius,
            row.plasma_current,
        )
    )
    relative = abs(dropped - exact) / abs(exact)
    assert relative > 1.0, (
        "dropping the toroidal volume moved the combination from "
        f"{exact:.8f} to {dropped:.8f} (relative {relative:.3e}); the falling "
        f"error gate above is only meaningful while this control still fails; "
        f"fitted order {order:.3f} from errors {errors} at resolutions "
        f"{resolutions}"
    )


def test_shafranov_contour_integrals_trace_and_differentiate_in_the_field():
    """The integrals are traced and their field gradients are finite."""
    case = cerfon_freidberg_single_null()
    row = _analytic_row(case, sampling=721, resolution=321)
    count = row.contour.shape[0]
    size = count + 7
    contour = jnp.asarray(_padded(row.contour, size))
    radial = jnp.asarray(_padded(row.radial_field, size))
    vertical = jnp.asarray(_padded(row.vertical_field, size))
    toroidal = jnp.asarray(_padded(row.toroidal_field, size))

    def vertical_moment(radial_field, vertical_field, toroidal_field):
        return shafranov_contour_integrals(
            contour, radial_field, vertical_field, toroidal_field, count
        ).vertical_moment

    gradient = jax.grad(vertical_moment, argnums=(0, 1, 2))(radial, vertical, toroidal)
    assert all(
        bool(np.all(np.isfinite(np.asarray(component)))) for component in gradient
    )
    jitted = jax.jit(vertical_moment)(radial, vertical, toroidal)
    assert float(jitted) == pytest.approx(
        float(vertical_moment(radial, vertical, toroidal)), rel=1e-12
    )


def test_shafranov_contour_integrals_close_on_a_compact_shape():
    """The identities close at spherical-tokamak aspect and shaping.

    The conventional reference leaves the shaping terms small, so a gate run
    only on it cannot distinguish a correct identity from one that silently
    drops the shaping.  The compact row puts the aspect at 0.79, where those
    terms are large.
    """
    row = _analytic_row(_compact_case(), sampling=2001, resolution=641)
    integrals = _contour_integrals(row)
    vertical_side, radial_side, stress_side = row.volume_sides
    residuals = {
        "vertical_moment": abs(float(integrals.vertical_moment) - vertical_side)
        / abs(vertical_side),
        "radial_moment": abs(float(integrals.radial_moment) - radial_side)
        / abs(radial_side),
        "toroidal_stress": abs(float(integrals.toroidal_stress) - stress_side)
        / abs(stress_side),
    }
    assert max(residuals.values()) < CONTOUR_RESIDUAL_BOUND, (
        f"compact shape R0={row.major_radius:.3f} a_eq="
        f"{float(np.sqrt(row.enclosed_area / np.pi)):.4f}: residuals {residuals}"
    )


def test_large_aspect_ratio_formula_geometric_bias_at_compact_aspect():
    """Record the shaping term the circular formula carries at compact aspect.

    The formula is retired as a constraint side, so the number recorded here is
    its geometric bias and not a target.  Both evaluations carry the same
    bracket, taken from the formula's own argument, so the two differ only
    through the geometry each is given.
    """
    row = _analytic_row(_compact_case(), sampling=2001, resolution=401)
    exact = 4.0 * row.pressure_integral / (
        MU0 * row.major_radius * row.plasma_current**2
    ) + row.poloidal_field_integral / (
        MU0**2 * row.plasma_current**2 * row.major_radius
    )
    minor = float(np.sqrt(row.enclosed_area / np.pi))
    circular = shafranov_vertical_field(
        row.plasma_current, row.major_radius, minor, exact
    )
    elongated = shafranov_vertical_field_elongated(
        row.plasma_current, row.major_radius, minor, 1.8, exact
    )
    assert math.isfinite(circular)
    assert math.isfinite(elongated)
    bias = abs(circular - elongated) / abs(circular)
    assert bias > 0.0, (
        f"R0 {row.major_radius:.3f} m, a_equivalent {minor:.4f} m, "
        f"ln(8R/a) {math.log(8.0 * row.major_radius / minor):.4f}, "
        f"exact combination {exact:.6f}, circular {circular:.6e} T, "
        f"elongated {elongated:.6e} T, geometric bias {bias:.3e}"
    )
