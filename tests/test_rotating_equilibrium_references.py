"""Self-verifying checks on the analytic rotating-equilibrium references.

Nothing here imports the package under test. The references are asserted
against finite differences of their own flux, against independent quadrature
routes to the same integral, and against the closure algebra they claim to
satisfy, so a later solver can be benchmarked on numbers that were themselves
verified without it.
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from tests.rotating_equilibrium_references import (
    MU_0,
    RotatingEquilibrium,
    delta_star_finite_difference,
    interior_sample,
    reference_cases,
)

CASES = reference_cases()
CASE_VALUES = list(CASES.values())
CASE_IDS = list(CASES)

# Stencil widths as a fraction of the major radius. The window sits where the
# fourth-order truncation error still dominates double-precision cancellation
# in the second-difference denominator.
STENCIL_FRACTIONS = (0.04, 0.02, 0.01)
STENCIL_RESIDUAL_TOLERANCE = 2.0e-8


def _relative(actual: float, expected: float) -> float:
    """Return the relative difference between two scalars."""
    return abs(actual - expected) / abs(expected)


def _observed_order(errors: list[float]) -> list[float]:
    """Return the convergence orders implied by successive halvings."""
    return [
        math.log(coarse / fine) / math.log(2.0)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]


def _sampled_flux_labels(case: RotatingEquilibrium) -> np.ndarray:
    """Return interior flux labels spanning axis to boundary."""
    return np.linspace(0.05, 0.95, 9) * case.axis_flux


# ----------------------------------------------------------------------
# independence from the package under test
# ----------------------------------------------------------------------
def test_references_and_checks_import_no_package_under_test():
    """Both files must stay valid while the solver façade changes."""
    directory = Path(__file__).parent
    for filename in (
        "rotating_equilibrium_references.py",
        "test_rotating_equilibrium_references.py",
    ):
        tree = ast.parse((directory / filename).read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        offending = [name for name in imported if name.split(".")[0] == "nova"]
        assert offending == [], f"{filename} imports {offending}"


# ----------------------------------------------------------------------
# the equilibrium itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_analytic_grad_shafranov_residual_is_machine_zero(case):
    """The closed form must cancel term by term, not merely approximately."""
    for member in (case, case.static_limit()):
        radius, height = interior_sample(member)
        residual = np.abs(member.grad_shafranov_residual(radius, height))
        assert np.max(residual / member.source_scale(radius)) < 1.0e-14


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_finite_difference_residual_converges_at_fourth_order(case):
    """Differentiate only the flux, then confront it with the closure.

    The stencil never sees the pressure closure or the toroidal-field function,
    so this fails if the analytic source and the analytic solution disagree, and
    it fails at the wrong rate if either is subtly non-smooth.
    """
    radius, height = interior_sample(case)
    scale = case.source_scale(radius)
    errors = []
    for fraction in STENCIL_FRACTIONS:
        spacing = fraction * case.major_radius
        operator = delta_star_finite_difference(case, radius, height, spacing)
        residual = (
            operator
            + MU_0 * radius**2 * case.pressure_flux_derivative(radius)
            + case.f_f_prime
        )
        errors.append(float(np.max(np.abs(residual) / scale)))
    assert errors[-1] < STENCIL_RESIDUAL_TOLERANCE
    for order in _observed_order(errors):
        assert 3.7 < order < 4.3


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_toroidal_current_from_the_operator_matches_the_source_form(case):
    """``-DeltaStar(psi) / (mu0 R)`` and ``R p' + F F' / (mu0 R)`` agree."""
    radius, height = interior_sample(case)
    from_operator = case.toroidal_current_density(radius, height)
    from_source = case.toroidal_current_source(radius)
    assert np.allclose(from_operator, from_source, rtol=1.0e-14, atol=0.0)


# ----------------------------------------------------------------------
# the static limit
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_static_limit_reproduces_the_solovev_closed_form(case):
    """A vanishing rotation parameter must reach Solov'ev with no separate path."""
    static = case.static_limit()
    radius, height = interior_sample(static)
    label = radius**2 - static.major_radius**2
    expected = (
        static.axis_flux
        - 0.5 * static.pressure_coefficient * label**2
        - static.field_coefficient * height**2
    )
    assert np.allclose(static.flux(radius, height), expected, rtol=1.0e-15, atol=0.0)
    operator = -4.0 * static.pressure_coefficient * radius**2 - 2.0 * (
        static.field_coefficient
    )
    assert np.allclose(
        static.delta_star(radius, height), operator, rtol=1.0e-15, atol=0.0
    )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_flux_approaches_the_static_limit_linearly_in_the_rotation_parameter(case):
    """The centrifugal factor enters at first order in ``theta``."""
    static = case.static_limit()
    radius, height = interior_sample(case)
    reference = static.flux(radius, height)
    errors = []
    for divisor in (1.0, 2.0, 4.0):
        rotated = case.with_rotation_parameter(case.rotation_parameter / divisor)
        departure = np.abs(rotated.flux(radius, height) - reference)
        errors.append(float(np.max(departure) / case.axis_flux))
    for order in _observed_order(errors):
        assert 0.9 < order < 1.1


# ----------------------------------------------------------------------
# the isothermal closure
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_pressure_major_radius_factor_follows_the_isothermal_closure(case):
    """Pressure must vary along a flux surface exactly as the closure says.

    Three statements, each independently sufficient to catch a wrong exponent:
    the explicit exponential factor, the logarithmic derivative at fixed flux,
    and the radial force balance the exponential was integrated from.
    """
    flux_values = _sampled_flux_labels(case)
    inboard, outboard = case.boundary_midplane_radii()
    radius = np.linspace(inboard, outboard, 11)
    label = radius**2 - case.major_radius**2
    for flux_value in flux_values:
        along = case.pressure_at(flux_value, radius)
        at_reference = case.pressure_at(flux_value, case.major_radius)
        expected = np.exp(case.rotation_parameter * label)
        assert np.allclose(along / at_reference, expected, rtol=1.0e-14, atol=0.0)

        spacing = 1.0e-4 * case.major_radius
        forward = np.log(case.pressure_at(flux_value, radius + spacing))
        backward = np.log(case.pressure_at(flux_value, radius - spacing))
        logarithmic = (forward - backward) / (2.0 * spacing)
        assert np.allclose(
            logarithmic, 2.0 * case.rotation_parameter * radius, rtol=1.0e-7, atol=0.0
        )

        density = case.mass_density_at(flux_value, radius)
        centrifugal = density * radius * case.angular_frequency(flux_value) ** 2
        assert np.allclose(
            2.0 * case.rotation_parameter * radius * along,
            centrifugal,
            rtol=1.0e-14,
            atol=0.0,
        )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_thermal_mach_number_is_uniform_across_flux_surfaces(case):
    """The exactly solvable branch is the uniform-thermal-Mach-number branch."""
    flux_values = _sampled_flux_labels(case)
    thermal_speed = np.sqrt(
        2.0 * case.temperature(flux_values) / case.mean_particle_mass
    )
    local = case.angular_frequency(flux_values) * case.major_radius / thermal_speed
    assert np.allclose(local, case.thermal_mach_number, rtol=1.0e-14, atol=0.0)
    # The temperature genuinely varies, so uniformity is a statement about the
    # pairing of rotation with temperature and not about a constant profile.
    assert case.temperature(case.axis_flux) > 5.0 * case.temperature(0.0)
    assert case.angular_frequency(case.axis_flux) > 2.0 * case.angular_frequency(0.0)


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_low_mach_truncation_leaves_a_quadratic_rotation_remainder(case):
    """A free quartic-radius coefficient is the truncation, not the closure.

    Keeping the first rotation correction reproduces the familiar quartic term
    with coefficient ``mu0 p_1 theta``; what it drops is second order in
    ``theta``, so the truncation is a genuine approximation of this family
    rather than a reparametrisation of it.
    """
    radius, height = interior_sample(case)
    label = radius**2 - case.major_radius**2
    amplitude = MU_0 * case.pressure_flux_gradient * radius**2
    errors = []
    for divisor in (1.0, 2.0, 4.0):
        rotation = case.rotation_parameter / divisor
        rotated = case.with_rotation_parameter(rotation)
        exact = -MU_0 * radius**2 * rotated.pressure_flux_derivative(radius)
        truncated = -amplitude * (1.0 + rotation * label)
        errors.append(float(np.max(np.abs(exact - truncated)) / np.max(amplitude)))
    for order in _observed_order(errors):
        assert 1.9 < order < 2.1

    small = case.with_rotation_parameter(1.0e-6 * case.rotation_parameter)
    static = case.static_limit()
    difference = (
        -MU_0 * radius**2 * small.pressure_flux_derivative(radius)
        + MU_0 * radius**2 * static.pressure_flux_derivative(radius)
    )
    quartic = -amplitude * small.rotation_parameter * label
    assert np.allclose(difference, quartic, rtol=1.0e-5, atol=0.0)


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_density_and_pressure_pile_up_on_the_outboard_side(case):
    """Centrifugal accumulation is the same exponential for both."""
    inboard, outboard = case.boundary_midplane_radii()
    flux_value = 0.5 * case.axis_flux
    expected = math.exp(case.rotation_parameter * (outboard**2 - inboard**2))
    pressure_ratio = float(
        case.pressure_at(flux_value, outboard) / case.pressure_at(flux_value, inboard)
    )
    density_ratio = float(
        case.mass_density_at(flux_value, outboard)
        / case.mass_density_at(flux_value, inboard)
    )
    assert _relative(pressure_ratio, expected) < 1.0e-14
    assert _relative(density_ratio, expected) < 1.0e-14
    assert pressure_ratio > 1.0


# ----------------------------------------------------------------------
# orientation and sign conventions
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_orientation_and_sign_conventions_are_pinned(case):
    """Fix every sign a convention pin would have to agree with."""
    radius, height = interior_sample(case)
    axis_radius, axis_height = case.magnetic_axis

    assert case.axis_flux > 0.0
    assert float(case.flux(axis_radius, axis_height)) == pytest.approx(case.axis_flux)
    assert np.all(case.flux(radius, height) < case.axis_flux)
    inboard, outboard = case.boundary_midplane_radii()
    assert abs(float(case.flux(outboard, 0.0))) < 1.0e-14 * case.axis_flux

    current = case.toroidal_current_density(radius, height)
    assert np.all(current > 0.0)
    assert np.allclose(
        MU_0 * current,
        -case.delta_star(radius, height) / radius,
        rtol=1.0e-14,
        atol=0.0,
    )

    probe = np.array([0.5 * (inboard + axis_radius), 0.5 * (axis_radius + outboard)])
    _, toroidal_field, vertical_field = case.magnetic_field(probe, 0.0)
    assert vertical_field[0] > 0.0
    assert vertical_field[1] < 0.0
    assert np.all(toroidal_field > 0.0)
    assert np.all(case.f_function(np.linspace(0.0, case.axis_flux, 5)) > 0.0)

    above, _, _ = case.magnetic_field(axis_radius, 0.25 * math.sqrt(
        case.axis_flux / case.field_coefficient
    ))
    assert float(above) > 0.0


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_magnetic_axis_is_exact_and_independent_of_rotation(case):
    """The axis stays at the reference radius at every Mach number."""
    for member in (case, case.static_limit(), case.with_rotation_parameter(4.0e-2)):
        radial, vertical = member.flux_gradient(*member.magnetic_axis)
        assert float(radial) == 0.0
        assert float(vertical) == 0.0
    offset = 1.0e-3 * case.major_radius
    for shift in (-offset, offset):
        assert float(case.flux(case.major_radius + shift, 0.0)) < case.axis_flux
        assert float(case.flux(case.major_radius, shift)) < case.axis_flux


# ----------------------------------------------------------------------
# derived invariants
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_boundary_radii_match_the_lambert_closed_form(case):
    """The two real Lambert branches locate the midplane boundary."""
    inboard_label, outboard_label = case.boundary_flux_labels_closed_form()
    closed_form = (
        math.sqrt(case.major_radius**2 + inboard_label),
        math.sqrt(case.major_radius**2 + outboard_label),
    )
    refined = case.boundary_midplane_radii()
    for from_lambert, from_newton in zip(closed_form, refined, strict=True):
        assert _relative(from_lambert, from_newton) < 1.0e-9
    for radius in refined:
        assert abs(float(case.flux(radius, 0.0))) < 1.0e-14 * case.axis_flux
    assert refined[0] < case.major_radius < refined[1]


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_axis_safety_factor_matches_the_contour_quadrature(case):
    """The elliptic closed form is the limit of the surface contour integral."""
    departures = []
    for offset in (1.0e-3, 1.0e-4, 1.0e-5):
        surface = case.safety_factor((1.0 - offset) * case.axis_flux)
        departures.append(_relative(surface, case.axis_safety_factor))
    assert departures[-1] < 1.0e-5
    for order in _observed_order(departures[::-1]):
        assert order < -2.5  # a decade of offset buys a decade of agreement


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_axis_safety_factor_and_elongation_are_unchanged_by_rotation(case):
    """Both are set at the axis, where the centrifugal factor is unity."""
    static = case.static_limit()
    assert case.axis_safety_factor == pytest.approx(
        static.axis_safety_factor, rel=1.0e-15
    )
    assert case.axis_elongation == pytest.approx(static.axis_elongation, rel=1.0e-15)


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_plasma_current_matches_amperes_law(case):
    """The current-density volume integral and the boundary loop integral agree."""
    assert case.plasma_current() == pytest.approx(
        case.plasma_current_from_ampere_law(), rel=1.0e-10
    )


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_surface_quadratures_are_node_independent(case):
    """The cosine map makes every integrand smooth and periodic in the angle."""
    for quantity in (
        case.plasma_current,
        case.plasma_volume,
        case.boundary_perimeter,
        case.pressure_volume_integral,
        case.poloidal_field_volume_integral,
    ):
        assert quantity(128) == pytest.approx(quantity(512), rel=1.0e-12)


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_rotation_increases_the_outward_shafranov_shift(case):
    """The outboard boundary is pulled in harder than the inboard one.

    At fixed source coefficients both midplane boundary labels move inward by
    ``-theta psi_a / (3 k_p)`` to first order, so the geometric centre falls
    while the axis stays put and the shift grows linearly in ``theta``.
    """
    static_shift = case.static_limit().shafranov_shift
    assert case.shafranov_shift > static_shift > 0.0

    growth = []
    for divisor in (1.0, 2.0, 4.0):
        rotation = case.rotation_parameter / divisor
        rotated = case.with_rotation_parameter(rotation)
        growth.append(rotated.shafranov_shift - static_shift)
    assert growth[0] > growth[1] > growth[2] > 0.0
    for order in _observed_order(growth):
        assert 0.9 < order < 1.1

    # The coefficient is an expansion about zero rotation, so it is built from
    # the static boundary radii; using the rotating ones mixes in the very
    # displacement being predicted.
    inboard, outboard = case.static_limit().boundary_midplane_radii()
    small = case.with_rotation_parameter(1.0e-3 * case.rotation_parameter)
    predicted = (
        small.rotation_parameter
        * case.axis_flux
        / (3.0 * case.pressure_coefficient)
        * (1.0 / inboard + 1.0 / outboard)
        / 4.0
    )
    assert _relative(small.shafranov_shift - static_shift, predicted) < 1.0e-3


@pytest.mark.parametrize("case", CASE_VALUES, ids=CASE_IDS)
def test_shape_anchors_survive_a_mach_number_rebuild(case):
    """Rebuilding at another Mach number holds the geometry and axis pressure."""
    half_height = math.sqrt(case.axis_flux / case.field_coefficient)
    _, outboard = case.boundary_midplane_radii()
    for mach_number in (0.0, 0.5 * case.thermal_mach_number, case.thermal_mach_number):
        rebuilt = case.with_thermal_mach_number(mach_number)
        assert rebuilt.thermal_mach_number == pytest.approx(mach_number, abs=1.0e-15)
        assert rebuilt.boundary_midplane_radii()[1] == pytest.approx(
            outboard, rel=1.0e-12
        )
        assert math.sqrt(
            rebuilt.axis_flux / rebuilt.field_coefficient
        ) == pytest.approx(half_height, rel=1.0e-12)
        assert rebuilt.axis_pressure == pytest.approx(case.axis_pressure, rel=1.0e-12)
    identical = case.with_thermal_mach_number(case.thermal_mach_number)
    assert identical.pressure_coefficient == pytest.approx(
        case.pressure_coefficient, rel=1.0e-12
    )


# ----------------------------------------------------------------------
# a canonical case, pinned
# ----------------------------------------------------------------------
def test_reactor_scale_case_reproduces_its_derived_invariants():
    """Pin the reactor-scale member against the closed forms it is built from.

    Geometry and axis quantities are exact closed forms: the axis sits at the
    reference radius, ``psi_a = k_p u_out^2 phi2(theta u_out)``, the boundary
    radii come from the Lambert branches, and ``q_axis`` and ``kappa_axis`` are
    the elliptic near-axis forms. The integral moments are the surface
    quadratures listed in the reference module, all tied to the perimeter-based
    poloidal field scale.
    """
    case = CASES["weak-rotation-reactor"]
    inboard, outboard = case.boundary_midplane_radii()

    assert case.thermal_mach_number == pytest.approx(0.10, abs=1.0e-15)
    assert case.rotation_parameter == pytest.approx(2.601456816e-4, rel=1.0e-9)
    assert case.axis_flux == pytest.approx(7.94831273522, rel=1.0e-9)
    assert case.pressure_coefficient == pytest.approx(3.16202218794e-2, rel=1.0e-9)
    assert case.field_coefficient == pytest.approx(0.613295735742, rel=1.0e-9)
    assert case.pressure_flux_gradient == pytest.approx(1.006502923e5, rel=1.0e-9)
    assert case.f_f_prime == pytest.approx(1.226591471, rel=1.0e-9)
    assert case.axis_pressure == pytest.approx(8.0e5, rel=1.0e-12)

    assert case.magnetic_axis == (6.2, 0.0)
    assert inboard == pytest.approx(3.99955061178, rel=1.0e-9)
    assert outboard == pytest.approx(7.8, rel=1.0e-12)
    assert case.geometric_axis_radius == pytest.approx(5.8997753059, rel=1.0e-9)
    assert case.minor_radius == pytest.approx(1.9002246941, rel=1.0e-9)
    assert case.shafranov_shift == pytest.approx(0.3002246941, rel=1.0e-8)
    assert case.axis_elongation == pytest.approx(1.99092166, rel=1.0e-8)
    assert case.axis_safety_factor == pytest.approx(2.189817989, rel=1.0e-9)
    assert float(case.f_function(case.axis_flux)) == pytest.approx(
        33.1553658, rel=1.0e-8
    )

    assert case.plasma_current() == pytest.approx(1.6316711493e7, rel=1.0e-9)
    assert case.plasma_volume() == pytest.approx(796.65918674, rel=1.0e-9)
    assert case.boundary_perimeter() == pytest.approx(17.6367116, rel=1.0e-8)
    assert case.mean_pressure == pytest.approx(3.999999055e5, rel=1.0e-9)
    assert case.beta_toroidal == pytest.approx(3.5788872e-2, rel=1.0e-7)
    assert case.beta_poloidal == pytest.approx(0.74378939, rel=1.0e-7)
    assert case.internal_inductance == pytest.approx(0.47189633, rel=1.0e-7)

    assert float(case.number_density(6.2, 0.0)) == pytest.approx(4.99321e20, rel=1.0e-5)
    axis_speed = float(case.angular_frequency(case.axis_flux)) * case.major_radius
    assert axis_speed == pytest.approx(9.78958e4, rel=1.0e-5)

    static = case.static_limit()
    assert static.axis_safety_factor == pytest.approx(
        case.axis_safety_factor, rel=1.0e-15
    )
    assert static.shafranov_shift == pytest.approx(0.29816345, rel=1.0e-7)
    assert static.plasma_current() == pytest.approx(1.6314773313e7, rel=1.0e-9)
