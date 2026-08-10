"""Contract for the source-free toroidal-harmonic flux basis.

The basis makes three claims and each is checked against something that does not
share its derivation.

* The ring functions are what they say they are.  Both radial ladders are checked
  against a quadrature of the Legendre integral representation, which knows
  nothing about elliptic integrals or recurrences, over the range of ``cosh eta``
  a real sensor set reaches.  The second kind is the recessive solution of the
  degree recurrence, so the forward climb that serves the first kind destroys it;
  that failure is asserted here rather than assumed, because it is the entire
  reason the second kind is built backward.
* The columns solve the homogeneous Grad-Shafranov operator.  Checked by finite
  differences of the columns themselves on a raster, away from each family's own
  singular set -- an oracle independent of the separation the columns came from.
  The analytic field is checked the same way, against differences of the flux.
* A filament's expansion is exact.  Checked against the ring Green's function of
  :mod:`nova.biot.greens`, on both sides of the source's own focal circle, and the
  position read back off those coefficients is checked against where the filament
  was put.

The recovery checks run on the real MAST pickup poses and sensitive axes, listed
below, so the resolution they report is that array's and not a convenient one.
The pair-failure channel is excluded exactly as a real read excludes it.
"""

import numpy as np
import pytest
import scipy.integrate

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot import toroidalharmonic as th

# --- the real pickup array --------------------------------------------------

CENTRE_RADIUS = 0.1803
"""Radius of the centre-column pickup stack [m]; all of it reads vertical field."""

CENTRE_HEIGHT = (
    1.4487,
    1.3725,
    1.2962,
    1.2200,
    1.1438,
    1.0675,
    0.9912,
    0.9150,
    0.8387,
    0.7589,
    0.6863,
    0.6100,
    0.5337,
    0.4575,
    0.3812,
    0.3050,
    0.2288,
    0.1525,
    0.0762,
    0.0000,
    -0.0762,
    -0.1525,
    -0.2288,
    -0.3050,
    -0.3812,
    -0.4575,
    -0.5337,
    -0.6100,
    -0.6863,
    -0.7625,
    -0.8387,
    -0.9150,
    -0.9912,
    -1.0675,
    -1.1438,
    -1.2200,
    -1.2962,
    -1.3725,
    -1.4487,
    -1.5250,
)

# radius, radial-reading height, vertical-reading height.  The two orientations
# share a pose everywhere except one station, where they are 3 mm apart.
OUTBOARD = (
    (1.4420, 1.3350, 1.3350),
    (1.4420, 1.2600, 1.2600),
    (1.5897, 0.8105, 0.8105),
    (1.5897, 0.7355, 0.7355),
    (1.5897, 0.6604, 0.6604),
    (1.8449, 0.3070, 0.3070),
    (1.8449, 0.2320, 0.2320),
    (1.8449, 0.1570, 0.1570),
    (1.8449, 0.0820, 0.0820),
    (1.8449, 0.0070, 0.0040),
    (1.8449, -0.0680, -0.0680),
    (1.8449, -0.1430, -0.1430),
    (1.8449, -0.2180, -0.2180),
    (1.8449, -0.2930, -0.2930),
    (1.5913, -0.6404, -0.6404),
    (1.5913, -0.7154, -0.7154),
    (1.5913, -0.7904, -0.7904),
    (1.4401, -1.2602, -1.2602),
    (1.4401, -1.3352, -1.3352),
)

FAULTED_STATION = 16
"""Outboard station whose radial channel is a known pair failure, excluded."""

SENSOR_FLOOR = 387.0e-6
"""Measured field-probe noise with nothing driven [T]."""

FIRST_KIND_REFERENCE_FLOOR = 1.0e-7
"""Accuracy the quadrature reference reaches on the dominant ladder."""

SECOND_KIND_REFERENCE_FLOOR = 2.0e-5
"""Accuracy the quadrature reference reaches on the recessive ladder.

Weaker by two decades, and it is the REFERENCE that is weak: by degree eight the
second kind has decayed twelve decades below the first, so differencing two
quadrature values of it is differencing their trailing digits.  Where the two
routes are compared against each other instead -- the ratio chain against its own
closed-form seed -- they agree to nine digits.
"""

SURVIVING_AMPLITUDE = 6.7e-3
"""Amplitude of the surviving misfit at the largest drive [T]."""


def pickup_array():
    """Return ``(R, Z, cos, sin)`` of the trusted pickup channels."""
    radius = [CENTRE_RADIUS] * len(CENTRE_HEIGHT)
    height = list(CENTRE_HEIGHT)
    cosine = [0.0] * len(CENTRE_HEIGHT)
    sine = [1.0] * len(CENTRE_HEIGHT)
    for station, (station_r, radial_z, vertical_z) in enumerate(OUTBOARD):
        if station != FAULTED_STATION:
            radius.append(station_r)
            height.append(radial_z)
            cosine.append(1.0)
            sine.append(0.0)
        radius.append(station_r)
        height.append(vertical_z)
        cosine.append(0.0)
        sine.append(1.0)
    return tuple(
        np.asarray(part, dtype=float) for part in (radius, height, cosine, sine)
    )


def projected_field(r, z, cosine, sine, sources):
    """Return the axis-projected field of ``(R, Z, current)`` filaments [T]."""
    out = np.zeros_like(r)
    for source_r, source_z, current in sources:
        bz, br = greens_bz_br(r, z, source_r, source_z)
        out += current * (cosine * br + sine * bz)
    return out


def matched_current(r, z, cosine, sine, source):
    """Return the current putting a filament at the surviving-misfit amplitude."""
    unit = projected_field(r, z, cosine, sine, [(*source, 1.0)])
    return SURVIVING_AMPLITUDE / float(np.sqrt(np.mean(unit**2)))


def offset_focus(source, offset, bearing=0.5 * np.pi):
    """Return a focal circle displaced from a source by ``offset`` metres."""
    return th.FocalCircle(
        source[0] + offset * np.cos(bearing), source[1] + offset * np.sin(bearing)
    )


# --- references that share no derivation with the module --------------------


def quadrature_legendre(nu, x, second_kind):
    """Return ``P_nu(x)`` or ``Q_nu(x)`` from the Legendre integral representation."""
    span = np.sqrt(x * x - 1.0)
    if second_kind:

        def decaying_integrand(t):
            """Evaluate the infinite-range tail without forming ``cosh(t)``."""
            magnitude = abs(t)
            log_cosh = magnitude + np.log1p(np.exp(-2.0 * magnitude)) - np.log(2.0)
            log_denominator = np.logaddexp(np.log(x), np.log(span) + log_cosh)
            return np.exp((-nu - 1.0) * log_denominator)

        value, _ = scipy.integrate.quad(decaying_integrand, 0.0, np.inf, limit=400)
        return value
    value, _ = scipy.integrate.quad(
        lambda t: (x + span * np.cos(t)) ** nu, 0.0, np.pi, limit=400
    )
    return value / np.pi


def quadrature_order_one(n, x, second_kind, relative_step=1.0e-5):
    """Return ``F^1_{n-1/2}(x) = sqrt(x^2-1) dF/dx`` from the quadrature reference.

    The difference step scales with ``x`` because the degree functions vary on the
    scale of their argument: a fixed absolute step differences two quadrature
    values that agree in their first ten digits once ``x`` reaches a few hundred,
    and the reference then measures its own round-off rather than the function.
    """
    nu = n - 0.5
    step = relative_step * x
    return (
        np.sqrt(x * x - 1.0)
        * (
            quadrature_legendre(nu, x + step, second_kind)
            - quadrature_legendre(nu, x - step, second_kind)
        )
        / (2.0 * step)
    )


def forward_second_kind(order, x):
    """Climb the second kind FORWARD, the arrangement the module rejects."""
    ladder = list(th.ring_legendre_second(1, np.asarray(x, dtype=float))[0])
    for n in range(1, order):
        ladder.append((2.0 * n * x * ladder[n] - (n + 0.5) * ladder[n - 1]) / (n - 0.5))
    return np.stack(ladder[: order + 1])


LADDER_ARGUMENTS = [1.05, 1.3, 2.0, 5.0, 30.0, 300.0]
"""``cosh eta`` over the span a focal circle inside a machine reaches."""

# Rounded float64 projections of 100-decimal associated Legendre values.  These
# pin cancellation and recurrence accuracy without sharing the implementation's
# elliptic-integral seeds or either recurrence direction.
BOUNDARY_FIRST_REFERENCE = np.asarray(
    [
        -0.0005590152474536093,
        0.0016770499349725635,
        0.00838531256424848,
        0.01956597388804339,
        0.03521936932472958,
        0.05534596847176781,
        0.07994637511744725,
        0.10902132725598004,
        0.1425716971059511,
    ]
)
BOUNDARY_SECOND_REFERENCE = np.asarray(
    [
        -223.61070489318686,
        -223.5961943239923,
        -223.56160646418158,
        -223.50992167400275,
        -223.44292720587256,
        -223.3618987280413,
        -223.2678275195526,
        -223.16152375303002,
        -223.04367211678482,
    ]
)

LARGE_ARGUMENTS = np.asarray([30.0, 30_000.0])
LARGE_FIRST_REFERENCE = np.asarray(
    [
        [-0.14309881872246607, 2.4608407008521786, 3.2885120057541387e13],
        [-0.013499662888983075, 77.96967974797411, 1.0419746769180037e36],
    ]
)
LARGE_FIRST_GRADIENT = np.asarray([0.0010189787879987784, 1.8167789293604108e-7])
LARGE_SECOND_REFERENCE = np.asarray(
    [
        [-0.20288759422205875, -0.005073070811188697, -4.040899536419433e-15],
        [-0.006412749153926629, -1.6031872887599884e-7, -1.2746231075955195e-40],
    ]
)
LARGE_SECOND_GRADIENT = np.asarray([0.0033880427683386825, 1.068791527732644e-7])


# --- coordinates ------------------------------------------------------------


def test_focal_coordinates_invert_the_forward_map():
    """The frame's coordinates map back onto the points they came from."""
    focus = th.FocalCircle(1.1, -0.2)
    rng = np.random.default_rng(0)
    r = rng.uniform(0.15, 2.0, 200)
    z = rng.uniform(-1.6, 1.6, 200)
    frame = th.focal_frame(r, z, focus)
    back_r, back_z = th.focal_position(focus, frame.distance, frame.angle)
    assert np.allclose(back_r, r, rtol=1e-12, atol=1e-12)
    assert np.allclose(back_z, z, rtol=1e-12, atol=1e-12)


def test_focal_position_retains_tiny_distance_and_angle_denominators():
    """The sum-of-squares gap resolves either approach to coordinate infinity."""
    focus = th.FocalCircle(1.3, -0.2)
    tiny = 1.0e-8
    along_distance_r, along_distance_z = th.focal_position(focus, tiny, 0.0)
    along_angle_r, along_angle_z = th.focal_position(focus, 0.0, tiny)
    assert float(along_distance_r) == pytest.approx(
        focus.radius / np.tanh(0.5 * tiny), rel=2e-16
    )
    assert float(along_distance_z) == focus.height
    assert float(along_angle_r) == 0.0
    assert float(along_angle_z) == pytest.approx(
        focus.height + focus.radius / np.tan(0.5 * tiny), rel=2e-16
    )


def test_focal_position_rejects_the_path_dependent_point_at_infinity():
    """Exactly zero in both coordinates has no single finite inverse image."""
    with pytest.raises(ValueError, match="path-dependent point at infinity"):
        th.focal_position(th.FocalCircle(1.0), 0.0, 0.0)


def test_focal_position_broadcasts_distance_and_angle():
    """Scalar distance and vector angle return one coordinate per angle."""
    radius, height = th.focal_position(
        th.FocalCircle(1.0), 0.2, np.asarray([0.1, 0.3, 0.5])
    )
    assert radius.shape == height.shape == (3,)


def test_focal_gap_is_the_focal_distance_product():
    """The gap is twice the squared focal radius over the focal-distance product."""
    focus = th.FocalCircle(0.9, 0.3)
    rng = np.random.default_rng(1)
    r = rng.uniform(0.2, 2.0, 200)
    z = rng.uniform(-1.5, 1.5, 200)
    frame = th.focal_frame(r, z, focus)
    near = np.hypot(r - focus.radius, z - focus.height)
    far = np.hypot(r + focus.radius, z - focus.height)
    assert np.allclose(frame.gap, 2.0 * focus.radius**2 / (near * far), rtol=1e-12)
    assert np.allclose(frame.gap, frame.cosine - np.cos(frame.angle), rtol=1e-11)


def test_focal_jacobian_matches_finite_differences_and_is_conformal():
    """The two stored partials carry all four, and satisfy the conformal relations."""
    focus = th.FocalCircle(1.0, 0.0)
    rng = np.random.default_rng(2)
    r = rng.uniform(0.4, 1.8, 80)
    z = rng.uniform(-1.0, 1.0, 80)
    step = 1.0e-6
    frame = th.focal_frame(r, z, focus)

    def coordinates(shift_r, shift_z):
        moved = th.focal_frame(r + shift_r, z + shift_z, focus)
        return moved.distance, moved.angle

    by_radius = (coordinates(step, 0.0)[0] - coordinates(-step, 0.0)[0]) / (2.0 * step)
    by_height = (coordinates(0.0, step)[0] - coordinates(0.0, -step)[0]) / (2.0 * step)
    angle_by_radius = (coordinates(step, 0.0)[1] - coordinates(-step, 0.0)[1]) / (
        2.0 * step
    )
    angle_by_height = (coordinates(0.0, step)[1] - coordinates(0.0, -step)[1]) / (
        2.0 * step
    )
    assert np.allclose(frame.radial_gradient, by_radius, rtol=1e-6)
    assert np.allclose(frame.height_gradient, by_height, rtol=1e-6)
    # conformality: the angle's partials are the distance's, swapped and signed
    assert np.allclose(angle_by_radius, frame.height_gradient, rtol=1e-6)
    assert np.allclose(angle_by_height, -frame.radial_gradient, rtol=1e-6)


def test_focal_frame_survives_a_point_on_the_focal_circle():
    """The coordinate singularity returns a huge finite value, never a non-number."""
    focus = th.FocalCircle(1.45, -1.08)
    frame = th.focal_frame(
        np.array([focus.radius, 1.0]), np.array([focus.height, 0.0]), focus
    )
    assert np.all(np.isfinite(frame.cosine))
    assert frame.cosine[0] > 1.0e6


# --- the radial ladders -----------------------------------------------------


@pytest.mark.parametrize("x", LADDER_ARGUMENTS)
@pytest.mark.parametrize("second_kind", [False, True])
def test_ring_ladders_match_the_integral_representation(x, second_kind):
    """Both order-one ladders reproduce a quadrature of the Legendre integral."""
    order = 8
    ladder = (th.ring_legendre_second if second_kind else th.ring_legendre_first)(
        order, np.array([x])
    )[0][:, 0]
    reference = np.array(
        [quadrature_order_one(n, x, second_kind) for n in range(order + 1)]
    )
    floor = SECOND_KIND_REFERENCE_FLOOR if second_kind else FIRST_KIND_REFERENCE_FLOOR
    assert np.allclose(ladder, reference, rtol=floor)


def test_forward_second_kind_climb_is_destroyed_by_the_dominant_solution():
    """The rejected arrangement is asserted to fail, since it motivates the design."""
    x = np.array([30.0])
    order = 8
    backward = th.ring_legendre_second(order, x)[0][:, 0]
    forward = forward_second_kind(order, x)[:, 0]
    reference = quadrature_order_one(order, float(x[0]), True)
    assert abs(backward[order] / reference - 1.0) < SECOND_KIND_REFERENCE_FLOOR
    assert abs(forward[order] / reference - 1.0) > 1.0e3


@pytest.mark.parametrize(
    "ladder, reference",
    [
        (th.ring_legendre_first, BOUNDARY_FIRST_REFERENCE),
        (th.ring_legendre_second, BOUNDARY_SECOND_REFERENCE),
    ],
)
def test_radial_domain_boundary_matches_a_hundred_decimal_arbiter(ladder, reference):
    """The accepted boundary has a measured accuracy contract through degree eight."""
    x = np.asarray([1.0 + th.MINIMUM_COSH_GAP])
    value = ladder(8, x)[0][:, 0]
    assert np.max(np.abs(value / reference - 1.0)) < 1.0e-9


@pytest.mark.parametrize("ladder", [th.ring_legendre_first, th.ring_legendre_second])
@pytest.mark.parametrize("x", [1.0, np.nan, np.inf])
def test_radial_ladders_reject_the_axis_boundary_and_nonfinite_arguments(ladder, x):
    """No held coordinate stands in for a divergent or unsupported value."""
    with pytest.raises(ValueError, match="x - 1"):
        ladder(0, np.asarray([x]))


@pytest.mark.parametrize("ladder", [th.ring_legendre_first, th.ring_legendre_second])
def test_radial_ladders_reject_just_below_the_public_boundary(ladder):
    """The documented boundary is inclusive and the preceding float is not."""
    boundary = 1.0 + th.MINIMUM_COSH_GAP
    below = np.nextafter(boundary, 1.0)
    with pytest.raises(ValueError, match="x - 1"):
        ladder(8, np.asarray([below]))


@pytest.mark.parametrize("ladder", [th.ring_legendre_first, th.ring_legendre_second])
def test_order_zero_ladders_keep_the_reflected_adjacent_value(ladder):
    """The reflected degree supplies the value derivative even when not returned."""
    x = np.asarray([1.05, 30.0, 30_000.0])
    value, gradient = ladder(0, x)
    adjacent_value, adjacent_gradient = ladder(1, x)
    assert value.shape == gradient.shape == (1, x.size)
    assert np.array_equal(value, adjacent_value[:1])
    assert np.array_equal(gradient, adjacent_gradient[:1])
    span = (x - 1.0) * (x + 1.0)
    reflected = (-0.5 * x * adjacent_value[0] - 0.5 * adjacent_value[1]) / span
    assert np.array_equal(gradient[0], reflected)


@pytest.mark.parametrize(
    "ladder, reference, gradient_reference",
    [
        (th.ring_legendre_first, LARGE_FIRST_REFERENCE, LARGE_FIRST_GRADIENT),
        (th.ring_legendre_second, LARGE_SECOND_REFERENCE, LARGE_SECOND_GRADIENT),
    ],
)
def test_large_argument_ladders_match_hundred_decimal_values(
    ladder, reference, gradient_reference
):
    """Degrees zero, one and eight pin the dominant and recessive large-x limits."""
    value, gradient = ladder(8, LARGE_ARGUMENTS)
    selected = value[[0, 1, 8]].T
    assert np.max(np.abs(selected / reference - 1.0)) < 5.0e-13
    assert np.max(np.abs(gradient[0] / gradient_reference - 1.0)) < 5.0e-13


@pytest.mark.parametrize("second_kind", [False, True])
def test_ladder_derivatives_match_finite_differences(second_kind):
    """The returned ``x`` derivative is the derivative of the returned ladder."""
    ladder = th.ring_legendre_second if second_kind else th.ring_legendre_first
    x = np.asarray([1.2, 1.8, 4.0, 20.0])
    step = 1.0e-7
    value, gradient = ladder(6, x)
    difference = (ladder(6, x + step)[0] - ladder(6, x - step)[0]) / (2.0 * step)
    assert np.allclose(gradient, difference, rtol=1e-5)
    assert value.shape == gradient.shape == (7, x.size)


# --- the basis --------------------------------------------------------------


def test_frame_keeps_the_axis_boundary_while_basis_evaluation_rejects_it():
    """Coordinates report eta zero exactly; radial values do not mask the boundary."""
    focus = th.FocalCircle(1.0, 0.0)
    frame = th.focal_frame(np.asarray([0.0]), np.asarray([0.0]), focus)
    assert float(frame.cosine[0]) == 1.0
    assert float(frame.sine[0]) == 0.0
    for family in (th.INNER, th.OUTER):
        basis = th.ToroidalHarmonics(focus, order=0, families=(family,))
        with pytest.raises(ValueError, match="x - 1"):
            basis.flux(np.asarray([0.0]), np.asarray([0.0]))
        with pytest.raises(ValueError, match="x - 1"):
            basis.field(np.asarray([0.0]), np.asarray([0.0]))


@pytest.mark.parametrize("family", [th.INNER, th.OUTER])
def test_columns_solve_the_homogeneous_operator(family):
    """Every column drives ``Delta* psi`` to the raster truncation floor."""
    focus = th.FocalCircle(1.0, 0.0)
    grid_r = np.linspace(0.55, 1.45, 181)
    grid_z = np.linspace(-0.45, 0.45, 181)
    mesh_r, mesh_z = np.meshgrid(grid_r, grid_z)
    away = np.hypot(mesh_r - focus.radius, mesh_z - focus.height) > 0.18
    columns = th.ToroidalHarmonics(focus, order=4, families=(family,)).flux(
        mesh_r.ravel(), mesh_z.ravel()
    )
    step = grid_r[1] - grid_r[0]
    for index in range(columns.shape[1]):
        flux = columns[:, index].reshape(mesh_r.shape)
        residual = np.where(
            away, th.grad_shafranov_residual(flux, grid_r, grid_z), np.nan
        )
        scale = np.nanmax(np.abs(np.where(away, flux, np.nan))) / step**2
        assert np.nanmax(np.abs(residual)) / scale < 1.0e-3


def test_field_columns_are_the_flux_gradient():
    """Analytic field matches central differences of the analytic flux."""
    basis = th.ToroidalHarmonics(
        th.FocalCircle(1.0, 0.0), order=4, families=(th.INNER, th.OUTER)
    )
    rng = np.random.default_rng(3)
    r = rng.uniform(0.4, 1.7, 60)
    z = rng.uniform(-0.6, 0.6, 60)
    step = 1.0e-6
    by_radius = (basis.flux(r + step, z) - basis.flux(r - step, z)) / (2.0 * step)
    by_height = (basis.flux(r, z + step) - basis.flux(r, z - step)) / (2.0 * step)
    radial, vertical = basis.field(r, z)
    circumference = 2.0 * np.pi * r[:, None]
    assert np.allclose(radial, -by_height / circumference, rtol=1e-6, atol=1e-14)
    assert np.allclose(vertical, by_radius / circumference, rtol=1e-6, atol=1e-14)


def test_field_columns_are_the_flux_gradient_at_small_focal_distance():
    """The positive prefactor identity retains the derivative near eta zero."""
    focus = th.FocalCircle(1.0, 0.0)
    radius, height = th.focal_position(focus, 0.005, 0.0)
    radius, height = float(radius), float(height)
    frame = th.focal_frame(np.asarray([radius]), np.asarray([height]), focus)
    assert float(frame.cosine[0] - 1.0) < 2.0 * th.MINIMUM_COSH_GAP

    basis = th.ToroidalHarmonics(focus, order=1, families=(th.INNER, th.OUTER))
    step = radius * 1.0e-5
    by_radius = (
        basis.flux(np.asarray([radius + step]), np.asarray([height]))
        - basis.flux(np.asarray([radius - step]), np.asarray([height]))
    ) / (2.0 * step)
    by_height = (
        basis.flux(np.asarray([radius]), np.asarray([height + step]))
        - basis.flux(np.asarray([radius]), np.asarray([height - step]))
    ) / (2.0 * step)
    radial, vertical = basis.field(np.asarray([radius]), np.asarray([height]))
    circumference = 2.0 * np.pi * radius
    assert np.allclose(radial, -by_height / circumference, rtol=2e-6, atol=1e-14)
    assert np.allclose(vertical, by_radius / circumference, rtol=2e-6, atol=1e-14)


def test_projection_is_the_axis_weighted_field():
    """A probe row is its pose's field columns contracted with its sensitive axis."""
    basis = th.ToroidalHarmonics(th.FocalCircle(1.0, 0.0), order=3)
    r, z, cosine, sine = pickup_array()
    radial, vertical = basis.field(r, z)
    assert np.allclose(
        basis.project(r, z, cosine, sine),
        cosine[:, None] * radial + sine[:, None] * vertical,
    )


def test_labels_track_the_column_count_and_family_order():
    """Labels name every column exactly once, family by family."""
    basis = th.ToroidalHarmonics(
        th.FocalCircle(1.0, 0.0), order=3, families=(th.INNER, th.OUTER)
    )
    labels = basis.labels
    assert (
        len(labels)
        == len(set(labels))
        == basis.flux(np.array([1.2]), np.array([0.1])).shape[1]
    )
    assert labels[:3] == ["inner0", "inner1c", "inner1s"]
    assert labels[7] == "outer0"


def test_basis_rejects_an_unknown_family_and_a_negative_order():
    """Configuration errors are refused where they are made, not downstream."""
    with pytest.raises(ValueError, match="unknown radial families"):
        th.ToroidalHarmonics(th.FocalCircle(1.0), families=("sideways",))
    with pytest.raises(ValueError, match="non-negative"):
        th.ToroidalHarmonics(th.FocalCircle(1.0), order=-1)


# --- the filament expansion -------------------------------------------------


@pytest.mark.parametrize("family, fraction", [(th.INNER, 0.5), (th.OUTER, 1.7)])
@pytest.mark.parametrize("source", [(1.3, 0.25), (0.75, -0.2), (1.45, -1.2)])
def test_filament_expansion_reproduces_the_ring_kernel(family, fraction, source):
    """Each family reproduces the ring Green's function on its own valid side."""
    focus = th.FocalCircle(1.0, 0.0)
    basis = th.ToroidalHarmonics(focus, order=24, families=(family,))
    coefficients = th.filament_coefficients(basis, *source, current=3.7)
    distance = th.focal_frame(
        np.array([source[0]]), np.array([source[1]]), focus
    ).distance[0]
    angle = np.linspace(0.0, 2.0 * np.pi, 300, endpoint=False)
    ring_r, ring_z = th.focal_position(focus, fraction * distance, angle)
    live = ring_r > 0.05
    expected = 3.7 * greens_psi(ring_r[live], ring_z[live], *source)
    got = basis.flux(ring_r[live], ring_z[live]) @ coefficients
    # The expansion's own truncation rate is the bound, so this checks the RATE
    # rather than a magnitude: successive degrees fall by the ratio of the
    # observer's focal distance to the source's, and the contour sits at a known
    # fraction of it.  Measured against the ring kernel the margin is a decade
    # everywhere, from a contour close in to one where the series is at round-off.
    gap = abs(1.0 - fraction) * distance
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < np.exp(
        -basis.order * gap
    )


def test_filament_expansion_needs_a_single_sided_basis():
    """A two-family basis has no single filament expansion and says so."""
    basis = th.ToroidalHarmonics(
        th.FocalCircle(1.0, 0.0), order=4, families=(th.INNER, th.OUTER)
    )
    with pytest.raises(ValueError, match="single-sided"):
        th.filament_coefficients(basis, 1.3, 0.2)


@pytest.mark.parametrize(
    "source", [(1.3, 0.25), (0.75, -0.2), (1.45, -1.2), (0.35, 0.9)]
)
def test_locate_source_inverts_the_filament_expansion(source):
    """Position and current come back out of the coefficients they went into."""
    basis = th.ToroidalHarmonics(th.FocalCircle(1.0, 0.0), order=10)
    estimate = th.locate_source(
        basis, th.filament_coefficients(basis, *source, current=1234.0)
    )
    assert np.hypot(estimate.r - source[0], estimate.z - source[1]) < 1.0e-6
    assert estimate.current == pytest.approx(1234.0, rel=1e-6)
    assert estimate.modulus_residual < 1.0e-6
    assert estimate.phase_residual < 1.0e-6


def test_convergent_points_split_at_the_source_focal_circle():
    """The validity mask is the source's own focal circle, from either side."""
    focus = th.FocalCircle(1.0, 0.0)
    source = (1.3, 0.25)
    distance = float(
        th.focal_frame(np.array([source[0]]), np.array([source[1]]), focus).distance[0]
    )
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    outside_r, outside_z = th.focal_position(focus, 0.9 * distance, angle)
    inside_r, inside_z = th.focal_position(focus, 1.1 * distance, angle)
    inner = th.ToroidalHarmonics(focus, order=4, families=(th.INNER,))
    outer = th.ToroidalHarmonics(focus, order=4, families=(th.OUTER,))
    assert th.convergent_points(inner, outside_r, outside_z, distance).all()
    assert not th.convergent_points(inner, inside_r, inside_z, distance).any()
    assert th.convergent_points(outer, inside_r, inside_z, distance).all()
    assert not th.convergent_points(outer, outside_r, outside_z, distance).any()


# --- conditioning -----------------------------------------------------------


def test_equilibration_recovers_the_conditioning_the_units_destroy():
    """Scaling columns on the window cuts the condition number by many decades."""
    r, z, cosine, sine = pickup_array()
    focus = offset_focus((1.45, -1.20), 0.12)
    for order, gain in ((3, 1.0e2), (6, 1.0e4)):
        design = th.ToroidalHarmonics(focus, order=order).project(r, z, cosine, sine)
        fit = th.solve_equilibrated(design, np.zeros(r.size))
        assert fit.raw_condition / fit.equilibrated_condition > gain


def test_significance_cut_discards_directions_the_noise_buries():
    """A whitened direction below the threshold is dropped, not amplified."""
    r, z, cosine, sine = pickup_array()
    source = (1.45, -1.20)
    focus = offset_focus(source, 0.12)
    basis = th.ToroidalHarmonics(focus, order=12)
    current = matched_current(r, z, cosine, sine, source)
    data = projected_field(r, z, cosine, sine, [(*source, current)])
    noisy = data + np.random.default_rng(5).normal(0.0, SENSOR_FLOOR, data.shape)
    weight = np.full(data.shape, 1.0 / SENSOR_FLOOR)
    design = basis.project(r, z, cosine, sine)
    loose = th.solve_equilibrated(design, noisy, weight=weight)
    cut = th.solve_equilibrated(design, noisy, weight=weight, significance=3.0)
    assert cut.rank < loose.rank
    predicted = design @ cut.coefficients
    assert np.sqrt(np.mean((predicted - data) ** 2)) < np.sqrt(
        np.mean((design @ loose.coefficients - data) ** 2)
    )


def test_held_out_selection_prefers_a_degree_the_probes_resolve():
    """Degree selection improves on the lowest degree and stops short of the top."""
    r, z, cosine, sine = pickup_array()
    source = (1.45, -1.20)
    focus = offset_focus(source, 0.12)
    current = matched_current(r, z, cosine, sine, source)
    data = projected_field(r, z, cosine, sine, [(*source, current)])
    orders = list(range(1, 15))

    def build(order):
        return th.ToroidalHarmonics(focus, order=order).project(r, z, cosine, sine)

    best, scores = th.select_order(build, data, orders, folds=5, seed=3)
    assert best >= 4
    assert scores[best] < 0.1 * scores[1]
    noisy = data + np.random.default_rng(7).normal(0.0, SENSOR_FLOOR, data.shape)
    noisy_best, noisy_scores = th.select_order(build, noisy, orders, folds=5, seed=3)
    # noise removes the high degrees from contention outright
    assert noisy_best < best
    assert noisy_scores[max(orders)] > noisy_scores[noisy_best]


# --- recovery on the real array ---------------------------------------------


def test_probe_array_recovers_a_filament_without_noise():
    """The trusted channels determine the expansion and the source that made it."""
    r, z, cosine, sine = pickup_array()
    assert r.size == 77
    source = (1.45, -1.20)
    current = matched_current(r, z, cosine, sine, source)
    focus = offset_focus(source, 0.15)
    basis = th.ToroidalHarmonics(focus, order=10)
    data = projected_field(r, z, cosine, sine, [(*source, current)])
    design = basis.project(r, z, cosine, sine)
    fit = th.solve_equilibrated(design, data)
    assert fit.residual < 1.0e-6 * np.sqrt(np.mean(data**2))
    estimate = th.locate_source(basis, fit.coefficients)
    assert np.hypot(estimate.r - source[0], estimate.z - source[1]) < 1.0e-3
    assert estimate.current == pytest.approx(current, rel=1e-3)


def test_probe_array_recovers_filament_flux_where_no_probe_looked():
    """Flux is recovered on closed contours between the source and the probes."""
    r, z, cosine, sine = pickup_array()
    source = (1.45, -1.20)
    current = matched_current(r, z, cosine, sine, source)
    focus = offset_focus(source, 0.12)
    distance = float(
        th.focal_frame(np.array([source[0]]), np.array([source[1]]), focus).distance[0]
    )
    basis = th.ToroidalHarmonics(focus, order=12)
    data = projected_field(r, z, cosine, sine, [(*source, current)])
    fit = th.solve_equilibrated(basis.project(r, z, cosine, sine), data)
    angle = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    for fraction, tolerance in ((0.2, 1.0e-6), (0.6, 1.0e-4), (0.8, 1.0e-2)):
        ring_r, ring_z = th.focal_position(focus, fraction * distance, angle)
        live = ring_r > 0.05
        expected = current * greens_psi(ring_r[live], ring_z[live], *source)
        got = basis.flux(ring_r[live], ring_z[live]) @ fit.coefficients
        assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < tolerance


def test_probe_array_position_error_at_the_measured_sensor_floor():
    """The array's resolution at its own noise floor, on the surviving amplitude."""
    r, z, cosine, sine = pickup_array()
    source = (1.45, -1.20)
    current = matched_current(r, z, cosine, sine, source)
    focus = offset_focus(source, 0.15)
    basis = th.ToroidalHarmonics(focus, order=6)
    clean = projected_field(r, z, cosine, sine, [(*source, current)])
    design = basis.project(r, z, cosine, sine)
    weight = np.full(clean.shape, 1.0 / SENSOR_FLOOR)
    rng = np.random.default_rng(29)
    errors = []
    for _ in range(60):
        fit = th.solve_equilibrated(
            design,
            clean + rng.normal(0.0, SENSOR_FLOOR, clean.shape),
            weight=weight,
            significance=3.0,
        )
        estimate = th.locate_source(basis, fit.coefficients)
        errors.append(np.hypot(estimate.r - source[0], estimate.z - source[1]))
    assert np.median(errors) < 0.06
    assert np.percentile(errors, 90) < 0.09


def test_two_filament_shortfall_grows_with_separation():
    """The single-source law's shortfall is the pair-detection statistic."""
    r, z, cosine, sine = pickup_array()
    centre = (1.45, -1.20)
    focus = offset_focus(centre, 0.20)
    basis = th.ToroidalHarmonics(focus, order=10)
    design = basis.project(r, z, cosine, sine)
    shortfall = []
    for separation in (0.0, 0.05, 0.2, 0.8):
        pair = [
            (centre[0], centre[1] - 0.5 * separation, 0.5),
            (centre[0], centre[1] + 0.5 * separation, 0.5),
        ]
        field = projected_field(r, z, cosine, sine, pair)
        field *= SURVIVING_AMPLITUDE / np.sqrt(np.mean(field**2))
        fit = th.solve_equilibrated(design, field)
        shortfall.append(th.locate_source(basis, fit.coefficients).modulus_residual)
    assert np.all(np.diff(shortfall) > 0.0)
    assert shortfall[0] < 1.0e-5


def iterate_focus(r, z, cosine, sine, data, seed, *, noise=None, order=6, steps=10):
    """Move the focal circle onto the source it implies, and refit, until settled.

    The read is exact for any focal placement whose circle through the source
    encloses no probe, so a coarse seed can be walked in: fit, take the implied
    source, move the focal circle half-way onto it, refit.  Half-steps rather than
    whole ones because the map is not a contraction -- a focal circle landing ON
    the source drives every degree above zero to nothing and the distance it reads
    back becomes indeterminate.
    """
    weight = None if noise is None else np.full(data.shape, 1.0 / noise)
    estimate = None
    for _step in range(steps):
        basis = th.ToroidalHarmonics(th.FocalCircle(*seed), order=order)
        fit = th.solve_equilibrated(
            basis.project(r, z, cosine, sine),
            data,
            weight=weight,
            significance=0.0 if noise is None else 3.0,
        )
        estimate = th.locate_source(basis, fit.coefficients)
        seed = (0.5 * (seed[0] + estimate.r), 0.5 * (seed[1] + estimate.z))
    return estimate


@pytest.mark.parametrize("source", [(1.45, -1.20), (1.57, 0.71), (0.30, 0.60)])
def test_focal_placement_converges_from_a_coarse_seed(source):
    """A metre-scale seed walks onto the source, so no scan is needed to place it."""
    r, z, cosine, sine = pickup_array()
    current = matched_current(r, z, cosine, sine, source)
    data = projected_field(r, z, cosine, sine, [(*source, current)])
    estimate = iterate_focus(
        r, z, cosine, sine, data, (source[0] + 0.6, source[1] + 0.8)
    )
    assert np.hypot(estimate.r - source[0], estimate.z - source[1]) < 1.0e-3
    assert estimate.current == pytest.approx(current, rel=1e-3)


def test_iterated_placement_beats_a_fixed_one_at_the_sensor_floor():
    """Walking the focal circle in sharpens the read the noise floor allows."""
    r, z, cosine, sine = pickup_array()
    source = (1.45, -1.20)
    current = matched_current(r, z, cosine, sine, source)
    clean = projected_field(r, z, cosine, sine, [(*source, current)])
    fixed_basis = th.ToroidalHarmonics(offset_focus(source, 0.15), order=6)
    fixed_design = fixed_basis.project(r, z, cosine, sine)
    weight = np.full(clean.shape, 1.0 / SENSOR_FLOOR)
    rng = np.random.default_rng(17)
    fixed, walked = [], []
    for _ in range(40):
        data = clean + rng.normal(0.0, SENSOR_FLOOR, clean.shape)
        held = th.locate_source(
            fixed_basis,
            th.solve_equilibrated(
                fixed_design, data, weight=weight, significance=3.0
            ).coefficients,
        )
        moved = iterate_focus(
            r,
            z,
            cosine,
            sine,
            data,
            (source[0] + 0.3, source[1] + 0.4),
            noise=SENSOR_FLOOR,
        )
        fixed.append(np.hypot(held.r - source[0], held.z - source[1]))
        walked.append(np.hypot(moved.r - source[0], moved.z - source[1]))
    assert np.median(walked) < np.median(fixed)
    assert np.median(walked) < 0.02
