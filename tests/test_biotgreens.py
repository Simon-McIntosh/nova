"""Tests for the canonical axisymmetric Green's-function library.

These run on plain numpy arrays with no frame machinery.  They pin:

* the single corner antiderivative ``corner_fields`` against the Urankar
  Part III rectangular-section formula it was extracted from -- the dataclass
  cylinder kernel and the functional ``cylinder_greens`` both drive this one
  routine, so the guard protects against the two paths silently diverging;
* continuity of the finite-area rectangle kernel across the section's OWN corner
  planes, where the antiderivative's branch terms change sign -- twice over, since
  an oracle comparison bounds the VALUE either side and a ladder of difference
  quotients is what separates a discontinuity from the field's own slope;
* far-field agreement of the finite-area rectangle kernel with the point
  circular-loop kernel;
* psi<->B consistency of the point-loop kernel (Jackson forms);
* agreement of each kernel's numpy form with its ``xp``-threaded twin driven
  under numpy -- the two reach the complete elliptic integrals and the ``zeta``
  quadrature by different routines, so the pin is what keeps the trace and the
  host one physics, and it carries the ``k^2 -> 1`` corner and the exact place
  the two conventions genuinely part.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy.special

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.greens import (
    _ELLIPTIC_DIRECT_LIMIT,
    _ELLIPTIC_SERIES_LIMIT,
    _filament_elliptic_combinations,
    MU0,
    corner_fields,
    cylinder_greens,
    greens_bz_br,
    greens_psi,
    traced_corner_fields,
    traced_cylinder_greens,
    traced_filament_greens,
)
from nova.biot.zeta import zeta

# rectangular section + off-section targets (inside and outside)
_A, _Z0, _DA, _DZ = 0.90, 0.10, 0.12, 0.18
_TR = np.array([1.02, 1.20, 1.50, 0.93, 0.30, 1.90])
_TZ = np.array([0.10, 0.40, 0.00, 0.13, -1.20, 1.50])
_ULP = np.finfo(np.float64).eps


def _corner_stacks(a, z0, da, dz, tr, tz):
    """Return (rs, zs, r, z) corner arrays shaped (T, 4), matching the kernel."""
    rs = np.broadcast_to(
        np.array([a - da / 2, a + da / 2, a + da / 2, a - da / 2]), tr.shape + (4,)
    ).copy()
    zs = np.broadcast_to(
        np.array([z0 - dz / 2, z0 - dz / 2, z0 + dz / 2, z0 + dz / 2]), tr.shape + (4,)
    ).copy()
    r4 = np.repeat(tr[..., None], 4, axis=-1)
    z4 = np.repeat(tz[..., None], 4, axis=-1)
    return rs, zs, r4, z4


def _reference_corner_fields(rs, zs, r4, z4):
    """The Urankar Part III corner antiderivative, spelled out independently.

    Written out term by term against the paper, in the paper's own grouping: the
    ``P`` and ``Q`` coefficient families each carrying their own factor of gamma,
    and one ``Pi`` per characteristic.  The kernel instead folds a factor of gamma
    out of the two ring poles and carries the bounded product, which is what makes
    it finite and continuous on a corner plane; off those planes the two groupings
    are the same algebra and this pins that they are.

    What this reference does NOT do is form a pole or the modulus by subtraction.
    The paper's printed ``2r/(r - c)`` and ``1 - 4 r rs/a^2`` cost the branch
    cancellation a relative ``eps r^2/gamma^2``, which is already 3e-09 in the flux
    at a tenth of the section height off a corner level -- so a reference spelled
    that way disagrees with an accurate kernel by four decades more than the pin
    allows, and would be measuring its own conditioning rather than the algebra.
    Both spellings therefore take the same exact complements, the same ``zeta`` and
    the same complete-elliptic descent, leaving only the grouping between them;
    the quadrature and the special functions have their own accuracy gates in
    ``test_biotzeta`` and ``test_biotcompleteelliptic``.

    Returns ``(values, widest)``: the three antiderivative coefficients, and the
    magnitude of the largest term each of them is assembled from.  The second is
    what a rounding is proportional to and the first is not -- the corner value is a
    cancellation of those terms -- so a comparison of the two spellings that means
    anything about the ALGEBRA is taken over the widest term.
    """
    gamma = zs - z4
    b = rs + r4
    a = np.sqrt(gamma**2 + b**2)
    c2 = gamma**2 + r4**2
    c = np.sqrt(c2)
    k2 = 4 * r4 * rs / (gamma**2 + b**2)
    complement = (gamma**2 + (rs - r4) ** 2) / (gamma**2 + b**2)
    big_k, big_e = complete_kind(complement)
    v = 1 + k2 * (gamma**2 - b * r4) / (2 * r4 * rs)
    u_coef = k2 * (4 * gamma**2 + 3 * rs**2 - 5 * r4**2) / (4 * r4)
    zt = zeta(rs, r4, gamma, (np.pi / 2) * np.ones_like(rs))

    # r - c is -gamma^2/(r + c) exactly, which is what the first characteristic and
    # its own pole are built from rather than from r less c
    np2 = {
        1: -2 * r4 * (r4 + c) / gamma**2,
        2: 2 * r4 / (r4 + c),
        3: 4 * r4 * rs / b**2,
    }
    pole = {
        1: ((r4 + c) / gamma) ** 2,
        2: (gamma / (r4 + c)) ** 2,
        3: ((rs - r4) / b) ** 2,
    }
    pi3 = {p: complete_pole(pole[p], complement) for p in (1, 2, 3)}
    # rs -/+ c: the minus sign cancels to nothing on a corner's own radius, where it
    # is (rs - r) less gamma^2/(r + c) instead
    edge = {1: rs + c, 2: (rs - r4) - gamma**2 / (r4 + c)}
    qr = {p: edge[p] * np2[p] * gamma**2 * c / r4 for p in (1, 2)}
    qr[3] = np.zeros_like(r4)
    qz = {p: edge[p] * -2 * gamma * c * np2[p] for p in (1, 2)}
    qz[3] = gamma * b * (rs - r4) * np2[3]
    pphi = {p: edge[p] * np2[p] * c * (3 * r4**2 - c2) / (2 * r4) for p in (1, 2)}
    pphi[3] = -rs / b * (rs - r4) * (3 * r4**2 - rs**2)

    def p_sum(coef):
        return sum((-1) ** p * coef[p] * pi3[p] for p in (1, 2, 3))

    cphi = -1 / 3 * r4**2 * np.pi / 2 * np.sign(gamma) * (np.sign(rs - r4) + 1)
    aphi = (
        cphi
        + gamma * r4 * zt
        + gamma * a / (6 * r4) * (u_coef * big_k - 2 * rs * big_e)
        + gamma / (6 * a * r4) * p_sum(pphi)
    )
    br = (
        r4 * zt - a / (2 * r4) * rs * (big_e - v * big_k) - 1 / (4 * a * r4) * p_sum(qr)
    )
    bz = (
        3 / r4 * cphi
        + 2 * gamma * zt
        - a / (2 * r4) * 3 / 2 * gamma * k2 * big_k
        - 1 / (4 * a * r4) * p_sum(qz)
    )

    def widest(*terms):
        return np.max(np.abs(np.array(terms)), axis=0)

    # every addend that carries its own rounding, the inner differences and the
    # third-kind sum broken back out: those two are where the cancellation is, so
    # the terms and not the bracketed groups are what the envelope is taken over
    envelope = (
        widest(
            cphi,
            gamma * r4 * zt,
            gamma * a / (6 * r4) * u_coef * big_k,
            gamma * a / (6 * r4) * 2 * rs * big_e,
            *[gamma / (6 * a * r4) * pphi[p] * pi3[p] for p in (1, 2, 3)],
        ),
        widest(
            r4 * zt,
            a / (2 * r4) * rs * big_e,
            a / (2 * r4) * rs * v * big_k,
            *[qr[p] * pi3[p] / (4 * a * r4) for p in (1, 2, 3)],
        ),
        widest(
            3 / r4 * cphi,
            2 * gamma * zt,
            a / (2 * r4) * 3 / 2 * gamma * k2 * big_k,
            *[qz[p] * pi3[p] / (4 * a * r4) for p in (1, 2, 3)],
        ),
    )
    return (aphi, br, bz), envelope


def test_corner_fields_matches_reference_formula():
    """The single corner antiderivative reproduces the Part III formula exactly.

    Held at a few ulp rather than a physics tolerance: with the quadrature shared
    between the two spellings, any drift here is the algebra diverging, and that
    should never cost more than the last couple of bits.  The targets are all well
    off the section's corner planes, which is where the paper's own arrangement of
    the characteristics is still faithful to its algebra.

    A few ulp OF WHAT is the whole question, because the corner value is a
    cancellation.  The widest term the antiderivative is assembled from reaches 206
    times the four-corner stack's own scale on these targets and four decades of it
    on targets they do not visit, and the third-kind sum inside cancels by a further
    800 -- so one rounding of one term's last bit arrives as a couple of THOUSAND
    ulp of the corner value.  Counting ulp of the value measures that cancellation
    rather than the algebra, and it is the target's cancellation, not the formula's.

    What the algebra answers for is the last bit of each TERM, so that is the pin.
    The scaled bound below is the cruder statement of the same thing, kept because
    it is the difference the corner rule actually forms.
    """
    stacks = _corner_stacks(_A, _Z0, _DA, _DZ, _TR, _TZ)
    got = corner_fields(*stacks)
    ref, envelope = _reference_corner_fields(*stacks)
    for one, other, widest in zip(got, ref, envelope, strict=True):
        # 2.7 ulp of the widest term here, and no worse than 3.9 across several
        # hundred random target and section geometries, over which the cancellation
        # factor spans three decades and this does not move -- the two groupings
        # differ by rounding and nothing else.  Twice the measured worst.
        assert np.max(np.abs(one - other) / widest) < 8 * _ULP
        # the same disagreement against the corner STACK's scale, which is what the
        # corner rule differences the four values against: 3.7e-14 measured, the
        # term bound multiplied by this target set's own cancellation, so 2.7 times
        # the measured worst is as tight as it goes.  Tightening it towards the term
        # bound would only make it a statement about which targets were chosen.
        scale = np.abs(other).max(axis=-1, keepdims=True)
        assert np.max(np.abs(one - other) / scale) < 1e-13


def test_cylinder_far_field_matches_point_loop():
    """Far from a small section the rectangle kernel tends to the point loop."""
    # targets several section-widths away, where the finite-area correction is
    # a sub-percent second-moment term
    tr = np.array([2.5, 0.4, 1.8])
    tz = np.array([1.5, -1.4, 1.2])
    psi_c, br_c, bz_c = cylinder_greens(tr, tz, _A, _Z0, 0.02, 0.02)
    psi_p = greens_psi(tr, tz, _A, _Z0)
    bz_p, br_p = greens_bz_br(tr, tz, _A, _Z0)
    np.testing.assert_allclose(psi_c, psi_p, rtol=1e-3)
    np.testing.assert_allclose(bz_c, bz_p, rtol=1e-3)
    np.testing.assert_allclose(br_c, br_p, rtol=1e-3)


def test_point_loop_psi_b_consistency():
    """B_Z = (1/2piR) dpsi/dR and B_R = -(1/2piR) dpsi/dZ for the point loop."""
    a, z0 = 0.9, 0.1
    h = 1e-6
    for r, z in [(1.3, 0.3), (0.6, -0.2), (1.1, 0.5)]:
        psi_rp = greens_psi(np.array([r + h]), np.array([z]), a, z0)[0]
        psi_rm = greens_psi(np.array([r - h]), np.array([z]), a, z0)[0]
        psi_zp = greens_psi(np.array([r]), np.array([z + h]), a, z0)[0]
        psi_zm = greens_psi(np.array([r]), np.array([z - h]), a, z0)[0]
        bz, br = greens_bz_br(np.array([r]), np.array([z]), a, z0)
        np.testing.assert_allclose(
            bz[0], (psi_rp - psi_rm) / (2 * h) / (2 * np.pi * r), rtol=1e-5
        )
        np.testing.assert_allclose(
            br[0], -(psi_zp - psi_zm) / (2 * h) / (2 * np.pi * r), rtol=1e-5
        )


def test_point_loop_exact_axis_uses_the_symmetry_limit_without_warnings():
    """The exact axis has zero linked flux and radial field with analytic Bz."""
    source_r = 1.7
    target_z = np.array([-3.0, -0.4, 0.0, 0.9, 4.0])
    target_r = np.zeros_like(target_z)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        psi = greens_psi(target_r, target_z, source_r, 0.2)
        bz, br = greens_bz_br(target_r, target_z, source_r, 0.2)
        traced_psi, traced_br, traced_bz = traced_filament_greens(
            np, target_r, target_z, source_r, 0.2
        )
    expected_bz = (
        MU0 * source_r**2 / (2.0 * (source_r**2 + (target_z - 0.2) ** 2) ** 1.5)
    )
    np.testing.assert_array_equal(psi, 0.0)
    np.testing.assert_array_equal(br, 0.0)
    np.testing.assert_allclose(bz, expected_bz, rtol=3e-16, atol=0.0)
    np.testing.assert_array_equal(traced_psi, psi)
    np.testing.assert_array_equal(traced_br, br)
    np.testing.assert_allclose(traced_bz, expected_bz, rtol=3e-16, atol=0.0)


def _decimal_elliptic_combinations(parameter, *, derivative=False):
    """Return a 100-digit power-series arbiter for the three loop combinations."""
    with localcontext() as context:
        context.prec = 110
        one = Decimal(1)
        two = Decimal(2)
        arithmetic_mean = one
        geometric_mean = (one / two).sqrt()
        correction = one / Decimal(4)
        weight = one
        for _ in range(9):
            next_mean = (arithmetic_mean + geometric_mean) / two
            geometric_mean = (arithmetic_mean * geometric_mean).sqrt()
            correction -= weight * (arithmetic_mean - next_mean) ** 2
            arithmetic_mean = next_mean
            weight *= two
        pi = (arithmetic_mean + geometric_mean) ** 2 / (Decimal(4) * correction)

        m = Decimal.from_float(float(parameter))
        complete_coefficient = one
        first_minus_second = Decimal(0)
        flux = Decimal(0)
        radial = Decimal(0)
        for degree in range(1, 80):
            prior_coefficient = complete_coefficient
            ratio = Decimal(2 * degree - 1) / Decimal(2 * degree)
            complete_coefficient *= ratio * ratio
            if derivative:
                power = one if degree == 1 else m ** (degree - 1)
                power *= degree
            else:
                power = m**degree
            first_minus_second += (
                complete_coefficient
                * Decimal(2 * degree)
                / Decimal(2 * degree - 1)
                * power
            )
            if degree >= 2:
                coefficient = (
                    prior_coefficient * Decimal(degree - 1) / Decimal(2 * degree)
                )
                flux += coefficient * power
                radial += Decimal(2 * degree - 1) * coefficient * power
        scale = pi / two
        return np.array(
            [float(scale * value) for value in (first_minus_second, flux, radial)]
        )


def test_small_parameter_elliptic_combinations_match_a_high_precision_arbiter():
    """Cancellation-free loop combinations retain all binary64 digits."""
    parameter = np.array(
        [
            1e-16,
            1e-12,
            1.2e-9,
            3.67e-9,
            3.67e-8,
            1e-6,
            0.5 * _ELLIPTIC_SERIES_LIMIT,
            np.nextafter(_ELLIPTIC_SERIES_LIMIT, 0.0),
            _ELLIPTIC_SERIES_LIMIT,
        ]
    )
    first = scipy.special.ellipk(parameter)
    second = scipy.special.ellipe(parameter)
    got = np.array(
        _filament_elliptic_combinations(np, parameter, 1.0 - parameter, first, second)
    ).T
    expected = np.array([_decimal_elliptic_combinations(value) for value in parameter])
    np.testing.assert_allclose(got, expected, rtol=3e-16, atol=0.0)


def test_elliptic_series_transition_has_no_artificial_boundary_jump():
    """Adjacent values at the series boundary move only by the analytic slope."""
    parameter = np.array(
        [
            np.nextafter(_ELLIPTIC_SERIES_LIMIT, 0.0),
            _ELLIPTIC_SERIES_LIMIT,
            np.nextafter(_ELLIPTIC_SERIES_LIMIT, np.inf),
        ]
    )
    first = scipy.special.ellipk(parameter)
    second = scipy.special.ellipe(parameter)
    got = np.array(
        _filament_elliptic_combinations(np, parameter, 1.0 - parameter, first, second)
    ).T
    expected = np.array([_decimal_elliptic_combinations(value) for value in parameter])
    np.testing.assert_allclose(got, expected, rtol=4e-16, atol=0.0)
    observed_step = got[2] - got[0]
    expected_step = expected[2] - expected[0]
    roundoff = 4.0 * np.spacing(np.abs(got[1]))
    assert np.all(np.abs(observed_step - expected_step) <= roundoff)

    direct_parameter = np.array(
        [
            np.nextafter(_ELLIPTIC_DIRECT_LIMIT, 0.0),
            _ELLIPTIC_DIRECT_LIMIT,
            np.nextafter(_ELLIPTIC_DIRECT_LIMIT, np.inf),
        ]
    )
    first = scipy.special.ellipk(direct_parameter)
    second = scipy.special.ellipe(direct_parameter)
    direct = np.array(
        _filament_elliptic_combinations(
            np,
            direct_parameter,
            1.0 - direct_parameter,
            first,
            second,
        )
    ).T
    direct_expected = np.array(
        [_decimal_elliptic_combinations(value) for value in direct_parameter]
    )
    np.testing.assert_allclose(direct, direct_expected, rtol=2e-12, atol=0.0)


def test_small_parameter_point_filament_matches_a_high_precision_arbiter():
    """Psi and both field components keep their scale in the far regime."""
    requested = np.array([1e-16, 1e-12, 1.2e-9, 3.67e-9, 3.67e-8, 1e-6, 0.009])
    target_z = 2.0 * np.sqrt(1.0 / requested - 1.0)
    target_r = np.ones_like(target_z)
    parameter = 4.0 / (4.0 + target_z**2)
    combinations = np.array(
        [_decimal_elliptic_combinations(value) for value in parameter]
    )
    span_root = np.sqrt(4.0 + target_z**2)
    expected = np.column_stack(
        [
            MU0 * span_root * combinations[:, 1],
            MU0 / (2.0 * np.pi) * target_z / span_root * combinations[:, 2],
            MU0 / (2.0 * np.pi) / span_root * combinations[:, 0],
        ]
    )
    host_bz, host_br = greens_bz_br(target_r, target_z, 1.0, 0.0)
    host = np.column_stack([greens_psi(target_r, target_z, 1.0, 0.0), host_br, host_bz])
    traced = np.column_stack(traced_filament_greens(np, target_r, target_z, 1.0, 0.0))
    np.testing.assert_allclose(host, expected, rtol=5e-15, atol=0.0)
    np.testing.assert_allclose(traced, expected, rtol=5e-15, atol=0.0)


def test_small_parameter_branch_has_traced_value_and_tangent_continuity():
    """The held series arm supplies finite, accurate traced tangents."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")

    def combinations(parameter):
        first, second = complete_kind(1.0 - parameter, xp=jnp)
        return jnp.stack(
            _filament_elliptic_combinations(
                jnp, parameter, 1.0 - parameter, first, second
            )
        )

    for parameter in (1e-12, 0.5 * _ELLIPTIC_SERIES_LIMIT):
        value, tangent = jax.jvp(
            combinations, (jnp.asarray(parameter),), (jnp.asarray(1.0),)
        )
        np.testing.assert_allclose(
            np.asarray(value),
            _decimal_elliptic_combinations(parameter),
            rtol=5e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(tangent),
            _decimal_elliptic_combinations(parameter, derivative=True),
            rtol=5e-14,
            atol=0.0,
        )

    distance = 1e-8
    lower_value, lower_tangent = jax.jvp(
        combinations,
        (jnp.asarray(_ELLIPTIC_SERIES_LIMIT - distance),),
        (jnp.asarray(1.0),),
    )
    upper_value, upper_tangent = jax.jvp(
        combinations,
        (jnp.asarray(_ELLIPTIC_SERIES_LIMIT + distance),),
        (jnp.asarray(1.0),),
    )
    expected_change = (
        2.0
        * distance
        * _decimal_elliptic_combinations(_ELLIPTIC_SERIES_LIMIT, derivative=True)
    )
    np.testing.assert_allclose(
        np.asarray(upper_value - lower_value), expected_change, rtol=2e-5, atol=1e-18
    )
    lower_expected = _decimal_elliptic_combinations(
        _ELLIPTIC_SERIES_LIMIT - distance, derivative=True
    )
    upper_expected = _decimal_elliptic_combinations(
        _ELLIPTIC_SERIES_LIMIT + distance, derivative=True
    )
    np.testing.assert_allclose(
        np.asarray(lower_tangent), lower_expected, rtol=5e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(upper_tangent), upper_expected, rtol=5e-10, atol=1e-14
    )


def test_units_constant():
    """The kernel carries the SI vacuum permeability."""
    assert MU0 == 4.0e-7 * np.pi


# --- continuity across the section's own corner planes -----------------------
#
# The corner antiderivative carries branch terms that are ODD in the target's
# offset from a corner plane: an arctangent boundary term whose limit is a signed
# right angle, and two third-kind integrals whose poles collapse onto the ends of
# the integration range there.  Those jumps cancel each other exactly, so the
# antiderivative is continuous -- but only if every characteristic and the modulus
# complement are formed from the geometry.  A mis-set branch, or a characteristic
# reached by subtraction, leaves an offset-INDEPENDENT jump; a ladder of standoffs
# over nine decades is what separates that from the smooth variation across the
# plane, since only the jump fails to fall with the standoff.

_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ = 4.0, 0.0, 0.1, 0.08
_LADDER = 10.0 ** -np.arange(2.0, 11.0)
_STRADDLE = np.concatenate([-_LADDER, [0.0], _LADDER[::-1]])
# a target radius strictly inside the section's radial span, off its mid-plane and
# off both corner radii, so the two lower corners straddle it in radius
_INSIDE_SPAN = _FACE_A - 0.0435

# Sections the crossings below are exercised on, as ``(a, z0, da, dz)``: the one the
# rest of this group is written around, and a production winding-pack shape -- small,
# square, and at a large radius, where the corner cancellation is thirty times worse
# and everything built on differencing the kernel inherits that.  Section shape is a
# test INPUT here rather than a constant, because a bound that holds on one aspect
# ratio and one radius is a statement about that geometry and not about the kernel.
_SECTIONS = (
    (4.0, 0.0, 0.1, 0.08),
    (6.2, 0.0, 0.02, 0.02),
)

# The four ways a target crosses a corner plane, as fractions of the section's own
# extents: ``(radial offset, vertical offset, across)``, where ``across`` selects
# which coordinate the standoff is added to and the other is offset onto the corner.
# The first two cross a face LEVEL, at a radius inside the section's radial span and
# at one outside it; the last two cross a corner RADIUS, at a level inside the
# section's vertical span and at one below it.  The pair that stays outside the span
# never enters the conductor, so the branch terms there change sign without the
# field's own kink to hide behind.  Fractions rather than metres so the same four
# configurations follow the section: on the first section they are the radii and
# levels the straddle tests above use.
_CROSSINGS = (
    (-0.435, 0.0, 1.0),
    (0.21, 0.0, 1.0),
    (0.0, 0.2125, 0.0),
    (0.0, -1.125, 0.0),
)

# Standoffs as a fraction of the section's extent in the direction they are taken.
# The quotient's floor is the kernel's own relative round-off divided by the RELATIVE
# field difference a rung forms, and that difference is set by ``d/L`` for a section
# of extent ``L`` -- so a ladder fixed in metres spends a different part of its budget
# on every geometry, and a small section at a large radius loses its finest rungs
# outright.  Fixed as a fraction, every section gets a ladder its own field supports.
_LADDER_FRACTIONS = 10.0 ** -np.arange(1.0, 8.0)
# rungs before this one still carry the symmetric quotient's own second-order
# truncation, which falls as ``(d/L)^2`` and is a part in 1e6 by here
_SETTLED_RUNG = 3


def _crossing_centres(section, radial, vertical, across):
    """Yield the two corner-plane centres of a crossing, one per corner."""
    a, z0, da, dz = section
    for corner in (-1.0, 1.0):
        yield (
            a + radial * da + (1.0 - across) * corner * da / 2,
            z0 + vertical * dz + across * corner * dz / 2,
        )


def _rectangle(a, z0, da, dz):
    """Return the four ``(r, z)`` corners of an axis-aligned rectangular section."""
    return np.array(
        [
            [a - da / 2, z0 - dz / 2],
            [a + da / 2, z0 - dz / 2],
            [a + da / 2, z0 + dz / 2],
            [a - da / 2, z0 + dz / 2],
        ]
    )


def _worst_against_the_polygon(target_r, target_z, section=None):
    """Return the worst relative ``(psi, B_R, B_Z)`` deviation from the oracle.

    The oracle is the fully analytic polygon reduction, not the quadrature one: a
    target ON a face sits on the quadrature integrand's own boundary layer, where
    the shipped 16x48 rule is six decades off in ``B_R``, so the quadrature form
    cannot referee the singular configuration.  Away from the plane the two agree,
    which :func:`test_the_two_polygon_oracles_agree_off_the_face` pins.
    """
    from nova.biot.polygonanalytic import polygon_analytic_greens

    a, z0, da, dz = section or (_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ)
    got = cylinder_greens(target_r, target_z, a, z0, da, dz)
    exact = polygon_analytic_greens(target_r, target_z, _rectangle(a, z0, da, dz))
    # the field is scaled by its own magnitude rather than per component: B_R
    # passes through zero on the section's mid-plane, where a per-component
    # relative measure has no meaning
    field = np.hypot(exact[1], exact[2])
    local = (np.abs(exact[0]), field, field)
    return [
        float(np.max(np.abs(one - other) / scale))
        for one, other, scale in zip(got, exact, local, strict=True)
    ]


def test_the_two_polygon_oracles_agree_off_the_face():
    """Off the plane the quadrature and analytic polygon kernels are one oracle.

    Which is what licenses the analytic one as the referee ON the plane, where the
    quadrature rule's panels straddle the integrand's boundary layer.
    """
    from nova.biot.polygon import polygon_greens
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = _rectangle(_FACE_A, _FACE_Z0, _FACE_DA, _FACE_DZ)
    target_r = np.full(6, _INSIDE_SPAN)
    target_z = (
        _FACE_Z0 - _FACE_DZ / 2 + np.array([-0.03, -0.01, -3e-3, 3e-3, 0.01, 0.03])
    )
    quadrature = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
    analytic = polygon_analytic_greens(target_r, target_z, vertices)
    for one, other in zip(quadrature, analytic, strict=True):
        np.testing.assert_allclose(one, other, rtol=2e-9)


def test_the_rectangle_kernel_is_continuous_across_a_horizontal_face():
    """Approaching a section face from either side reaches the same field.

    The configuration a plasma bundle produces by alignment: a grid row level with
    the cell's own lower or upper face, at a radius inside the cell's radial span,
    so the two corners on that face straddle the target in radius and the
    arctangent boundary term is live at one of them and dead at the other.
    """
    for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
        worst = _worst_against_the_polygon(
            np.full(_STRADDLE.shape, _INSIDE_SPAN), level + _STRADDLE
        )
        assert max(worst) < 1e-10, worst


def test_the_rectangle_kernel_is_continuous_across_a_source_radius_plane():
    """Crossing a corner's own radius leaves the field unchanged.

    The other pair of planes the branch terms change sign across: ``rs = r`` flips
    the arctangent's limit at the far end of the angle range, and it also drives
    the third pole and the modulus complement to their own confluence.
    """
    for radius in (_FACE_A - _FACE_DA / 2, _FACE_A + _FACE_DA / 2):
        worst = _worst_against_the_polygon(
            radius + _STRADDLE, np.full(_STRADDLE.shape, _FACE_Z0 + 0.017)
        )
        assert max(worst) < 1e-10, worst


def test_a_target_on_the_source_radius_line_survives_the_corner():
    """Both confluences at once -- the corner itself -- stays on the oracle.

    Sliding along ``rs = r`` through a corner drives the modulus complement, the
    near pole and the third pole to zero together, where the first kind diverges
    logarithmically and the reduction's total weight on it is zero.  It is reached
    rather than approached: the middle row is the vertex exactly.  This is also the
    line the near pole's own numerator ``rs - c`` vanishes on, which is why it is
    formed as ``(rs - r)`` less ``gamma^2/(r + c)``; taken as a subtraction it costs
    the flux 2e-07 a tenth of a micrometre out.
    """
    for radius in (_FACE_A - _FACE_DA / 2, _FACE_A + _FACE_DA / 2):
        for level in (_FACE_Z0 - _FACE_DZ / 2, _FACE_Z0 + _FACE_DZ / 2):
            worst = _worst_against_the_polygon(
                np.full(_STRADDLE.shape, radius), level + _STRADDLE
            )
            assert max(worst) < 1e-10, worst


def test_every_corner_plane_crossing_holds_the_oracle():
    """All four ways of crossing a corner plane land on the analytic polygon kernel.

    The oracle half of the crossing set, and it reaches two configurations the
    straddle tests above do not: a radius OUTSIDE the section's radial span crossing
    a face level, and a level below the section crossing a corner radius.  In both
    the target stays outside the conductor throughout, so the branch terms change
    sign with no kink in the field to absorb an error, and neither the flux nor
    either field component may move.

    The middle row of the straddle is the plane exactly.  On the two level crossings
    that is ``gamma == 0`` to the bit, where the kernel masks the pole moments onto
    the mean of their two one-sided limits instead of forming a reciprocal
    characteristic -- an assignment no difference quotient reaches, because every
    quotient stands off the plane on both sides.
    """
    for section in _SECTIONS:
        extents = (section[3], section[2])  # the extent the standoff is taken along
        for radial, vertical, across_the_level in _CROSSINGS:
            straddle = _LADDER_FRACTIONS * extents[int(across_the_level)]
            straddle = np.concatenate([-straddle, [0.0], straddle[::-1]])
            for centre in _crossing_centres(
                section, radial, vertical, across_the_level
            ):
                worst = _worst_against_the_polygon(
                    centre[0] + (1.0 - across_the_level) * straddle,
                    centre[1] + across_the_level * straddle,
                    section,
                )
                # over the whole ladder including the plane itself: 1.5e-12 measured
                # on the 4 m section and 2.2e-11 on the 2 cm one, where the corner
                # cancellation is thirty times worse -- so the reserve is 4.6-fold on
                # the smaller section and the bound is set by it.  What holds the
                # floor down at all is what the sibling straddle tests rest on --
                # every characteristic and the modulus complement formed from the
                # geometry rather than by subtraction -- so a mis-masked branch on
                # the plane is a full ``pi r^2/3``, nine decades clear of the bound.
                assert max(worst) < 1e-10, worst


def test_the_corner_planes_carry_no_jump_of_their_own():
    """The difference quotient across a corner plane converges instead of blowing up.

    Oracle-free, and the sharpest statement of the defect this guards.  A branch
    term left uncancelled is a jump ``J`` that does not depend on the standoff, so
    the quotient ``|f(+d) - f(-d)|/2d`` grows as ``J/2d`` -- a decade per rung, all
    the way down -- where a continuous field's quotient settles on its own
    derivative.  Comparing the rungs against each other rather than against a bound
    is what frees this from the field's magnitude; it does NOT by itself free it from
    the section's geometry, which is why the standoffs are fractions of the section
    and not lengths.  Held in metres, the plateau a small section at a large radius
    can support is two decades worse than a large one's, for no reason that has
    anything to do with the kernel.

    A CONVERGENCE statement, not an error against a reference: nothing here is a
    tolerance on an accuracy.  It says the quotient has reached the derivative
    instead of growing as ``J/2d``, which is why the number below is a spread across
    rungs rather than a bound on any one of them.  The oracle statement about the
    same crossings is :func:`test_every_corner_plane_crossing_holds_the_oracle` and
    the two are complementary: an oracle catches a wrong VALUE, this catches a
    DISCONTINUITY, and a jump far below what an oracle comparison resolves still
    puts decades between the first rung and the last.
    """
    for section in _SECTIONS:
        extents = (section[3], section[2])  # the extent the standoff is taken along
        for radial, vertical, across_the_level in _CROSSINGS:
            extent = extents[int(across_the_level)]
            for centre in _crossing_centres(
                section, radial, vertical, across_the_level
            ):
                quotients = []
                for standoff in _LADDER_FRACTIONS * extent:
                    step = np.array([-standoff, standoff])
                    got = cylinder_greens(
                        centre[0] + (1.0 - across_the_level) * step,
                        centre[1] + across_the_level * step,
                        *section,
                    )
                    quotients.append(
                        [
                            abs(part[1] - part[0])
                            / (2.0 * standoff)
                            / np.abs(part).max()
                            for part in got
                        ]
                    )
                # Every rung from a ten-thousandth of the section's extent down is on
                # the limit already, and they agree with each other to 2.4e-04 on the
                # 4 m section and 5.1e-04 on the 2 cm one -- a factor of two apart,
                # where the same two sections on a ladder fixed in METRES give 4.3e-03
                # and 2.9e-02, the smaller one past any bound the larger supports.
                # What sets the floor is the quotient and not the kernel's accuracy:
                # the finest rung's relative field difference is of order ``d/L``, so
                # the kernel's own relative round-off arrives divided by it, and
                # holding ``d/L`` fixed holds the floor fixed across geometries.
                # Fourfold reserve on the worst measured.  A jump does not fall with
                # the standoff at all, so it grows as ``J/2d`` against a plateau and
                # puts three decades across this window: the gate fires on one of
                # 1e-10 of the field, ten decades below the ``pi r^2/3`` an
                # uncancelled branch term leaves.
                settled = np.array(quotients[_SETTLED_RUNG:])
                spread = np.ptp(settled, axis=0) / settled.mean(axis=0)
                assert np.max(spread) < 2e-3, (section, spread)


# --- quadrupole-corrected filament ------------------------------------------


def _hexagon(r0=6.2, z0=0.0, radius=0.06):
    """Return a regular hexagon section of the given circumradius."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def _trapezium(r0=6.2, z0=0.0):
    """Return an asymmetric section whose cross moment does not vanish."""
    return np.array(
        [
            [r0 - 0.05, z0 - 0.03],
            [r0 + 0.04, z0 - 0.03],
            [r0 + 0.06, z0 + 0.02],
            [r0 - 0.02, z0 + 0.04],
        ]
    )


def _sampled_moments(vertices, count=1400):
    """Return area-normalised second central moments by dense grid sampling.

    An independent oracle for the shoelace formulae: uniform points on the
    section's bounding box, kept by a crossing-number test against the polygon.
    """
    v = np.asarray(vertices, dtype=np.float64)
    grid_r, grid_z = np.meshgrid(
        np.linspace(v[:, 0].min(), v[:, 0].max(), count),
        np.linspace(v[:, 1].min(), v[:, 1].max(), count),
    )
    r, z = grid_r.ravel(), grid_z.ravel()
    inside = np.zeros(r.shape, dtype=bool)
    for (ra, za), (rb, zb) in zip(v, np.roll(v, -1, axis=0)):
        if zb == za:  # a horizontal ray never crosses a horizontal edge
            continue
        straddles = (za > z) != (zb > z)
        crossing = ra + (z - za) * (rb - ra) / (zb - za)
        inside ^= straddles & (r < crossing)
    r, z = r[inside], z[inside]
    r, z = r - r.mean(), z - z.mean()
    return float(np.mean(r * r)), float(np.mean(z * z)), float(np.mean(r * z))


def test_the_shoelace_second_moments_match_a_sampled_section():
    """The closed-form moments reproduce a dense sampling of the same polygon."""
    from nova.biot.greens import second_moments

    for vertices in (_hexagon(), _trapezium()):
        got = second_moments(vertices)
        expected = _sampled_moments(vertices)
        np.testing.assert_allclose(got, expected, rtol=3e-3, atol=1e-9)


def test_the_moment_filament_removes_the_bare_filament_floor():
    """Far out the bare filament stalls on a floor; the corrected one does not.

    The floor is the section second moment weighted by a curvature the MAJOR
    radius sets, so it does not decay with target distance. Measured against the
    exact polygon kernel at ten section radii, relative to the local magnitude.
    """
    from nova.biot.greens import moment_filament
    from nova.biot.polygon import polygon_greens

    vertices = _hexagon()
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    target_r = 6.2 + 0.6 * np.cos(angle)
    target_z = 0.6 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = moment_filament(target_r, target_z, vertices)
    bare_psi = greens_psi(target_r, target_z, 6.2, 0.0)
    bare_bz, bare_br = greens_bz_br(target_r, target_z, 6.2, 0.0)

    field = np.hypot(exact[1], exact[2])
    local = (np.abs(exact[0]), field, field)
    bare = (bare_psi, bare_br, bare_bz)
    for index in range(3):
        got = np.max(np.abs(corrected[index] - exact[index]) / local[index])
        floor = np.max(np.abs(bare[index] - exact[index]) / local[index])
        assert got < 1e-6
        assert floor > 20.0 * got


def test_the_quadrupole_correction_scales_as_the_section_area():
    """Shrinking the section by ten shrinks its correction by a hundred.

    The correction is a second moment, so it is quadratic in the section size and
    a section small against the ring reduces to the bare filament. Testing the
    SCALING rather than an absolute floor is what separates the physics from the
    difference scheme's round-off: the step scales with the section, which holds
    the correction's relative accuracy section-size independent.
    """
    from nova.biot.greens import moment_filament

    target_r = np.array([6.6, 5.9, 6.2])
    target_z = np.array([0.3, -0.4, 0.5])
    point = np.array(
        [
            greens_psi(target_r, target_z, 6.2, 0.0),
            greens_bz_br(target_r, target_z, 6.2, 0.0)[1],
            greens_bz_br(target_r, target_z, 6.2, 0.0)[0],
        ]
    )
    correction = {}
    for radius in (0.06, 0.006):
        got = np.array(moment_filament(target_r, target_z, _hexagon(radius=radius)))
        correction[radius] = np.abs(got - point) / np.abs(point)
    np.testing.assert_allclose(correction[0.06] / correction[0.006], 100.0, rtol=0.03)
    assert np.max(correction[0.06]) > 1e-4  # real at the shipped plasma cell size


def test_the_moment_filament_carries_the_cross_moment():
    """An asymmetric section needs the off-diagonal moment, so it is applied.

    Dropping the cross term is the cheap five-evaluation path, valid only when the
    section is symmetric about one of its own axes. On a section that is not, the
    term measurably closes the gap to the exact kernel, which is checked here by
    zeroing it: an axis-aligned diagonal-only correction is the worse model.
    """
    from nova.biot.greens import moment_filament, second_moments
    from nova.biot.polygon import polygon_greens

    vertices = _trapezium()
    assert abs(second_moments(vertices)[2]) > 1e-5
    angle = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target_r = 6.2 + 1.0 * np.cos(angle)
    target_z = 1.0 * np.sin(angle)

    exact = polygon_greens(target_r, target_z, vertices)
    corrected = moment_filament(target_r, target_z, vertices)
    diagonal = moment_filament(target_r, target_z, vertices, cross_moment=False)
    for index in range(3):
        scale = np.max(np.abs(exact[index]))
        full = np.max(np.abs(corrected[index] - exact[index])) / scale
        without = np.max(np.abs(diagonal[index] - exact[index])) / scale
        assert full < 1e-5
        assert without > 5.0 * full


def _wall_clipped(r0=6.2, z0=0.0, radius=0.06):
    """Return a hexagon with one corner cut by a straight wall.

    Simple (non-self-intersecting) and mildly asymmetric, which is what a plasma
    cell clipped by the first wall looks like: its third moments do not vanish,
    where a regular hexagon's do by symmetry.
    """
    v = list(_hexagon(r0, z0, radius))
    return np.array(
        v[:2]
        + [[r0 - 0.35 * radius, z0 + 0.75 * radius]]
        + [[r0 - 0.95 * radius, z0 + 0.30 * radius]]
        + v[3:]
    )


def _sampled_third_moments(vertices, count=1400):
    """Return area-normalised third central moments by dense grid sampling."""
    v = np.asarray(vertices, dtype=np.float64)
    grid_r, grid_z = np.meshgrid(
        np.linspace(v[:, 0].min(), v[:, 0].max(), count),
        np.linspace(v[:, 1].min(), v[:, 1].max(), count),
    )
    r, z = grid_r.ravel(), grid_z.ravel()
    inside = np.zeros(r.shape, dtype=bool)
    for (ra, za), (rb, zb) in zip(v, np.roll(v, -1, axis=0)):
        if zb == za:
            continue
        straddles = (za > z) != (zb > z)
        crossing = ra + (z - za) * (rb - ra) / (zb - za)
        inside ^= straddles & (r < crossing)
    r, z = r[inside], z[inside]
    r, z = r - r.mean(), z - z.mean()
    return (
        float(np.mean(r**3)),
        float(np.mean(r * r * z)),
        float(np.mean(r * z * z)),
        float(np.mean(z**3)),
    )


def test_the_third_moments_match_a_sampled_section():
    """The triangulated moments reproduce a dense sampling of the same polygon."""
    from nova.biot.greens import third_moments

    vertices = _wall_clipped()
    got = third_moments(vertices)
    expected = _sampled_third_moments(vertices)
    scale = max(abs(value) for value in expected)
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=1e-3 * scale)


def test_a_section_symmetric_about_its_centroid_has_no_third_moment():
    """A regular hexagon's odd moments vanish, so it pays nothing for them."""
    from nova.biot.greens import moment_filament, third_moments

    vertices = _hexagon()
    assert np.max(np.abs(third_moments(vertices))) < 1e-16
    target_r = np.array([6.8, 5.7, 6.2])
    target_z = np.array([0.4, -0.5, 0.6])
    for quadrupole, full in zip(
        moment_filament(target_r, target_z, vertices, order=2),
        moment_filament(target_r, target_z, vertices, order=3),
    ):
        np.testing.assert_array_equal(quadrupole, full)


def test_the_section_skew_dominates_an_asymmetric_cell_far_field():
    """On a wall-clipped cell the third moment is the leading far-field residual.

    Carrying it wins better than an order of magnitude at every distance. What it
    leaves behind is the fourth moment, so the bound is reached a little further
    out rather than never -- where the quadrupole form alone leaves a third-order
    residual that no practical band width recovers.
    """
    from nova.biot.greens import moment_filament
    from nova.biot.polygon import polygon_greens

    vertices = _wall_clipped()
    radius = float(np.max(np.hypot(*(vertices - vertices.mean(axis=0)).T)))
    angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    worst = {}
    for reach in (7.8, 16.0):
        target_r = 6.2 + reach * radius * np.cos(angle)
        target_z = reach * radius * np.sin(angle)
        exact = polygon_greens(target_r, target_z, vertices, n_panels=64, n_nodes=96)
        field = np.hypot(exact[1], exact[2])
        local = (np.abs(exact[0]), field, field)
        for order in (2, 3):
            got = moment_filament(target_r, target_z, vertices, order=order)
            worst[reach, order] = max(
                float(np.max(np.abs(got[index] - exact[index]) / local[index]))
                for index in range(3)
            )
    for reach in (7.8, 16.0):
        assert worst[reach, 3] < 0.2 * worst[reach, 2]
    assert worst[7.8, 3] > 1e-6
    assert worst[16.0, 3] < 1e-6


# A filament model cannot answer for a point ON the filament, and the one thing it
# must not do is answer for a point NEAR it instead. Both halves are pinned here.


def test_a_target_on_the_filament_returns_the_divergence():
    """No phantom standoff: the flux diverges and the field has no limit at all.

    ``psi`` grows without bound from every direction, so ``inf`` is its limit.
    ``B_Z`` does not have one -- approached radially in the plane its sign follows
    which side the target is on -- so ``nan`` is the honest answer rather than either
    infinity. What neither may be is a finite number: an absolute floor on the
    squared distance to the filament answers for a target 1.4 um away instead, with
    nothing in the result to say so, and it does that at every ring radius because
    the ring radius does not appear in it.
    """
    for radius in (0.9, 1.0, 6.2):
        target = np.array([radius])
        level = np.array([0.0])
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            psi = np.asarray(greens_psi(target, level, radius, 0.0))
            bz, br = (
                np.asarray(part) for part in greens_bz_br(target, level, radius, 0.0)
            )
            traced = traced_filament_greens(np, target, level, radius, 0.0)
        assert psi[0] == np.inf
        assert np.isnan(bz[0])
        assert np.isnan(br[0])
        assert traced[0][0] == np.inf
        assert np.isnan(traced[1][0])
        assert np.isnan(traced[2][0])


def test_one_ulp_off_the_filament_is_finite_at_every_ring_radius():
    """Adjacent is not coincident, and only the complement tells the two apart.

    The two radicals are formed independently, so their ratio -- the parameter --
    can land an ulp ABOVE one this close, and whether it does depends on the ring
    radius: a unit ring trips it one ULP off the filament where a 6.2 m ring does
    not. The second kind is held at one for that reason; the first kind never sees
    the parameter, taking the complement from the geometry instead, which is what
    makes these values right rather than merely finite.
    """
    for radius in (0.9, 1.0, 6.2, 12.0):
        target = np.array([np.nextafter(radius, np.inf)])
        level = np.array([0.0])
        psi = np.asarray(greens_psi(target, level, radius, 0.0))
        bz, br = (np.asarray(part) for part in greens_bz_br(target, level, radius, 0.0))
        assert np.isfinite(psi).all() and psi[0] > 0.0
        assert np.isfinite(bz).all() and bz[0] < 0.0
        assert np.isfinite(br).all()
        # the flux of a loop against itself one ulp away is 2 mu0 a log(8a/d)-ish:
        # bounded by the ring's own scale rather than by a floor
        assert 1e-5 < psi[0] / radius < 1e-3


# --- the two routes to the same kernel --------------------------------

_TWIN_RING = (6.2, 0.0)
_TWIN_SECTION = (6.2, 0.0, 0.1, 0.08)


def _spiral(offsets):
    """Return targets at the given distances from the ring point, one per angle.

    A spiral rather than a radial line: the two field brackets cancel
    differently above the ring, beside it and in its own plane, so a radial
    sweep would report one of the three and call it the kernel.
    """
    angle = np.linspace(0.0, 8.0 * np.pi, offsets.size)
    radius = _TWIN_RING[0] + offsets * np.cos(angle)
    return radius, _TWIN_RING[1] + offsets * np.sin(angle)


def _set_deviation(one, other):
    """Return the largest ``|one - other|`` over the largest magnitude in the set.

    Set-normalised, not element-wise: ``B_Z`` changes sign inside a wide sweep
    and ``B_R`` is identically zero in the ring's own plane, so an element-wise
    ratio measures where the zero sits rather than what the two routes did.
    """
    one, other = np.asarray(one, dtype=float), np.asarray(other, dtype=float)
    scale = np.abs(other).max()
    return 0.0 if scale == 0.0 else float((np.abs(one - other) / scale).max())


def test_the_point_filament_routes_agree_including_the_coincident_limit():
    """Cephes and the Bulirsch descent give the same filament, to round-off.

    The two forms of the point loop reach ``K`` and ``E`` by different routines
    -- ``scipy.special.ellipkm1``/``ellipe`` against the complement-native
    descent -- and this pins that the choice is a COST one and not a physics
    one, so neither can be tuned without the other noticing.

    The corner that decides it is ``k^2 -> 1``, a nanometre off a 6.2 m ring,
    where the complement runs to 1e-20 and the first kind grows like
    ``-log k'``: it is the configuration a self-coupling read and a plasma cell
    landing on a conductor both reach, and the one where a route that formed the
    parameter instead would have nothing left.
    """
    for offsets in (
        np.geomspace(1e-9, 1e-2, 400),  # k^2 -> 1: complement 1e-20 to 1e-6
        np.geomspace(1e-2, 1.0, 400),  # the hand-over band of a section kernel
        np.geomspace(1.0, 5.0, 400),  # diagnostic loops, the far side
    ):
        target_r, target_z = _spiral(offsets)
        host_psi = greens_psi(target_r, target_z, *_TWIN_RING)
        host_bz, host_br = greens_bz_br(target_r, target_z, *_TWIN_RING)
        psi, br, bz = traced_filament_greens(np, target_r, target_z, *_TWIN_RING)
        assert _set_deviation(psi, host_psi) < 1e-14
        assert _set_deviation(bz, host_bz) < 1e-14
        assert _set_deviation(br, host_br) < 1e-14


def test_the_point_filament_routes_agree_in_the_ring_plane():
    """The plane where both field brackets cancel hardest and ``B_R`` vanishes."""
    target_r = _TWIN_RING[0] + np.geomspace(1e-6, 12.0, 400)
    target_z = np.full(target_r.shape, _TWIN_RING[1])
    host_psi = greens_psi(target_r, target_z, *_TWIN_RING)
    host_bz, host_br = greens_bz_br(target_r, target_z, *_TWIN_RING)
    psi, br, bz = traced_filament_greens(np, target_r, target_z, *_TWIN_RING)
    assert _set_deviation(psi, host_psi) < 1e-14
    assert _set_deviation(bz, host_bz) < 1e-14
    np.testing.assert_array_equal(br, host_br)


def test_the_two_routes_share_the_filament_singularity_contract():
    """One ulp off the routes agree; on the filament both expose the singularity."""
    for radius in (0.9, 1.0, 6.2, 12.0):
        adjacent = np.array([np.nextafter(radius, np.inf)])
        level = np.array([0.0])
        host_psi = greens_psi(adjacent, level, radius, 0.0)
        host_bz, _ = greens_bz_br(adjacent, level, radius, 0.0)
        psi, _, bz = traced_filament_greens(np, adjacent, level, radius, 0.0)
        assert abs(psi[0] - host_psi[0]) < 1e-14 * abs(host_psi[0])
        assert abs(bz[0] - host_bz[0]) < 1e-14 * abs(host_bz[0])

        on_source = np.array([radius])
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            assert greens_psi(on_source, level, radius, 0.0)[0] == np.inf
            traced = traced_filament_greens(np, on_source, level, radius, 0.0)
        assert traced[0][0] == np.inf
        assert np.isnan(traced[1][0])
        assert np.isnan(traced[2][0])


def test_the_rectangular_section_routes_agree_through_the_quadrature_switch():
    """The two section forms differ only in the ``zeta`` rule, and not in the answer.

    Both reach all three complete kinds through the same descent, so what is
    pinned here is the quadrature: the host routes each element between a
    48-node Gauss-Legendre rule and a 177-node tanh-sinh one, the traced form
    takes tanh-sinh throughout.  ``corner_plane`` sweeps a target through a
    corner's own level and so crosses the switch, which is the only place the
    two rules can disagree; the four-corner combination differences four corner
    values against each other, so it is carried alongside the bare corner
    antiderivative in case that cancellation amplifies a per-corner difference.
    """
    a, z0, da, dz = _TWIN_SECTION
    grid = np.meshgrid(
        np.linspace(a - 0.4 * da, a + 0.4 * da, 20),
        np.linspace(z0 - 0.4 * dz, z0 + 0.4 * dz, 20),
    )
    gap = np.geomspace(1.0, 1e-9, 200)
    regimes = {
        "inside": tuple(axis.ravel() for axis in grid),
        "corner_plane": (
            np.full(2 * gap.size, a + 0.3 * da),
            z0 + dz / 2.0 + np.concatenate([gap, -gap]) * da,
        ),
        "near": _spiral(np.geomspace(0.6 * da, 4.0 * da, 400)),
        "standoff": _spiral(np.geomspace(1.0, 5.0, 400)),
    }
    for target_r, target_z in regimes.values():
        host = cylinder_greens(target_r, target_z, a, z0, da, dz)
        traced = traced_cylinder_greens(np, target_r, target_z, a, z0, da, dz)
        for one, other in zip(traced, host):
            assert _set_deviation(one, other) < 2e-12

        stacks = _corner_stacks(a, z0, da, dz, target_r, target_z)
        host_corner = corner_fields(*stacks)
        for one, other in zip(traced_corner_fields(np, *stacks), host_corner):
            assert _set_deviation(one, other) < 1e-14
