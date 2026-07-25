"""Accuracy of the zeta quadrature over the whole reduced-variable domain.

The zeta integral

    zeta(alpha) = integral_0^alpha arcsinh(beta_1(a)) da,
    beta_1(a) = (rs - r cos phi) / sqrt(gamma^2 + r^2 sin^2 phi),  phi = pi - 2 a

depends on the source/target geometry only through the reduced pair
``(rs / r, gamma / r)`` and the arc half-angle ``alpha``, so a scan over those
three fixes the accuracy of the rule everywhere.  ``r = 1`` throughout and the
scan values below cover the target inside, on, and outside the source radius,
in and far from the plane of the source corner, over the full folded
half-angle range ``alpha in [0, pi/2]``.

**Metric.**  Relative error against the reference, floored by
``REFERENCE_SCALE * 1e-15`` so that limits where the integral itself passes
through zero are judged on an absolute footing rather than dividing by nothing.

**Reference.**  A tanh-sinh rule two refinement levels and a wider truncation
beyond the production one, cross-validated in
:func:`test_reference_is_converged` against an independent adaptive quadrature.
Both endpoints of the interval are where ``sin phi`` vanishes, so when the
target lies in the plane of the source corner (``gamma -> 0``) the integrand
carries a logarithmic endpoint singularity; the reference has to resolve that
to be worth anything, which is why it is a tanh-sinh rule and not a finer
uniform one.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate

from nova.biot.zeta import (
    GAUSS_ORDER,
    NEAR_PLANE_RATIO,
    TANH_SINH_HALF_COUNT,
    _gauss_legendre_rule,
    _tanh_sinh_rule,
    zeta,
    zeta_midpoint,
)

HALF = np.pi / 2.0

RS = [0.2, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 5.0, 20.0]
GAMMA = [0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0]
ALPHA = [0.05, 0.3, np.pi / 4, HALF, 1.4]

# alpha just short of pi/2 puts the upper log singularity outside the interval
# but arbitrarily close to it -- the hardest configuration for any rule whose
# node clustering is tied to the interval ends
ALPHA_APPROACH = [HALF - delta for delta in (1e-2, 1e-3, 1e-5)]

REFERENCE_SCALE = 5.0
"""Rough upper bound on |zeta| over the scan; sets the absolute error floor."""

GATE = 1e-12
"""Relative accuracy the production rule must hold over the whole scan."""


def _cases(alphas):
    """Return the scan triples, dropping only the genuinely singular point.

    ``rs = r`` with ``gamma = 0`` puts the target on the conductor surface,
    where the integrand diverges non-integrably; every other corner of the
    scan, including the plane ``gamma = 0`` away from ``rs = r``, is included.
    """
    return [
        (rs, gamma, alpha)
        for rs in RS
        for gamma in GAMMA
        for alpha in alphas
        if not (rs == 1.0 and gamma == 0.0)
    ]


CASES = _cases(ALPHA)
APPROACH_CASES = _cases(ALPHA_APPROACH)


def _integrand(alpha, rs, gamma):
    """Return arcsinh(beta_1) at arc half-angle ``alpha`` for ``r = 1``."""
    phi = np.pi - 2.0 * alpha
    return np.arcsinh((rs - np.cos(phi)) / np.sqrt(gamma**2 + np.sin(phi) ** 2))


def _reference(rs, gamma, alpha):
    """Return zeta from a tanh-sinh rule refined well past the production one."""
    step = 4.0 / 512
    t = np.arange(-512, 513) * step
    u = 0.5 * np.pi * np.sinh(t)
    offset = 1.0 / (1.0 + np.exp(2.0 * np.abs(u)))
    weight = step * 0.5 * np.pi * np.cosh(t) / np.cosh(u) ** 2 / 2.0
    angle = 2.0 * alpha * offset
    angle = np.where(u > 0.0, np.pi - 2.0 * alpha + angle, angle)
    values = np.arcsinh(
        (rs - np.where(u > 0.0, np.cos(angle), -np.cos(angle)))
        / np.sqrt(gamma**2 + np.sin(angle) ** 2)
    )
    return alpha * np.sum(weight * values)


def _relative_error(got, want):
    """Return the scan error metric: relative, floored for vanishing integrals."""
    return np.abs(got - want) / np.maximum(np.abs(want), REFERENCE_SCALE * 1e-15)


def _scan(cases):
    """Return (rs, gamma, alpha, reference) arrays over ``cases``."""
    rs, gamma, alpha = (np.array(column) for column in zip(*cases, strict=True))
    want = np.array([_reference(*case) for case in cases])
    return rs, gamma, alpha, want


@pytest.mark.parametrize(
    "rs,gamma,alpha",
    [(0.2, 0.0, HALF), (0.2, 1e-6, HALF - 1e-3), (5.0, 0.1, 1.4), (20.0, 10.0, 0.05)],
)
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_reference_is_converged(rs, gamma, alpha):
    """The scan reference agrees with an independent adaptive quadrature.

    The adaptive rule reports roundoff-limited convergence on the endpoint
    singularity, which is why its own error estimate is asserted rather than
    trusted implicitly.
    """
    want, error = scipy.integrate.quad(
        _integrand, 0.0, alpha, args=(rs, gamma), epsabs=1e-15, epsrel=1e-14, limit=500
    )
    assert error < 1e-12
    assert _relative_error(_reference(rs, gamma, alpha), want) < 1e-13


def test_fast_path_meets_the_gate():
    """The production rule holds 1e-12 relative over the whole scan."""
    rs, gamma, alpha, want = _scan(CASES)
    error = _relative_error(zeta(rs, np.ones_like(rs), gamma, alpha), want)
    assert error.max() < GATE, f"worst case {CASES[int(error.argmax())]}"


def test_fast_path_holds_as_alpha_approaches_the_upper_singularity():
    """The gate survives alpha just short of pi/2, singularity outside the range."""
    rs, gamma, alpha, want = _scan(APPROACH_CASES)
    error = _relative_error(zeta(rs, np.ones_like(rs), gamma, alpha), want)
    assert error.max() < GATE, f"worst case {APPROACH_CASES[int(error.argmax())]}"


def test_near_plane_rule_carries_the_region_gauss_legendre_cannot():
    """Below the engagement threshold the fallback passes where plain GL fails.

    Fixes the reason the threshold exists: at ``gamma = 0`` the integrand is
    logarithmically singular at the interval ends, Gauss-Legendre degrades to
    algebraic convergence, and no production order reaches the gate.
    """
    rs, gamma, alpha, want = _scan(
        [case for case in CASES + APPROACH_CASES if case[1] < NEAR_PLANE_RATIO]
    )
    assert _relative_error(zeta(rs, np.ones_like(rs), gamma, alpha), want).max() < GATE

    offset, upper, weight = _gauss_legendre_rule()
    angle = 2.0 * alpha[:, np.newaxis] * offset
    angle = np.where(upper, np.pi - 2.0 * alpha[:, np.newaxis] + angle, angle)
    values = np.arcsinh(
        (rs[:, np.newaxis] - np.where(upper, np.cos(angle), -np.cos(angle)))
        / np.sqrt(gamma[:, np.newaxis] ** 2 + np.sin(angle) ** 2)
    )
    assert _relative_error(alpha * (values @ weight), want).max() > 1e-6


def test_gauss_legendre_covers_the_region_it_is_engaged_on():
    """Above the engagement threshold Gauss-Legendre alone meets the gate.

    The threshold is only worth its branch if the cheap rule really does carry
    everything on its side of it -- so drive the far region with plain
    Gauss-Legendre and no fallback.
    """
    rs, gamma, alpha, want = _scan(
        [case for case in CASES + APPROACH_CASES if case[1] >= NEAR_PLANE_RATIO]
    )
    offset, upper, weight = _gauss_legendre_rule()
    angle = 2.0 * alpha[:, np.newaxis] * offset
    angle = np.where(upper, np.pi - 2.0 * alpha[:, np.newaxis] + angle, angle)
    values = np.arcsinh(
        (rs[:, np.newaxis] - np.where(upper, np.cos(angle), -np.cos(angle)))
        / np.sqrt(gamma[:, np.newaxis] ** 2 + np.sin(angle) ** 2)
    )
    assert _relative_error(alpha * (values @ weight), want).max() < GATE


def test_engagement_threshold_keeps_a_factor_of_two_in_reserve():
    """Gauss-Legendre still meets the gate well below where it is switched off.

    Pins the margin the threshold was chosen with: if a future order change
    erodes it, this fails before the production gate does.
    """
    _, _, _, want = _scan(
        cases := [
            (rs, gamma, alpha)
            for rs in RS
            for gamma in (NEAR_PLANE_RATIO / 2.0,)
            for alpha in ALPHA + ALPHA_APPROACH
        ]
    )
    rs, gamma, alpha = (np.array(column) for column in zip(*cases, strict=True))
    offset, upper, weight = _gauss_legendre_rule()
    angle = 2.0 * alpha[:, np.newaxis] * offset
    angle = np.where(upper, np.pi - 2.0 * alpha[:, np.newaxis] + angle, angle)
    values = np.arcsinh(
        (rs[:, np.newaxis] - np.where(upper, np.cos(angle), -np.cos(angle)))
        / np.sqrt(gamma[:, np.newaxis] ** 2 + np.sin(angle) ** 2)
    )
    assert _relative_error(alpha * (values @ weight), want).max() < GATE


def test_node_counts_are_fixed():
    """Both rules carry a data-independent node count, as the batched path needs."""
    assert _gauss_legendre_rule()[0].size == GAUSS_ORDER
    assert _tanh_sinh_rule()[0].size == 2 * TANH_SINH_HALF_COUNT + 1
    for rule in (_gauss_legendre_rule(), _tanh_sinh_rule()):
        assert rule[2].sum() == pytest.approx(1.0, abs=1e-15)


def test_midpoint_reference_is_reproduced_where_it_is_accurate():
    """The retained midpoint rule agrees with the fast path far from the plane.

    The two rules are independent implementations of the same integral, so far
    from the source plane -- the only place a uniform rule is worth anything --
    they must agree to the midpoint rule's own accuracy.
    """
    cases = [case for case in CASES if case[1] >= 1.0]
    rs, gamma, alpha, want = _scan(cases)
    got = zeta_midpoint(rs, np.ones_like(rs), gamma, alpha)
    assert _relative_error(got, want).max() < 1e-6


def test_zeta_is_even_in_alpha_and_vanishes_at_zero():
    """Sign convention: the integrand is even, so only |alpha| enters."""
    rs, gamma, alpha, _ = _scan(CASES)
    ones = np.ones_like(rs)
    np.testing.assert_allclose(
        zeta(rs, ones, gamma, -alpha), zeta(rs, ones, gamma, alpha), rtol=0, atol=0
    )
    assert zeta(rs, ones, gamma, np.zeros_like(alpha)).max() == 0.0


def test_result_takes_the_broadcast_shape():
    """Inputs broadcast; the result carries the common shape, not alpha's."""
    rs = np.linspace(0.5, 2.0, 6).reshape(3, 2)
    got = zeta(rs, 1.0, 0.5, HALF)
    assert got.shape == (3, 2)
    np.testing.assert_allclose(
        got.ravel(),
        [zeta(value, 1.0, 0.5, HALF) for value in rs.ravel()],
        rtol=1e-15,
    )


def test_blocking_does_not_change_the_result():
    """Splitting a long array across working-set blocks holds full precision.

    The node-axis contraction is a matrix-vector product, so the block length
    can shift how the library associates the sum -- at the last bit, never more.
    """
    from nova.biot import zeta as module

    rs = np.linspace(0.3, 3.0, 5000)
    gamma = np.linspace(0.0, 2.0, 5000)
    alpha = np.linspace(0.0, HALF, 5000)
    ones = np.ones_like(rs)
    whole = zeta(rs, ones, gamma, alpha)
    block = module._MAX_BLOCK
    module._MAX_BLOCK = 4096
    try:
        split = zeta(rs, ones, gamma, alpha)
    finally:
        module._MAX_BLOCK = block
    np.testing.assert_allclose(whole, split, rtol=1e-15, atol=0.0)


def test_batched_path_matches_the_numpy_rule():
    """The JAX form evaluates the same integral as the numpy one.

    It runs the tanh-sinh rule unconditionally rather than switching per
    element, so the two agree to the gate rather than to the last bit.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from nova.biot.zeta import Zeta

    rs, gamma, alpha, want = _scan(CASES + APPROACH_CASES)
    zs, z = gamma, np.zeros_like(gamma)
    got = np.asarray(Zeta(rs, zs, np.ones_like(rs), z, alpha)())
    assert _relative_error(got, want).max() < GATE
    np.testing.assert_allclose(
        got, zeta(rs, np.ones_like(rs), gamma, alpha), rtol=GATE, atol=0.0
    )
