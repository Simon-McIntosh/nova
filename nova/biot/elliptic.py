"""Complete elliptic integrals and Jacobi moment recursions for the full turn.

Urankar's closed-form evaluation of a polygon-section conductor (Part V, IEEE
Trans. Magn. 26(3), 1171-1180, 1990) reduces each polygon edge's toroidal
integral to elliptic integrals through the angle substitution ``phi = pi - 2
alpha`` followed by a change to Jacobi elliptic functions.  In general that
leaves INCOMPLETE integrals with argument ``alpha``.  For the axisymmetric FULL
TURN it does not, and that is the simplification this module exists to exploit:

* the transformation maps ``phi`` over one period onto ``alpha_i = 2 pi``, which
  is the paper's case C (eq 26, ``3 pi/2 < alpha_i <= 2 pi``), giving
  ``H(alpha_i) = 4 H(pi/2) - H(theta_i)`` with ``theta_i = 2 pi - alpha_i = 0``;
* ``H(0)`` vanishes, so the whole ring is ``4 x`` the value at ``alpha = pi/2``;
* ``alpha = pi/2`` is ``u = K(k)``, so every elliptic integral is COMPLETE, and
  the Jacobi functions take their quarter-period values ``sn = 1``, ``cn = 0``,
  ``dn = k'``.

``cn K = 0`` is worth noting on its own: it annihilates every term in the
paper's expressions that carries an odd power of ``cn u``, which is exactly the
set of terms forming the radial component of the vector potential -- the
component that must vanish for an axisymmetric ring.  The algebra reproduces the
symmetry rather than assuming it, which makes it a useful check on a
transcription.

Everything here is parametrised by the modulus COMPLEMENT ``k'^2``, never by the
parameter alone.  A float parameter cannot carry its own complement -- ``1 - m``
is then known only to ``eps`` -- and ``K``, which grows like ``-log k'``, is wrong
by ``eps/k'^2`` if formed that way.  For a polygon section ``k'^2`` is the target's
squared distance to an edge's END over the squared ring span, so that error is what
a target approaching a section VERTEX sees, and it reaches order one a micron out.
Carlson's forms take the complement as their argument and a caller that knows it in
closed form keeps every digit.  ``k'^2 = 0`` -- the target ON the vertex -- is the
confluence, where ``K`` diverges logarithmically; see :func:`_complete_kind`.

The three moment families below are what the final expressions (eqs 21-24) are
built from.  They are collected here, separately testable against direct
quadrature, because a mis-transcribed recursion is invisible inside the full
assembly but obvious against a reference integral.
"""

from __future__ import annotations

import numpy as np
import scipy.special  # type: ignore[import-untyped]

__all__ = [
    "cn_pole_moment",
    "cn_pole_moments",
    "complete_pi",
    "harmonic_moments",
    "harmonic_pole_moments",
    "harmonic_root_moments",
    "pole_moment",
    "pole_moments",
    "sn_moments",
    "sn_cn_moments",
    "sn_pole_moment",
    "sn_pole_moments",
    "stable_cn_moments",
    "stable_sn_moments",
]

# Below this the upward moment recursions divide by a small number once per
# order while the downward ones multiply by it; above it the roles swap.  One
# threshold serves both the parameter and a pole characteristic because the two
# recursions have the same shape.
_SWITCH = 0.5

# A pole further than this past the end of the integration range leaves the
# shifted recursion's upward direction -- which multiplies the error by the shift
# once per order -- for its downward one, which divides by it.
_SHIFT_SWITCH = 2.0

# The complement-basis recursion's parasitic branch is ``-k'^2/k^2``, so its
# upward direction grows by that factor per order and its downward one decays by
# the reciprocal.  The switch is NOT free: a shifted family running downward
# consumes plain moments up to the headroom order and divides by the shift once
# per order, so it survives an upward-grown plain moment only while the growth per
# order stays under the smallest shift that runs downward.  That fixes the switch
# at ``1/(1 + _SHIFT_SWITCH)``, where ``k'^2/k^2 = _SHIFT_SWITCH`` -- and it lands
# comfortably inside the downward direction's own requirement, a decay of one half
# per order at the same point.
_COMPLEMENT_SWITCH = 1.0 / (1.0 + _SHIFT_SWITCH)

# Orders carried past the last one wanted when a recursion runs downward from an
# arbitrary seed.  The parasitic branch decays by roughly the switch value per
# order, so this many take it below round-off.
_HEADROOM = 60

# Where the harmonic family leaves its downward direction for its upward one.
# The branch ratio there is ``(c - sqrt(c^2 - 1))^2`` with ``c = (1 + k'^2)/k^2``,
# which is 0.67 per order at this parameter: the headroom below takes that under
# round-off, and the upward direction -- which grows by the reciprocal -- costs
# under an order over the harmonics a numerator of degree eight reaches.
_HARMONIC_SWITCH = 0.99
_HARMONIC_HEADROOM = 96

# Orders the harmonic pole family is carried past the last one wanted before the
# system is closed.  A root a quarter of the range past its end -- the point below
# which a caller takes the pole's weight out exactly instead -- makes the family
# decay by 0.38 per order, so this many put the closure's own error under
# round-off.
POLE_HEADROOM = 32


def _complete_kind(complement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, E)`` from the modulus complement ``k'^2``.

    ``K = R_F(0, k'^2, 1)`` and ``E = 2 R_G(0, k'^2, 1)``: Carlson's forms take the
    complement as their argument, so a caller that knows it exactly keeps every
    digit, where ``scipy.special.ellipk`` of a float parameter cannot -- at
    ``k'^2 = 1e-14`` the two disagree in the fifth decimal.

    ``k'^2 = 0`` is the confluence itself: the target sits ON the end of the edge
    whose integral this is, the modulus reaches one and ``K`` diverges
    logarithmically.  It is returned as ZERO, which is not a floor but the whole
    evaluation.  Every moment family below carries that one divergence with a
    weight of one, so returning zero for it returns each family's FINITE PART; and
    the reduction that consumes them puts a total weight of ZERO on the divergence,
    because the flux and field of a section are bounded at its own corner.  The
    answer is therefore linear in whatever value is assigned here with a slope of
    zero, and zero is the assignment that evaluates the finite part directly --
    with no cancellation left to round off, rather than a large number cancelling
    against itself.  ``E`` stays finite, ``E(1) = 1``.
    """
    complement = np.asarray(complement, dtype=np.float64)
    reachable = complement > 0.0
    held = np.where(reachable, complement, 1.0)
    return (
        np.where(reachable, scipy.special.elliprf(0.0, held, 1.0), 0.0),
        2.0 * scipy.special.elliprg(0.0, complement, 1.0),
    )


def complete_pi(
    characteristic: np.ndarray,
    parameter: np.ndarray,
    *,
    complement: np.ndarray | None = None,
    parameter_complement: np.ndarray | None = None,
) -> np.ndarray:
    """Return the complete elliptic integral of the third kind, Pi(n | m).

    Carlson symmetric forms, as the rectangular-section kernel already uses:
    ``Pi(n | m) = R_F(0, 1 - m, 1) + (n/3) R_J(0, 1 - m, 1, 1 - n)``.  Both
    arguments are the PARAMETER convention (``m = k^2``), matching
    :func:`scipy.special.ellipk`.

    ``complement`` supplies ``1 - n`` directly.  ``Pi`` grows like
    ``(1 - n)^(-1/2)``, so forming the complement here caps the relative accuracy
    at ``eps / (1 - n)`` -- eight digits gone by ``1 - n = 1e-8``.  A caller that
    knows the complement in closed form keeps every digit by passing it, and the
    polygon reduction does know it: its characteristics are ratios whose
    complements are squares of small edge offsets.

    ``parameter_complement`` supplies ``1 - m`` for the same reason.  A float
    parameter cannot carry its own complement to better than ``eps``, so forming
    it here costs ``eps / (1 - m)`` however the parameter was computed -- and
    ``1 - m`` is the target's squared offset from the source ring divided by its
    squared span, which falls as the square of a section's aspect ratio.
    """
    characteristic = np.asarray(characteristic, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    if complement is None:
        complement = 1.0 - characteristic
    if parameter_complement is None:
        parameter_complement = 1.0 - parameter
    rf = scipy.special.elliprf(0.0, parameter_complement, 1.0)
    rj = scipy.special.elliprj(0.0, parameter_complement, 1.0, complement)
    return rf + rj * characteristic / 3.0


def harmonic_moments(
    parameter: np.ndarray,
    count: int,
    *,
    complement: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return ``[P_0, P_1, ...]``, ``P_n = integral_0^(pi/2) cos(2 n a)/Delta da``.

    The moments of the HARMONICS rather than of a power of ``sn`` or ``cn``, and
    what a numerator built as a cosine series contracts against.  The two power
    families are the natural basis for the paper's algebra and the wrong one for
    floating point: a polynomial of degree eight that stays of order one over the
    range reaches monomial coefficients of order ``10^4`` times its own size, so
    contracting it against a family of same-signed moments forms the answer as a
    sum of terms that exceed it by as much.  Cosines are bounded by one and their
    moments fall off, so the same contraction adds terms no larger than the
    result.

    Only ABSOLUTE accuracy of order ``eps K`` is wanted here, not relative: the
    high harmonics are small and multiply coefficients no larger than the low
    ones, so an error that is tiny beside ``P_0`` is tiny beside the contraction
    however large it is beside ``P_n`` itself.  That is what makes both directions
    below usable.

    The recursion comes from ``[sin 2na Delta]`` vanishing at both ends of the
    quarter period, with ``Delta^2 = (1 + k'^2)/2 + (k^2/2) cos 2a`` folding the
    root back onto the family:

        (2n + 1) k^2 P(n+1) + 4n (1 + k'^2) P(n) + (2n - 1) k^2 P(n-1) = 0

    Its two branches are reciprocal, so ``P_n`` -- which decays, being the Fourier
    series of a function analytic on the range -- is always the MINIMAL solution
    and the upward direction is always the unstable one.  Downward is therefore
    the direction of choice, and it is run on the RATIOS ``P_n/P(n-1)``, which
    stay bounded where the moments themselves would overflow: a far target makes
    the growth per downward step ``4/k^2``, which over a useful headroom is past
    any exponent range.  The ratio recursion also carries ``k^2 = 0`` exactly,
    returning zero for every harmonic above the mean.

    Downward fails only as ``k^2 -> 1``, where the two branches become degenerate
    -- and there the upward direction costs nothing for the same reason, provided
    it is run on ``R_n = P_n - (-1)^n K``.  That difference is what stays bounded
    at the confluence: ``cos 2na`` equals ``(-1)^n`` where ``Delta`` vanishes, so
    every ``P_n`` carries the SAME logarithmic divergence and their difference from
    it does not.  ``R`` obeys the same recursion with ``-(-1)^n 8 n k'^2 K`` on the
    right, and at the confluence itself that source vanishes with ``k'^2``.

    ``complement`` supplies ``k'^2``; see :func:`_complete_kind` for why it must be
    given rather than formed, and for the finite-part convention at ``k'^2 = 0``.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    if complement is None:
        complement = 1.0 - parameter
    complement = np.asarray(complement, dtype=np.float64) + np.zeros_like(parameter)
    complete_k, complete_e = _complete_kind(complement)
    degenerate = parameter > _HARMONIC_SWITCH

    held = np.where(degenerate, parameter, 1.0)
    held_complement = np.where(degenerate, complement, 0.0)
    upward = [
        np.zeros_like(parameter),
        2.0 * (complete_e - held_complement * complete_k) / held,
    ]
    for order in range(1, count - 1):
        upward.append(
            -(
                4.0 * order * (1.0 + held_complement) * upward[order]
                + (2 * order - 1) * held * upward[order - 1]
                + (-1.0) ** order * 8.0 * order * held_complement * complete_k
            )
            / ((2 * order + 1) * held)
        )

    ratio = np.zeros_like(parameter)
    ratios: list[np.ndarray] = [None] * (count + _HARMONIC_HEADROOM + 1)  # type: ignore[list-item]
    for order in range(count + _HARMONIC_HEADROOM, 0, -1):
        ratio = (
            -(2 * order - 1)
            * parameter
            / ((2 * order + 1) * parameter * ratio + 4.0 * order * (1.0 + complement))
        )
        if order <= count:
            ratios[order] = ratio
    downward = [complete_k]
    for order in range(1, count):
        downward.append(downward[order - 1] * ratios[order])
    return [
        np.where(
            degenerate,
            upward[order] + (-1.0) ** order * complete_k,
            downward[order],
        )
        for order in range(count)
    ]


def harmonic_pole_moments(
    shift: np.ndarray,
    seed: np.ndarray,
    moments: list[np.ndarray],
    count: int,
    *,
    mirrored: bool = False,
) -> list[np.ndarray]:
    """Return ``V_n = integral C_n/((cos^2 a + shift) Delta) da``, or its mirror.

    ``mirrored`` puts the pole factor on ``sin^2 a`` instead, so the root sits past
    the other end of the range.  ``seed`` is ``V_0``, which
    :func:`cn_pole_moment` and :func:`sn_pole_moment` give in closed form.

    Multiplying the definition back by the pole factor folds each harmonic onto
    its two neighbours -- ``cos^2 a = (1 + cos 2a)/2`` and
    ``cos 2m a cos 2n a`` splits -- so the family satisfies a three-term relation
    driven by the plain moments,

        V(n-1) + s(2 + 4 shift) V(n) + V(n+1) = 4 s P(n),   s = +/- 1

    with a diagonal of at least two against off-diagonals of one.  The system is
    therefore diagonally dominant and solved directly, closing it at a high order
    where the wanted solution -- which decays like the reciprocal of
    ``(1 + 2 shift) + sqrt((1 + 2 shift)^2 - 1)`` per order -- has died away.  That
    decay is what sets the headroom, and it is only fast enough for a root a
    reasonable distance past the range; for a root ON the range end the family is
    dominated by its seed anyway and the caller takes it out exactly instead.
    """
    sign = -1.0 if mirrored else 1.0
    diagonal = sign * (2.0 + 4.0 * shift)
    top = count + POLE_HEADROOM
    ratio = [1.0 / diagonal]
    solution = [(4.0 * sign * moments[1] - seed) / diagonal]
    for order in range(2, top + 1):
        pivot = diagonal - ratio[-1]
        ratio.append(1.0 / pivot)
        solution.append((4.0 * sign * moments[order] - solution[-1]) / pivot)
    values: list[np.ndarray] = [None] * (top + 1)  # type: ignore[list-item]
    values[top] = solution[top - 1]
    for order in range(top - 1, 0, -1):
        values[order] = solution[order - 1] - ratio[order - 1] * values[order + 1]
    values[0] = seed
    return values[:count]


def harmonic_root_moments(
    moments: list[np.ndarray], parameter: np.ndarray
) -> list[np.ndarray]:
    """Return ``[D_0, ...]``, ``D_n = integral_0^(pi/2) cos(2 n a) Delta da``.

    ``Delta^2`` is itself the first two harmonics, ``(1 + k'^2)/2 + (k^2/2) cos
    2a``, so multiplying it back under the integral folds each root moment onto
    three neighbouring reciprocal ones -- no new special function, and the
    ``P(-1) = P(1)`` reflection covers the mean.  ``moments`` must carry one order
    past the last root moment wanted.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    mean = 1.0 - 0.5 * parameter
    return [
        mean * moments[order]
        + 0.25 * parameter * (moments[order + 1] + moments[abs(order - 1)])
        for order in range(len(moments) - 1)
    ]


def stable_sn_moments(
    parameter: np.ndarray, count: int, *, complement: np.ndarray | None = None
) -> list[np.ndarray]:
    """Return ``[El0, El2, ...]`` accurately at every order and every parameter.

    ``complement`` supplies ``k'^2`` for the seeds, which is the only place the
    modulus enters other than as a nearly-unit multiplier: see
    :func:`_complete_kind`.  At the confluence the seeds become ``(0, -E)`` and the
    recursion carries every order's finite part exactly, the divergent branch of the
    ``k^2 = 1`` recursion being the constant one.

    The same quantity :func:`sn_moments` returns, from the same three-term
    recursion, run in whichever direction is accurate:

    * UPWARD, as printed, divides by ``k^2`` once per order.  Accurate while
      ``k^2`` is not small, losing about a digit per order once it is -- and the
      small-parameter limit is a distant target.
    * DOWNWARD from an arbitrary seed converges on the wanted branch because the
      parasitic one decays as ``k^(2m)``.  That fails as ``k^2 -> 1``, where the
      recursion's two branches (``1`` and ``k^(-2m)``) become degenerate and no
      practical headroom separates them -- and ``k^2 -> 1`` is the
      target-on-the-source-ring limit, which is common rather than exotic.

    Neither direction spans the range, so each is used where it holds and the
    result is selected per element.  The downward branch is normalised by
    ``El0 = K``, which the recursion cannot fix, being homogeneous.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    if complement is None:
        complement = 1.0 - parameter
    complete_k, complete_e = _complete_kind(complement + np.zeros_like(parameter))
    small = parameter < _SWITCH

    held = np.where(small, _SWITCH, parameter)
    upward = [complete_k, (complete_k - complete_e) / held]
    for order in range(1, count - 1):
        upward.append(
            (
                2.0 * order * (1.0 + held) * upward[order]
                - (2 * order - 1) * upward[order - 1]
            )
            / ((2 * order + 1) * held)
        )

    held = np.where(small, parameter, 0.5 * _SWITCH)
    top = count + _HEADROOM
    downward: list[np.ndarray] = [None] * (top + 2)  # type: ignore[list-item]
    downward[top + 1] = np.zeros_like(parameter)
    downward[top] = np.ones_like(parameter)
    for order in range(top, 0, -1):
        downward[order - 1] = (
            2.0 * order * (1.0 + held) * downward[order]
            - (2 * order + 1) * held * downward[order + 1]
        ) / (2 * order - 1)
    scale = complete_k / downward[0]
    return [
        np.where(small, downward[order] * scale, upward[order])
        for order in range(count)
    ]


def stable_cn_moments(
    parameter: np.ndarray,
    count: int,
    *,
    complement: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return ``[A0, A1, ...]``, the even cn moments over a quarter period.

    ``A_m = integral_0^K cn^(2m)(u, k) du = integral_0^(pi/2) cos^(2m) a / Delta
    da`` -- the mirror of :func:`stable_sn_moments` about the far end of the
    range, and what a numerator expanded in ``cos^2 a`` contracts against.

    Why the mirror is needed at all.  The pole whose moments diverge as a section
    thins sits just past ``a = pi/2``, and there every ``sin^(2m) a`` equals one.
    A numerator contracted against the plain moments therefore reconstructs its
    value AT that end as an alternating sum of coefficients of order one -- and
    that value is itself of order the squared aspect ratio, so the sum throws away
    two decades per pole family.  Expanded in ``cos^2 a`` the same value is the
    LEADING COEFFICIENT, formed once from the geometry.

    ``sn cn^(2m+1) dn`` vanishes at both ends of the quarter period, so its
    derivative integrates to nothing, which gives the three-term recursion

        (2m + 3) k^2 A(m+2) = (2m + 1) k'^2 A(m) - (2m + 2)(k'^2 - k^2) A(m+1),
        A0 = K,   A1 = (E - k'^2 K)/k^2.

    Its two branches are ``1`` and ``-k'^2/k^2``, so the parasitic one decays
    upward exactly where the parameter exceeds its complement and downward where
    it does not.  Neither direction spans the range, so each is used where it
    holds and the result selected per element, as in :func:`stable_sn_moments`;
    the downward branch is normalised on ``A0 = K``, which the recursion cannot
    fix, being homogeneous.

    ``complement`` supplies ``k'^2``, which this recursion carries explicitly as
    well as needing for its seeds; at the confluence they become ``(0, E)``, and
    ``k'^2`` multiplying the divergent ``A0`` in the recursion drops it out of every
    higher order, so all of those are exact.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    if complement is None:
        complement = 1.0 - parameter
    complement = np.asarray(complement, dtype=np.float64) + np.zeros_like(parameter)
    complete_k, complete_e = _complete_kind(complement)
    small = parameter < _COMPLEMENT_SWITCH

    held = np.where(small, _COMPLEMENT_SWITCH, parameter)
    held_complement = np.where(small, 1.0 - _COMPLEMENT_SWITCH, complement)
    upward = [complete_k, (complete_e - held_complement * complete_k) / held]
    for order in range(count - 2):
        upward.append(
            (
                (2 * order + 1) * held_complement * upward[order]
                - (2 * order + 2) * (held_complement - held) * upward[order + 1]
            )
            / ((2 * order + 3) * held)
        )

    held = np.where(small, parameter, 0.5 * _COMPLEMENT_SWITCH)
    held_complement = np.where(small, complement, 1.0 - 0.5 * _COMPLEMENT_SWITCH)
    top = count + _HEADROOM
    downward: list[np.ndarray] = [None] * (top + 2)  # type: ignore[list-item]
    downward[top + 1] = np.zeros_like(parameter)
    downward[top] = np.ones_like(parameter)
    for order in range(top, 0, -1):
        downward[order - 1] = (
            2 * order * (held_complement - held) * downward[order]
            + (2 * order + 1) * held * downward[order + 1]
        ) / ((2 * order - 1) * held_complement)
    scale = complete_k / downward[0]
    return [
        np.where(small, downward[order] * scale, upward[order])
        for order in range(count)
    ]


def _mirrored_pi(
    characteristic: np.ndarray,
    *,
    complement: np.ndarray,
    parameter_complement: np.ndarray,
) -> np.ndarray:
    """Return ``integral_0^(pi/2) da/((1 - n cos^2 a) sqrt(1 - m sin^2 a))``.

    The third-kind integral with its pole factor reflected onto the other end of
    the range.  Reflecting the angle turns it into an ordinary ``Pi`` of NEGATIVE
    parameter, ``Pi(n | -m/m')/sqrt(m')``, and Carlson's homogeneity collapses
    that to ``K + (n m'/3) R_J(0, m', 1, m'(1 - n))`` -- two positive terms.

    That is the point.  The same integral IS an ordinary ``Pi`` at a
    characteristic of ``-1/shift``, but there ``R_F`` and ``(n/3) R_J`` are of
    opposite sign and nearly equal: their sum falls as ``sqrt(shift)`` while each
    stays of order ``K``, so a shift of 1e-10 costs five digits.  Written this way
    the shift appears only as ``m'(1 - n)`` inside ``R_J``, where it makes the term
    large rather than cancelling.

    At the confluence ``m' = 0`` the whole integral is elementary, because
    ``sqrt(1 - m sin^2 a)`` collapses onto ``cos a`` and the substitution
    ``w = sin a`` leaves a rational function whose partial fractions are exactly the
    two terms above: ``sqrt(n/n') arctan sqrt(n/n')`` plus ``K``.  The finite part is
    therefore the arctangent term alone, and it is wanted rather than skipped even
    where the caller's weight on this pole family is zero, because ``R_J`` diverges
    there and a vanishing weight against it is not zero but undefined.
    """
    reachable = np.asarray(parameter_complement, dtype=np.float64) > 0.0
    held = np.where(reachable, parameter_complement, 1.0)
    root = np.sqrt(characteristic / complement)
    return np.where(
        reachable,
        scipy.special.elliprf(0.0, held, 1.0)
        + (characteristic * held / 3.0)
        * scipy.special.elliprj(0.0, held, 1.0, held * complement),
        root * np.arctan(root),
    )


def _shifted_family(
    shift: np.ndarray,
    leading: np.ndarray,
    moments: list[np.ndarray],
    count: int,
) -> list[np.ndarray]:
    """Return ``T_m = M(m-1) - shift T(m-1)`` from its seed ``T_0 = leading``.

    ``M`` is whichever plain moment family the numerator is to be expanded in, and
    the relation follows from writing ``t^m = t^(m-1)(t + shift) - shift
    t^(m-1)``: it is EXACT, so a whole pole family costs one special function for
    its seed and nothing else.  Upward it multiplies an error by the shift once
    per order and downward divides by it, so the direction switches on the shift
    -- and downward needs no seed at all, the parasitic branch decaying away from
    an arbitrary one carried past the top order.
    """
    shape = np.broadcast_shapes(
        np.shape(shift), np.shape(leading), np.shape(moments[0])
    )
    near = np.broadcast_to(shift <= _SHIFT_SWITCH, shape)

    rising = np.where(near, shift, 0.0)
    upward = [leading + np.zeros(shape)]
    for order in range(1, count):
        upward.append(moments[order - 1] - rising * upward[order - 1])

    falling = np.where(near, 1.0, shift)
    top = count + _HEADROOM
    downward: list[np.ndarray] = [None] * (top + 1)  # type: ignore[list-item]
    downward[top] = np.zeros(shape)
    for order in range(top, 0, -1):
        downward[order - 1] = (moments[order - 1] - downward[order]) / falling
    return [np.where(near, upward[order], downward[order]) for order in range(count)]


def cn_pole_moments(
    shift: np.ndarray,
    parameter: np.ndarray,
    count: int,
    *,
    moments: list[np.ndarray] | None = None,
    parameter_complement: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return ``T_m = integral cos^(2m) a/((cos^2 a + shift) Delta) da``.

    The complement-basis pole family: a denominator whose root sits a distance
    ``shift`` past the ``a = pi/2`` end of the range, taken in the basis that
    vanishes AT that end so a numerator's value there is a coefficient rather than
    a sum.  Only the seed needs a special function --
    ``cos^2 a + shift = (1 + shift)(1 - n sin^2 a)`` with ``n = 1/(1 + shift)``,
    whose complement ``shift/(1 + shift)`` is exact, so ``T_0 = Pi(n | k^2)/(1 +
    shift)`` -- and :func:`_shifted_family` carries the rest.

    ``shift = 0`` puts the root ON the end, where ``T_0`` diverges; it is returned
    as zero.  A numerator can only reach that configuration with its own leading
    coefficient exactly zero, because the pole factor and the numerator share the
    geometric quantity that vanishes there, and every higher moment stays finite.

    ``moments`` accepts an already-computed :func:`stable_cn_moments` list of at
    least ``count + 60`` entries, so several pole families over one parameter pay
    for the plain moments once.
    """
    shift = np.asarray(shift, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    if moments is None:
        moments = stable_cn_moments(
            parameter, count + _HEADROOM, complement=parameter_complement
        )
    return _shifted_family(
        shift,
        cn_pole_moment(shift, parameter, parameter_complement=parameter_complement),
        moments,
        count,
    )


def cn_pole_moment(
    shift: np.ndarray,
    parameter: np.ndarray,
    *,
    parameter_complement: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``integral_0^(pi/2) da/((cos^2 a + shift) Delta)``, the family's seed.

    ``cos^2 a + shift = (1 + shift)(1 - n sin^2 a)`` with ``n = 1/(1 + shift)``,
    whose complement ``shift/(1 + shift)`` is exact, so this is one ``Pi``.  It is
    also the ONLY special function a whole complement-basis pole family needs, and
    the only place a reduction's pole weight is multiplied by something that grows
    as the root approaches the range -- so it is separated out here for callers
    that want the seed alone.  ``shift = 0`` puts the root ON the end, where the
    integral diverges; zero is returned, on the finite-part convention of
    :func:`_complete_kind`.
    """
    shift = np.asarray(shift, dtype=np.float64)
    positive = shift > 0.0
    held = np.where(positive, shift, 1.0)
    return np.where(
        positive,
        complete_pi(
            1.0 / (1.0 + held),
            parameter,
            complement=held / (1.0 + held),
            parameter_complement=parameter_complement,
        )
        / (1.0 + held),
        0.0,
    )


def sn_pole_moments(
    shift: np.ndarray,
    parameter: np.ndarray,
    count: int,
    *,
    moments: list[np.ndarray] | None = None,
    parameter_complement: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return ``T_m = integral sin^(2m) a/((sin^2 a + shift) Delta) da``.

    The plain-basis mirror of :func:`cn_pole_moments`, for a root a distance
    ``shift`` past the ``a = 0`` end, and the seed is the mirror too:
    ``sin^2 a + shift = (1 + shift)(1 - n cos^2 a)`` with the SAME
    ``n = 1/(1 + shift)`` and complement ``shift/(1 + shift)``, only with the pole
    factor on ``cos^2 a`` instead of ``sin^2 a`` -- so ``T_0`` is
    :func:`_mirrored_pi` at that characteristic over ``1 + shift``, and everything
    after it is the same recursion.  ``shift = 0`` returns zero for the seed on the
    same reasoning as the complement family.

    Writing the factor as ``shift (1 - n sin^2 a)`` with ``n = -1/shift`` gives the
    same integral as an ordinary ``Pi``, and that is what
    :func:`_mirrored_pi` exists to avoid: at a hugely negative characteristic the
    Carlson form is a difference.
    """
    shift = np.asarray(shift, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    if parameter_complement is None:
        parameter_complement = 1.0 - parameter
    if moments is None:
        moments = stable_sn_moments(
            parameter, count + _HEADROOM, complement=parameter_complement
        )
    return _shifted_family(
        shift,
        sn_pole_moment(shift, parameter, parameter_complement=parameter_complement),
        moments,
        count,
    )


def sn_pole_moment(
    shift: np.ndarray,
    parameter: np.ndarray,
    *,
    parameter_complement: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``integral_0^(pi/2) da/((sin^2 a + shift) Delta)``, the family's seed.

    The mirror of :func:`cn_pole_moment`: the same characteristic with the pole
    factor on ``cos^2 a``, which is what :func:`_mirrored_pi` evaluates without the
    difference an ordinary ``Pi`` at a hugely negative characteristic would form.
    """
    shift = np.asarray(shift, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    if parameter_complement is None:
        parameter_complement = 1.0 - parameter
    positive = shift > 0.0
    held = np.where(positive, shift, 1.0)
    return np.where(
        positive,
        _mirrored_pi(
            1.0 / (1.0 + held),
            complement=held / (1.0 + held),
            parameter_complement=parameter_complement,
        )
        / (1.0 + held),
        0.0,
    )


def pole_moments(
    characteristic: np.ndarray,
    parameter: np.ndarray,
    count: int,
    *,
    complement: np.ndarray | None = None,
    moments: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Return ``V_m = integral_0^(pi/2) x^m / ((1 - n x) Delta) da``, ``x = sin^2 a``.

    ``Delta = sqrt(1 - k^2 x)``.  These are what a polygon edge's denominators
    reduce to: the toroidal integration leaves rational factors in ``x``, each of
    which splits into linear factors contributing one family here.

    ``V_(m-1) - n V_m = El_(m-1)`` ties the family to the plain moments and is
    accurate in one direction at a time -- upward from ``V_0 = Pi(n | k^2)``
    divides by ``n``, downward multiplies by it -- so the routes split on ``|n|``
    exactly as :func:`stable_sn_moments` splits on the parameter.  The downward
    branch needs no ``Pi`` at all: seeded past the top order it converges from
    above, which is why a characteristic of ``1e-3`` costs no special function.

    ``moments`` accepts an already-computed :func:`stable_sn_moments` list (of at
    least ``count + 60`` entries) so a caller with several pole families over one
    parameter pays for the plain moments once.
    """
    characteristic = np.asarray(characteristic, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    if moments is None:
        moments = stable_sn_moments(parameter, count + _HEADROOM)
    if complement is None:
        complement = 1.0 - characteristic
    shape = np.broadcast_shapes(np.shape(characteristic), np.shape(parameter))
    strong = np.broadcast_to(np.abs(characteristic) >= _SWITCH, shape)

    held = np.where(strong, 0.0, characteristic)
    downward: list[np.ndarray] = [None] * len(moments)  # type: ignore[list-item]
    downward[-1] = moments[-1] + np.zeros(shape)
    for order in range(len(moments) - 2, -1, -1):
        downward[order] = moments[order] + held * downward[order + 1]

    upward = [
        complete_pi(
            np.where(strong, characteristic, _SWITCH),
            parameter,
            complement=np.where(strong, complement, 1.0 - _SWITCH),
        )
    ]
    held = np.where(strong, characteristic, 1.0)
    for order in range(1, count):
        upward.append((upward[order - 1] - moments[order - 1]) / held)
    return [np.where(strong, upward[order], downward[order]) for order in range(count)]


def sn_moments(parameter: np.ndarray, count: int) -> list[np.ndarray]:
    """Return ``[El0, El2, El4, ...]``, the even sn moments over a quarter period.

    ``El2m = integral_0^K sn^(2m)(u, k) du`` -- the paper's ``El2m`` (eq 24b) at
    the complete limit, where the ``sn^(2m+1) cn dn`` boundary term drops
    because ``cn K = 0``:

        El0 = K,   k^2 El2 = K - E,
        (2m + 1) k^2 El(2m+2) = 2m (1 + k^2) El(2m) - (2m - 1) El(2m-2).

    ``count`` is the number of moments returned, so ``count=5`` gives El0 to El8
    -- the highest order the vector potential needs.

    Conditioning: each step divides by ``k^2``, so the recursion loses roughly
    one digit per moment as ``k^2 -> 0`` (a target far from the ring).  The
    seeds are the same story: ``El2 = (K - E)/k^2`` is a cancelling difference.
    A far-field evaluation must either carry the small-``k`` series or reach the
    far field by another route; this returns the recursion as printed and the
    accompanying test pins the parameter range over which it holds.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    complete_k = scipy.special.ellipk(parameter)
    complete_e = scipy.special.ellipe(parameter)
    moments = [complete_k, (complete_k - complete_e) / parameter]
    for order in range(1, count - 1):
        moments.append(
            (
                2.0 * order * (1.0 + parameter) * moments[order]
                - (2 * order - 1) * moments[order - 1]
            )
            / ((2 * order + 1) * parameter)
        )
    return moments[:count]


def sn_cn_moments(parameter: np.ndarray, count: int) -> list[np.ndarray]:
    """Return ``[I1, I3, I5, ...]``, the odd Jacobi moments over a quarter period.

    ``I(2m+1) = integral_0^K (k sn u)^(2m) k^2 sn u cn u du`` -- the paper's
    ``I_m(x)`` with ``x = k sn u`` (eq 24a), evaluated between ``0`` and ``K``.
    The paper gives it as the recursion ``(2m + 1) I(2m+1) = -k^(2m) k' + 2m
    I(2m-1)`` seeded by ``I1 = 1 - k'``, which follows from ``k^2 sn u cn u =
    -d(dn u)/du``.

    That same observation removes the Jacobi functions altogether.  Substituting
    ``v = dn u`` gives ``(k sn u)^(2m) = (1 - v^2)^m`` and ``k^2 sn u cn u du =
    -dv``, so ``I(2m+1) = integral_(k')^1 (1 - v^2)^m dv`` -- a polynomial
    integral.  Shifting once more to ``w = 1 - v``, which runs over ``[0,
    delta]`` with ``delta = 1 - k'``, puts the whole thing in powers of the
    small quantity:

        I(2m+1) = integral_0^delta (w (2 - w))^m dw
                = delta^(m+1) sum_j C(m, j) (-1)^j 2^(m-j) delta^j / (m + j + 1).

    Every factor of ``delta`` is now explicit and ``delta = k^2/(1 + k')``
    carries no cancellation, so the moments are accurate at any parameter --
    they simply vanish as ``k -> 0``, which is what they should do.  Neither the
    printed recursion nor the ``v`` form manages that: both build a result of
    order ``delta^(m+1)`` out of terms of order ``delta``, and a far target is
    exactly the small-``k`` case.
    """
    parameter = np.asarray(parameter, dtype=np.float64)
    complementary = np.sqrt(1.0 - parameter)
    delta = parameter / (1.0 + complementary)  # 1 - k', without the cancellation
    moments = []
    for order in range(count):
        total = np.zeros_like(delta)
        binomial = 1.0
        for term in range(order + 1):
            if term:
                binomial = binomial * (order - term + 1) / term
            total = total + (
                (-1.0) ** term
                * binomial
                * 2.0 ** (order - term)
                * delta**term
                / (order + term + 1)
            )
        moments.append(delta ** (order + 1) * total)
    return moments


def pole_moment(characteristic: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    """Return ``I(eta^2)`` over a quarter period, in closed elementary form.

    The paper defines (eq 23)

        I(eta^2) = eta^2 integral_0^u du sn u cn u / (1 - eta^2 sn^2 u)

    and refers the reader elsewhere for its value.  Over the quarter period it
    is elementary.  Substituting ``v = dn u`` -- so ``sn^2 u = (1 - v^2)/k^2``
    and ``k^2 sn u cn u du = -dv``, with ``v`` running from ``1`` down to
    ``k'`` -- the Jacobi functions disappear entirely:

        I(eta^2) = eta^2 integral_(k')^1 dv / (eta^2 v^2 + k^2 - eta^2).

    With ``A = k^2 - eta^2`` that is an arctangent for ``A > 0``, a reciprocal
    for ``A = 0``, and (since ``eta^2 v^2 - |A| > 0`` throughout the range for
    every ``eta^2 < 1``) an ``artanh`` of the RECIPROCAL argument for ``A < 0``.
    ``characteristic`` is ``eta^2`` and ``parameter`` is ``k^2``.
    """
    characteristic = np.asarray(characteristic, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    eta = np.sqrt(characteristic)
    complementary = np.sqrt(1.0 - parameter)
    offset = parameter - characteristic

    safe = np.where(np.abs(offset) < 1e-300, 1.0, offset)
    root = np.sqrt(np.abs(safe))
    positive = (
        eta / root * (np.arctan(eta / root) - np.arctan(eta * complementary / root))
    )
    negative = (
        eta
        / root
        * (
            np.arctanh(np.clip(root / (eta * complementary), -1.0 + 1e-16, 1.0 - 1e-16))
            - np.arctanh(np.clip(root / eta, -1.0 + 1e-16, 1.0 - 1e-16))
        )
    )
    degenerate = 1.0 / complementary - 1.0
    return np.where(
        np.abs(offset) < 1e-300, degenerate, np.where(offset > 0.0, positive, negative)
    )
