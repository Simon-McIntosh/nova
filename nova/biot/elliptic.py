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

The three moment families below are what the final expressions (eqs 21-24) are
built from.  They are collected here, separately testable against direct
quadrature, because a mis-transcribed recursion is invisible inside the full
assembly but obvious against a reference integral.
"""

from __future__ import annotations

import numpy as np
import scipy.special  # type: ignore[import-untyped]

__all__ = [
    "complete_pi",
    "pole_moment",
    "sn_moments",
    "sn_cn_moments",
]


def complete_pi(characteristic: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    """Return the complete elliptic integral of the third kind, Pi(n | m).

    Carlson symmetric forms, as the rectangular-section kernel already uses:
    ``Pi(n | m) = R_F(0, 1 - m, 1) + (n/3) R_J(0, 1 - m, 1, 1 - n)``.  Both
    arguments are the PARAMETER convention (``m = k^2``), matching
    :func:`scipy.special.ellipk`.
    """
    characteristic = np.asarray(characteristic, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64)
    zero = np.zeros_like(characteristic)
    one = np.ones_like(characteristic)
    rf = scipy.special.elliprf(zero, 1.0 - parameter, one)
    rj = scipy.special.elliprj(zero, 1.0 - parameter, one, 1.0 - characteristic)
    return rf + rj * characteristic / 3.0


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
