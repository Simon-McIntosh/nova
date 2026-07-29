"""Where a finite arc's two ends put the reduction's integration limit.

The polygon-section reduction integrates a transformed angle ``a`` from zero to
an amplitude, and for a full turn that amplitude is a quarter turn at both ends
of the range -- which is why :mod:`nova.biot.polygonanalytic` never forms one.
For an ARC it is set by where the target sits relative to each end,

    alpha_i = (pi + phi - phi_i)/2                          (Urankar Part V, eq 12)

with ``phi`` the target's azimuth and ``phi_i`` the arc's.  The antiderivative is
only defined for ``0 < a <= pi/2``, so each ``alpha_i`` has to be folded back into
that quarter before it can be evaluated, and the folding is where the signs live.

The paper folds it in three printed cases (eqs 25 and 26, p. 1178): case A takes
``|alpha_i| <= pi/2`` as it stands, case B takes ``pi/2 < |alpha_i| <= 3 pi/2``
through ``theta_i = pi - |alpha_i|`` and picks up ``2 X(pi/2)``, and case C takes
``3 pi/2 < alpha_i <= 2 pi`` through ``theta_i = 2 pi - alpha_i`` and picks up
``4 X(pi/2)``.  **The three are one formula**, and using it rather than the cases
removes the branch entirely -- which is what makes this evaluable inside a traced
tile, where a case selection on a per-target value cannot be.

The formula follows from what the integrands are.  Every one of them is a
function of ``2a``, so all are HALF-TURN PERIODIC in the amplitude, and each is
either even or odd about zero:

* the rows whose integrand is odd -- the potential's radial row and the field's
  azimuthal one -- integrate to an EVEN function of the amplitude, and their
  integral over a half turn vanishes because the integrand is odd about a quarter
  turn as well.  So those rows are half-turn periodic outright.
* the rows whose integrand is even -- the potential's azimuthal row and the
  field's radial and vertical ones -- integrate to an ODD function, and their
  integral over a half turn is twice the quarter-turn value.

Writing ``alpha = n pi + delta`` with ``delta`` in the closed quarter
``(-pi/2, pi/2]`` and ``n = round(alpha/pi)``, both statements read

    X_even(alpha) = X(|delta|)
    X_odd(alpha)  = 2 n X(pi/2) + sign(delta) X(|delta|)

and each of the paper's three cases is a value of ``n``: zero, one and two.  The
full turn is the check that it is right -- there ``alpha_2 = alpha_1 - pi``, so
the two ends differ by exactly one turn count, the folded amplitudes coincide and
the parities oppose, and everything cancels but ``-2 X(pi/2)``: the ring the
shipped reduction evaluates, recovered from the arc's own bookkeeping.

Exactness matters here for the same reason it does in
:mod:`nova.biot.incompleteelliptic`, and the half-angle supplies it.  With
``psi = phi - phi_i`` the amplitude's own pair is ``sin alpha = cos(psi/2)`` and
``cos alpha = -sin(psi/2)``, so a target ON an arc end gives a cosine of exactly
zero rather than 6e-17 -- and that end is not a corner case but the one the full
turn is built from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ArcLimit", "arc_limits", "fold"]


@dataclass
class ArcLimit:
    """One end of the arc, folded into the quarter the antiderivative covers.

    ``amplitude`` is ``|delta|`` and ``sine``/``cosine`` its own pair, formed from
    the half-separation so both stay exact where the target meets the end.
    ``turns`` is the half-turn count the fold picked up, ``parity`` the sign the
    odd rows carry, and ``weight`` the ``-(-1)^(i+1)`` the assembly puts on this
    end -- so an arc's value is the plain sum of its two ends.
    """

    amplitude: np.ndarray
    sine: np.ndarray
    cosine: np.ndarray
    turns: np.ndarray
    parity: np.ndarray
    weight: float


def _limit(separation, weight, xp):
    """Return one end's fold, everything formed from the HALF separation.

    ``alpha = (pi + psi)/2`` is never assembled: its sine and cosine come from
    ``psi/2`` directly, and the half-turn count from ``psi/(2 pi)`` -- so the only
    place a right angle appears is the amplitude itself, which is reported for
    readability and not used by the fold.
    """
    half = 0.5 * separation
    # sin alpha = cos(psi/2) and cos alpha = -sin(psi/2), both exact
    sine, cosine = xp.cos(half), -xp.sin(half)
    # n = round(alpha/pi) = round(1/2 + psi/(2 pi)); at a half-integer either
    # choice folds to the same value, the two differing by a whole X(pi/2) that
    # the parity flip takes back out
    turns = xp.round(0.5 + separation / (2.0 * np.pi))
    alternating = 1.0 - 2.0 * xp.abs(xp.remainder(turns, 2.0))
    # delta = alpha - n pi, then |delta|: its cosine is the folded cosine, which
    # the count makes non-negative, and its sine is the folded sine's magnitude
    folded_sine = alternating * sine
    return ArcLimit(
        amplitude=xp.arctan2(xp.abs(folded_sine), alternating * cosine),
        sine=xp.abs(folded_sine),
        cosine=alternating * cosine,
        turns=turns,
        parity=xp.where(folded_sine < 0.0, -1.0, 1.0),
        weight=weight,
    )


def arc_limits(azimuth, start, end, *, xp=np):
    """Return the two :class:`ArcLimit` folds for an arc from ``start`` to ``end``.

    ``azimuth`` is the target's, ``start`` and ``end`` the arc's, all in radians
    and all broadcasting together.  The two ends carry opposite assembly weights,
    which is the ``-(-1)^(i+1)`` of the paper's eq 13 and eq 17.

    Nothing here reduces the azimuth into a canonical turn, and nothing needs to:
    adding a whole turn to the target shifts BOTH amplitudes by a half turn, so
    both odd rows gain the same ``2 X(pi/2)`` and the difference between the ends
    -- which is the whole answer -- does not move.  The periodicity is a property
    of the fold rather than a normalisation imposed on the caller.
    """
    azimuth = xp.asarray(azimuth)
    return (
        _limit(azimuth - xp.asarray(start), -1.0, xp),
        _limit(azimuth - xp.asarray(end), 1.0, xp),
    )


def fold(limit: ArcLimit, folded, quarter=None):
    """Return one end's contribution, weighted and ready to sum.

    ``folded`` is the antiderivative evaluated at :attr:`ArcLimit.amplitude`.
    ``quarter`` is its value at a quarter turn and is what distinguishes the two
    row classes: an ODD row supplies it and picks up ``2 n`` of it, an even row
    leaves it out and is half-turn periodic without it.
    """
    if quarter is None:
        return limit.weight * folded
    return limit.weight * (2.0 * limit.turns * quarter + limit.parity * folded)
