"""One moment family, and how the polygon-section reductions contract against it.

Both reductions -- :mod:`nova.biot.polygonanalytic` over the full turn and
:mod:`nova.biot.polygonarc` over a finite arc -- reduce to the same three
questions about a numerator carried as a :mod:`nova.biot.rangefunction` object:
its integral over the range divided by the radical, multiplied by it, and divided
by one of two quadratic denominators as well.  The family of moments those
contractions run against is what differs -- complete or partial in the range, and
weighted by ``sin phi`` or not -- so the family is the object here and the
contractions are its methods.

The denominators are where the difficulty is, and the treatment is common to every
family.  Each is a quadratic with one root just past each end of the range, and
both shifts fall as the square of the section's aspect ratio -- so a root can come
arbitrarily close, and the weight it carries has to be exact in the RELATIVE
sense.  Two things answer that:

* :func:`factorise` splits a denominator between its two FACTORS rather than its
  two roots.  The factors sum to ``1 + p + q``, of order one however close either
  root comes; the difference of the two roots, which a joint partial fraction
  would divide by, falls as the square of the aspect ratio.
* :meth:`Channel.across` takes the numerator's value AT the end a root sits past
  out analytically, because that value is what multiplies the pole's own
  unbounded moment and no series of harmonics delivers it relatively.

Neither depends on what the family is, which is why they are written once here.
What each family supplies is its own two pole SEEDS -- the zeroth moment of each
pole orientation -- and those genuinely differ: the complete range takes them from
:mod:`nova.biot.elliptic`, a partial range's cosine-weighted family needs an
incomplete integral of the third kind, and its sine-weighted family needs one
inverse hyperbolic tangent.
"""

from __future__ import annotations

import numpy as np

from nova.biot.elliptic import harmonic_pole_moments, harmonic_root_moments
from nova.biot.rangefunction import (
    across_the_range,
    contract,
    deflate,
    harmonic_multiply,
)

__all__ = ["POLE_CEILING", "POLE_SWITCH", "Channel", "factorise"]

# Where a denominator's root is far enough past the range for the pole family's own
# moments to be contracted directly, and near enough below it for the weight on the
# pole to have to be taken out exactly.  The two routes overlap comfortably: at this
# shift the direct family decays by 0.38 per order and the deflation the other route
# runs grows by three, so each holds a couple of orders inside round-off.
POLE_SWITCH = 0.25

# Beyond this the factor varies across the range by less than round-off, so its
# root's exact distance stops mattering and holding it here keeps the pole family's
# recursion inside the exponent range.
POLE_CEILING = 1e12


def factorise(denominator: tuple, xp) -> tuple:
    """Return one denominator's two pole shifts and partial-fraction weights.

    A denominator ``L x y + n x + f y`` factors as ``L (y + p)(x + q)`` with
    ``n = L p (1 + q)`` and ``f = L q (1 + p)``, so the shifts follow from the two
    end values and the ``x y`` coefficient without a root-finding step and without
    a subtraction: ``p`` comes from the positive root of
    ``p^2 + p(1 + f/L - n/L) - n/L``, taken by its pivot, and ``q`` from ``f`` over
    ``L (1 + p)``.  Both denominators are strictly positive over the range, so both
    shifts are non-negative and each root lies past its own end.

    A vanishing ``x y`` coefficient -- a vertical edge, whose ``B^2`` is linear --
    leaves one factor, on whichever side its single root falls, and a vanishing end
    difference too -- a target on the axis -- leaves a constant denominator with no
    factor at all.
    """
    bulk, near, far = denominator
    leading = bulk[0] if bulk else 0.0 * near
    curved = leading != 0.0
    held_leading = xp.where(curved, leading, 1.0)
    offset = near / held_leading
    pivot = 1.0 + far / held_leading - offset
    shift_y = xp.where(
        curved,
        2.0 * offset / (pivot + xp.sqrt(pivot * pivot + 4.0 * offset)),
        0.0,
    )
    shift_x = xp.where(curved, far / (held_leading * (1.0 + shift_y)), 0.0)

    rising = (~curved) & (far > near)
    falling = (~curved) & (far < near)
    gap = xp.where(curved, 1.0, xp.where(rising, far - near, near - far))
    held_gap = xp.where(gap != 0.0, gap, 1.0)
    shift_y = xp.where(rising, near / held_gap, shift_y)
    shift_x = xp.where(falling, far / held_gap, shift_x)
    live_y = curved | rising
    live_x = curved | falling

    divisor = xp.where(curved, held_leading * (1.0 + shift_y + shift_x), held_gap)
    return (
        xp.where(live_y, 1.0 / divisor, 0.0),
        xp.where(live_x, 1.0 / divisor, 0.0),
        xp.where(live_y | live_x, 0.0, 1.0 / xp.where(near != 0.0, near, 1.0)),
        xp.where(live_y, shift_y, 1.0),
        xp.where(live_x, shift_x, 1.0),
    )


class Channel:
    """One moment family over the range, with the contractions it answers.

    ``moments`` is ``[M_0, ...]`` with ``M_n`` the integral of ``cos 2na`` over the
    range against whatever weight the family carries, divided by the radical;
    ``cn_seed`` and ``sn_seed`` return that family's zeroth moment against a pole
    factor on ``cos^2 a`` and on ``sin^2 a``.  ``harmonics`` is how deep the
    reduction's numerators reach, which is what the pole family is built to.
    """

    def __init__(self, moments, parameter, *, harmonics, cn_seed, sn_seed, xp=np):
        self.xp = xp
        self.moments = moments
        self.root_moments = harmonic_root_moments(moments, parameter, xp=xp)
        self.harmonics = harmonics
        self._cn_seed = cn_seed
        self._sn_seed = sn_seed

    def poles(self, factors: tuple) -> tuple:
        """Return ``(seed_y, seed_x, family_y, family_x)`` for one factorisation.

        Separate from :func:`factorise` because the shifts belong to the denominator
        and the seeds to the family: a reduction carrying two families splits each
        denominator ONCE and seeds it twice.
        """
        xp = self.xp
        shift_y, shift_x = factors[3], factors[4]
        # a root so far past the range that the factor is constant across it to
        # round-off is held here, where the family's own decay still separates its
        # orders; past that the factor IS constant and the family is P/shift
        capped_y = xp.minimum(shift_y, POLE_CEILING)
        capped_x = xp.minimum(shift_x, POLE_CEILING)
        seed_y = self._cn_seed(capped_y)
        seed_x = self._sn_seed(capped_x)
        return (
            seed_y,
            seed_x,
            self._family(capped_y, seed_y, False),
            self._family(capped_x, seed_x, True),
        )

    def split(self, denominator: tuple) -> tuple:
        """Return ``(factors, poles)`` for a denominator against this family alone."""
        factors = factorise(denominator, self.xp)
        return (factors, self.poles(factors))

    def _family(self, shift, seed, mirrored: bool):
        """Return the pole family, or nothing where no root is far enough out.

        The family is the route for a FAR root only, and a far root is the
        exception: both of the ring denominator's shifts are the edge's own height
        over the ring span, squared, and the plane's near one is the target's
        offset from the edge's line.  Skipping it when no column needs it takes a
        tridiagonal solve of forty orders out of the common build.

        The skip needs the shift's VALUE, and a traced evaluation does not have one
        -- so it is available on the host only, and a trace forms the family for
        every corner and lets :meth:`_pole`'s own selection discard it.  That is the
        one place the two paths differ in cost rather than in arithmetic.
        """
        if self.xp is np and not np.any(shift > POLE_SWITCH):
            return None
        return harmonic_pole_moments(
            shift, seed, self.moments, self.harmonics + 1, mirrored=mirrored
        )

    def plain(self, term: tuple):
        """Return ``integral term/Delta da`` over the range."""
        return contract(across_the_range(term), self.moments)

    def against_root(self, term: tuple):
        """Return ``integral term Delta da`` over the range."""
        return contract(across_the_range(term), self.root_moments)

    def _pole(self, numerator: tuple, shift, seed, family, mirrored: bool):
        """Return ``integral numerator/((v + shift) Delta) da`` past one end.

        Two routes, and which one holds depends on how far past the end the root
        sits.  A NEAR root is the hard case and the reason for the end values: the
        pole's own moment grows without bound as the root reaches the range, so the
        weight on it -- the numerator's value AT that end -- must be exact in the
        relative sense, which no series of harmonics delivers.  Taking it out
        analytically leaves the rest of the numerator reaching the pole only through
        ``x y/(v + shift)``, bounded by one, and a deflation of the bulk whose own
        rounding the shift then multiplies away.

        A FAR root is the easy case and the first route is the wrong one for it:
        the terms it separates grow with the shift while their sum falls, so at a
        shift of a hundred it has thrown away four decades.  There the pole is no
        pole at all -- the factor varies by a fraction of itself across the range --
        and the family's own moments, which need no exactness anywhere, are
        contracted directly.
        """
        bulk, near, far = numerator
        end, other = (far, near) if mirrored else (near, far)
        root = (1.0 if mirrored else -1.0) * (1.0 + 2.0 * shift)
        quotient, value = deflate(bulk, root) if bulk else ([], 0.0)
        held = (
            (end * (1.0 + shift) - other * shift) * seed
            + (other - end) * self.moments[0]
            + contract(
                harmonic_multiply([0.5 + shift, 0.5 if mirrored else -0.5], bulk),
                self.moments,
            )
            - shift
            * (1.0 + shift)
            * (
                value * seed
                + (-2.0 if mirrored else 2.0) * contract(quotient, self.moments)
            )
        )
        if family is None:
            return held
        return self.xp.where(
            shift <= POLE_SWITCH,
            held,
            contract(across_the_range(numerator), family),
        )

    def across(self, numerator: tuple, split: tuple):
        """Return ``integral numerator/(denominator Delta) da``.

        Split between the denominator's two factors, each taken with the range
        variable that vanishes where its own root sits.  The two factors sum to
        ``1 + p + q``, so partial fractions between them divides by a quantity of
        order one however close either root comes to the range -- where a partial
        fraction between the two ROOTS would divide by their difference, which
        falls as the square of the section's aspect ratio.
        """
        (weight_y, weight_x, weight_plain, shift_y, shift_x) = split[0]
        seed_y, seed_x, family_y, family_x = split[1]
        return (
            weight_plain * self.plain(numerator)
            + weight_y * self._pole(numerator, shift_y, seed_y, family_y, False)
            + weight_x * self._pole(numerator, shift_x, seed_x, family_x, True)
        )
