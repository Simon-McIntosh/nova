"""The two ``arsinh`` integrals the polygon-section reductions leave numerical.

Urankar's Part V does the whole angle integral analytically except for two smooth
quadratures per edge, which "evade an analytical treatment as yet".  Both
:mod:`nova.biot.polygonanalytic`, over the full turn's quarter range, and
:mod:`nova.biot.polygonarc`, over a finite arc's partial one, are left with the
same pair, and they are taken here.

Smooth is not the same as easy.  Each integrand is ``arsinh(N/W)`` and its
denominator ``W`` vanishes at a range end whenever the target is level with an
edge end or sits on the edge's extended line -- a grid across a section hits both
by alignment -- so what is left is log-singular exactly where the sections are
evaluated across themselves.  The logarithm is removed ANALYTICALLY rather than
resolved:

    arsinh(N/W) = log(N + sqrt(N^2 + W^2)) - log W

with the first term bounded, so subtracting a model of ``log W`` that matches its
end behaviour leaves a bounded integrand, and the model's own integral is
elementary.  Near either end both denominators go as ``sqrt(w^2 + h^2 b^2)`` in
the offset ``b`` from that end -- ``w`` the target's offset from the edge end's
level, or from the edge's line, and ``h`` the local curvature of the denominator
-- so that is the model, and the SAME two quantities set the panel grading.

The range is halved and each panel stretched by ``b = width sinh(s)`` from its own
end.  That map is EXACT for the model's quadratic --
``w^2 + h^2 width^2 sinh^2 s = w^2 cosh^2 s`` -- so it carries what is left of the
boundary layer after the logarithm has gone.

What the ARC adds is that a panel no longer has to reach the end it is graded
from.  Its two boundary layers still sit at ``a = 0`` and ``a = pi/2``, because
that is where the denominators vanish, but the integration stops at an interior
amplitude and the upper layer may be outside the range entirely.  So each panel
carries its own two limits in the offset from its end, the map is anchored at
``b = 0`` whether or not the panel reaches it, and the model is integrated between
the panel's own bounds rather than over a fixed quarter.  The full turn is the
case where both panels run from zero to a quarter of a turn, and it is unchanged
by the generalisation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

__all__ = ["QUARTER", "graded_residual"]

# One end of the quarter range to the other, which is as far as either panel can
# reach: the two layers sit at the ends of that range whatever the amplitude.
QUARTER = 0.25 * np.pi

# Narrowest boundary layer the graded panels chase before giving up on it.  It is
# a guard rather than a trade-off: the configurations that used to collapse a
# layer onto the range end -- a target level with an edge end, or on an edge's
# extended line, or on a vertex -- are the ones whose logarithm is removed
# analytically, and their width comes out of the OTHER end quantity instead of out
# of this floor.  What is left for it to catch is a layer that is merely narrow.
LAYER_FLOOR = 1e-8


@lru_cache(maxsize=None)
def _rule(nodes: int) -> tuple:
    """Return the Gauss-Legendre rule both graded panels share.

    Fixed for the life of the process, so it is built once rather than once per
    corner and per edge limit.
    """
    return leggauss(nodes // 2)


def _model_integral(offset, scale, lower, upper, xp):
    """Return ``integral_lower^upper log sqrt(offset^2 + scale^2 b^2) db``.

    Elementary, and finite in both degenerate directions: on the axis the model is
    constant and this is the width times its log; at a target level with the edge
    end the model collapses onto ``scale b`` and the arctangent term vanishes with
    the offset.
    """
    held_scale = xp.where(scale > 0.0, scale, 1.0)
    held_offset = xp.where(offset > 0.0, offset, 1.0)

    def primitive(bound):
        # b log b vanishes with b, so an empty lower bound contributes nothing even
        # where the model itself has collapsed onto the origin
        live = bound > 0.0
        held_bound = xp.where(live, bound, 1.0)
        return 0.5 * (
            xp.where(live, bound * xp.log(offset**2 + (scale * held_bound) ** 2), 0.0)
            - 2.0 * bound
            + xp.where(
                scale > 0.0,
                2.0 * offset / held_scale * xp.arctan(held_scale * bound / held_offset),
                2.0 * bound,
            )
        )

    return primitive(upper) - primitive(lower)


def _regularised(numerator, denominator, model, sign, xp):
    """Return ``arsinh(N/W) + sign log(model)``, bounded at the range end.

    The branch follows the sign of the numerator's own value AT that end, which is
    what the logarithm's coefficient is: positive there the ``N + sqrt(N^2 + W^2)``
    form is the stable one, negative there its mirror ``-log(sqrt(N^2 + W^2) - N)``,
    and exactly zero there means the numerator vanishes with the denominator and no
    logarithm survives at all -- the configuration a target ON a section vertex
    produces.

    Whichever branch the END picks, the numerator's sign can turn over INSIDE the
    half -- the azimuthal weight ``b1 X`` sweeps a whole ring span -- and there that
    branch's own sum cancels.  Its value is recovered through ``W^2`` from the other
    one instead, the two being reciprocal about it, and the other is a sum of
    positives exactly where the first is a difference.

    All three cases are then ONE logarithm: the no-logarithm branch is the positive
    one with the model replaced by unity, since ``arsinh`` is itself
    ``log(N + sqrt(N^2 + W^2)) - log W``.
    """
    # the plain root rather than the guarded one: both arguments are of order the
    # ring span here, so nothing overflows and the guard costs several times the
    # square root it protects
    root = xp.sqrt(numerator * numerator + denominator * denominator)
    direct = numerator + root
    mirror = root - numerator
    positive = sign >= 0.0
    pick = xp.where(positive, direct, mirror)
    other = xp.where(positive, mirror, direct)
    base = xp.where(
        positive == (numerator >= 0.0),
        pick,
        denominator * denominator / xp.where(other > 0.0, other, 1.0),
    )
    return xp.where(sign < 0.0, -1.0, 1.0) * xp.log(
        base * xp.where(sign != 0.0, model, 1.0) / denominator
    )


def graded_residual(panels, pieces, nodes: int, xp):
    """Return ``integral arsinh(N/W) da`` over two graded panels, log removed.

    Each entry of ``panels`` is ``(offset, end, scale, lower, upper)`` for one
    panel, in the offset ``b`` from the range end it is graded from: ``offset`` and
    ``end`` are the denominator's and the numerator's own values AT that end, whose
    ratio is what the ``arsinh`` turns over on and so what sets the layer's width;
    ``scale`` is the denominator's local curvature there; and ``lower``/``upper``
    are the panel's own bounds, which for the full turn are zero and a quarter turn
    and for an arc are set by the amplitude.

    ``pieces`` forms the numerator and the denominator from ``x`` and ``y`` and the
    small end offsets, both exact, rather than by evaluating a polynomial near its
    far end.  The FIRST panel is graded from ``a = 0`` and the second from
    ``a = pi/2``.
    """
    node, weight = _rule(nodes)
    total = 0.0
    for panel, (offset, end, scale, lower, upper) in enumerate(panels):
        # The denominator turns over at its own end value over the ring span, and
        # the arsinh saturates where the numerator overtakes it -- never nearer the
        # end than that, so grading on the denominator's scale reaches both.  What
        # is new here is what happens when that scale VANISHES: with the logarithm
        # gone the model is then exact to the order that matters, nothing is left at
        # the denominator's own scale, and the remaining feature is the numerator's.
        # A target on a section vertex sends both to zero and needs no grading at
        # all -- which is the whole gain, because a floor set low enough for that
        # case is what used to thin the nodes everywhere else.
        reach = xp.where(offset > 0.0, offset, xp.abs(end))
        width = xp.where(reach > 0.0, xp.clip(reach / scale, LAYER_FLOOR, 1.0), 1.0)
        held = width[:, None]
        start = xp.arcsinh(lower / width)[:, None]
        span = xp.arcsinh(upper / width)[:, None] - start
        stretch = start + 0.5 * span * (node + 1.0)[None, :]
        stretched = xp.sinh(stretch)
        panel_offset = held * stretched
        # the panel never reaches a quarter turn, so the complement is a subtraction
        # rather than a second transcendental, and the map's own jacobian follows
        # from the sinh it has already taken
        near = xp.sin(panel_offset) ** 2
        x, y = (1.0 - near, near) if panel else (near, 1.0 - near)
        numerator, denominator = pieces(x, y)
        sign = xp.sign(end)[:, None]
        scaled = scale[:, None] * panel_offset
        model = xp.sqrt(offset[:, None] ** 2 + scaled * scaled)
        jacobian = 0.5 * span * held * xp.sqrt(1.0 + stretched * stretched)
        bounded = _regularised(numerator, denominator, model, sign, xp)
        total = (
            total
            + (jacobian * bounded) @ weight
            - xp.sign(end) * _model_integral(offset, scale, lower, upper, xp)
        )
    return total
