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
boundary layer after the logarithm has gone. Below the denominator layer's
resolution floor, the remainder turns over where its square root resolves both
end quantities: its width is ``hypot(offset, end) / scale``. Following only
the denominator offset overgrades a smooth remainder when the numerator's end
value is larger. The width follows this joint scale without a lower clip;
when both end quantities vanish the regularised integrand needs no grading.
The model integral carries its end-layer derivative analytically. The grading
width is held fixed while differentiating the remainder, with moving integration
bounds retaining their map and Jacobian derivatives.

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

from nova.biot.rangefunction import (
    _array_program,
    _product_tangent,
    _quotient_tangent,
)

from nova.biot.pairedfloat import add as paired_add
from nova.biot.pairedfloat import multiply as paired_multiply
from nova.biot.pairedfloat import scale as paired_scale
from nova.biot.pairedfloat import subtract as paired_subtract
from nova.biot.pairedfloat import wrap as paired_wrap

__all__ = ["QUARTER", "graded_residual"]

# One end of the quarter range to the other, which is as far as either panel can
# reach: the two layers sit at the ends of that range whatever the amplitude.
QUARTER = 0.25 * np.pi
LAYER_FLOOR = 1e-8


@lru_cache(maxsize=None)
def _rule(nodes: int) -> tuple:
    """Return the Gauss-Legendre rule both graded panels share.

    Fixed for the life of the process, so it is built once rather than once per
    corner and per edge limit.
    """
    return leggauss(nodes // 2)


@_array_program
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


@_array_program
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


def _fixed_nodes(value, xp):
    """Keep a quadrature-map value fixed while differentiating its integrand."""
    if xp is np:
        return value
    from jax import lax

    return lax.stop_gradient(value)


def _grading_width(offset, end, scale, xp):
    """Resolve the remainder's joint end scale below the denominator floor."""
    denominator_reach = xp.where(offset > 0.0, offset, xp.abs(end))
    reach = xp.where(
        denominator_reach < LAYER_FLOOR * scale,
        xp.hypot(offset, end),
        denominator_reach,
    )
    return _fixed_nodes(
        xp.where(reach > 0.0, xp.clip(reach / scale, 0.0, 1.0), 1.0), xp
    )


def graded_residual(panels, pieces, nodes: int, xp, *, paired: bool = False):
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
    total = None if paired else 0.0
    for panel, (offset, end, scale, lower, upper) in enumerate(panels):
        # The logarithmic model resolves the denominator's singular end.
        # Its bounded remainder follows the joint numerator/denominator scale
        # when that end is too narrow for the denominator grading.
        width = _grading_width(offset, end, scale, xp)
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
        model_integral = xp.sign(end) * _model_integral(offset, scale, lower, upper, xp)
        if paired:
            quadrature = paired_wrap(xp.zeros_like(offset))
            for index, one_weight in enumerate(weight):
                quadrature = paired_add(
                    quadrature,
                    paired_scale(
                        paired_multiply(
                            paired_wrap(jacobian[:, index]),
                            paired_wrap(bounded[:, index]),
                        ),
                        one_weight,
                    ),
                )
            panel_value = paired_subtract(quadrature, paired_wrap(model_integral))
            total = panel_value if total is None else paired_add(total, panel_value)
        else:
            total = total + (jacobian * bounded) @ weight - model_integral
    return total


def _held_tangent(condition, value, d_value, fill, xp):
    """Return ``where(condition, value, fill)`` and its tangent."""
    return xp.where(condition, value, fill), xp.where(condition, d_value, 0.0)


def _model_integral_tangent(
    offset, d_offset, scale, d_scale, lower, d_lower, upper, d_upper, xp
):
    """Return :func:`_model_integral` and its tangent in closed form."""
    held_scale, d_held_scale = _held_tangent(scale > 0.0, scale, d_scale, 1.0, xp)
    held_offset, d_held_offset = _held_tangent(offset > 0.0, offset, d_offset, 1.0, xp)

    def primitive(bound, d_bound):
        live = bound > 0.0
        held_bound, d_held_bound = _held_tangent(live, bound, d_bound, 1.0, xp)
        stretched = scale * held_bound
        d_stretched = _product_tangent(scale, d_scale, held_bound, d_held_bound)
        square = offset**2 + stretched**2
        d_square = d_offset * (2.0 * offset) + d_stretched * (2.0 * stretched)
        logarithm = xp.log(square)
        d_logarithm = d_square / square
        coefficient = 2.0 * offset / held_scale
        d_coefficient = _quotient_tangent(
            2.0 * offset, 2.0 * d_offset, held_scale, d_held_scale
        )
        reach = held_scale * bound
        d_reach = _product_tangent(held_scale, d_held_scale, bound, d_bound)
        argument = reach / held_offset
        d_argument = _quotient_tangent(reach, d_reach, held_offset, d_held_offset)
        angle = xp.arctan(argument)
        d_angle = d_argument / (1.0 + argument * argument)
        value = 0.5 * (
            xp.where(live, bound * logarithm, 0.0)
            - 2.0 * bound
            + xp.where(scale > 0.0, coefficient * angle, 2.0 * bound)
        )
        d_value = 0.5 * (
            xp.where(
                live, _product_tangent(bound, d_bound, logarithm, d_logarithm), 0.0
            )
            - 2.0 * d_bound
            + xp.where(
                scale > 0.0,
                _product_tangent(coefficient, d_coefficient, angle, d_angle),
                2.0 * d_bound,
            )
        )
        return value, d_value

    (high, d_high), (low, d_low) = primitive(upper, d_upper), primitive(lower, d_lower)
    return high - low, d_high - d_low


def _regularised_tangent(
    numerator, d_numerator, denominator, d_denominator, model, d_model, sign, xp
):
    """Return :func:`_regularised` and its tangent, branch for branch."""
    square = denominator * denominator
    d_square = _product_tangent(denominator, d_denominator, denominator, d_denominator)
    radical = numerator * numerator + square
    d_radical = (
        _product_tangent(numerator, d_numerator, numerator, d_numerator) + d_square
    )
    root = xp.sqrt(radical)
    d_root = d_radical * (0.5 / root)
    direct, d_direct = numerator + root, d_numerator + d_root
    mirror, d_mirror = root - numerator, d_root - d_numerator
    positive = sign >= 0.0
    pick = xp.where(positive, direct, mirror)
    d_pick = xp.where(positive, d_direct, d_mirror)
    other = xp.where(positive, mirror, direct)
    d_other = xp.where(positive, d_mirror, d_direct)
    held_other, d_held_other = _held_tangent(other > 0.0, other, d_other, 1.0, xp)
    same = positive == (numerator >= 0.0)
    base = xp.where(same, pick, square / held_other)
    d_base = xp.where(
        same, d_pick, _quotient_tangent(square, d_square, held_other, d_held_other)
    )
    factor, d_factor = _held_tangent(sign != 0.0, model, d_model, 1.0, xp)
    product = base * factor
    d_product = _product_tangent(base, d_base, factor, d_factor)
    argument = product / denominator
    d_argument = _quotient_tangent(product, d_product, denominator, d_denominator)
    orientation = xp.where(sign < 0.0, -1.0, 1.0)
    return (
        orientation * xp.log(argument),
        orientation * (d_argument / argument),
    )


def _graded_residual_tangent(panels, d_panels, pieces_tangent, nodes: int, xp):
    """Return :func:`graded_residual` and its tangent over the same nodes.

    ``d_panels`` carries one tangent per panel quantity, and ``pieces_tangent``
    maps ``(x, d_x, y, d_y)`` to the numerator and denominator with their
    tangents. The grading width is fixed in the tangent; integration-bound
    motion still contributes through the nodes and their Jacobian.
    """
    node, weight = _rule(nodes)
    total = d_total = 0.0
    for panel, (values, tangents) in enumerate(zip(panels, d_panels, strict=True)):
        offset, end, scale, lower, upper = values
        d_offset, _, d_scale, d_lower, d_upper = tangents
        width = _grading_width(offset, end, scale, xp)
        held = width[:, None]
        start = xp.arcsinh(lower / width)[:, None]
        span = xp.arcsinh(upper / width)[:, None] - start
        d_start = ((d_lower / width) / xp.sqrt(1.0 + (lower / width) ** 2))[:, None]
        d_high = ((d_upper / width) / xp.sqrt(1.0 + (upper / width) ** 2))[:, None]
        d_span = d_high - d_start
        half_span = 0.5 * span
        stretch = start + half_span * (node + 1.0)[None, :]
        d_stretch = d_start + 0.5 * d_span * (node + 1.0)[None, :]
        stretched = xp.sinh(stretch)
        d_stretched = d_stretch * xp.cosh(stretch)
        panel_offset = held * stretched
        d_panel_offset = held * d_stretched
        sine = xp.sin(panel_offset)
        near = sine**2
        d_near = (d_panel_offset * xp.cos(panel_offset)) * (2.0 * sine)
        if panel:
            x, d_x, y, d_y = 1.0 - near, -d_near, near, d_near
        else:
            x, d_x, y, d_y = near, d_near, 1.0 - near, -d_near
        (numerator, denominator), (d_numerator, d_denominator) = pieces_tangent(
            x, d_x, y, d_y
        )
        sign = xp.sign(end)[:, None]
        scaled = scale[:, None] * panel_offset
        d_scaled = _product_tangent(
            scale[:, None], d_scale[:, None], panel_offset, d_panel_offset
        )
        model = xp.sqrt(offset[:, None] ** 2 + scaled * scaled)
        d_model = (
            d_offset[:, None] * (2.0 * offset[:, None])
            + _product_tangent(scaled, d_scaled, scaled, d_scaled)
        ) * (0.5 / model)
        cosine = xp.sqrt(1.0 + stretched * stretched)
        d_cosine = (2.0 * stretched * d_stretched) * (0.5 / cosine)
        jacobian = half_span * held * cosine
        d_jacobian = _product_tangent(
            half_span * held, 0.5 * d_span * held, cosine, d_cosine
        )
        bounded, d_bounded = _regularised_tangent(
            numerator, d_numerator, denominator, d_denominator, model, d_model,
            sign, xp,
        )  # fmt: skip
        model_integral, d_model_integral = _model_integral_tangent(
            offset, d_offset, scale, d_scale, lower, d_lower, upper, d_upper, xp
        )
        orientation = xp.sign(end)
        total = total + (jacobian * bounded) @ weight - orientation * model_integral
        d_total = (
            d_total
            + _product_tangent(jacobian, d_jacobian, bounded, d_bounded) @ weight
            - orientation * d_model_integral
        )
    return total, d_total
