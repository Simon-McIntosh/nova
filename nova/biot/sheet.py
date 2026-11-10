"""Poloidal flux of a conic toroidal current sheet.

A magnetised cell's edge carries a thin sheet: a straight segment in the poloidal
plane swept through a full turn, bearing a uniform azimuthal surface current
``K``.  This module evaluates the poloidal flux ``psi`` threading a circle at
``(r, z)`` from the closed form of L. K. Urankar, "Current-carrying finite arc
segment, Part II", IEEE Trans. Magn. MAG-18(3), 911-917, 1982.

The finite-field domain is positive distance from the source's closed segment.
The same domain applies on and off the symmetry axis. Source endpoints and
interior source points carry no finite-field contract.

Route.  The sheet's flux is ``2 pi r A_phi``, and the azimuthal vector potential
is the segment integral of the ring kernel.  Part II reduces that integral by
parts to complete elliptic integrals of all three kinds plus one smooth integral
of an ``arsinh``, which the paper's eq (12b) leaves to quadrature.  The elliptic
kinds come from :func:`nova.biot.completeelliptic.complete_kind` (first and
second) and :func:`nova.biot.completeelliptic.complete_pole` (third), each taking
the exact modulus complement ``k'^2`` rather than forming ``1 - k^2`` from a
parameter, so a target close to the sheet keeps its digits.

The expressions of Part II are antiderivatives in the source coordinate, not
definite values: the finite segment enters as the difference between the two
ends.  For a sloped segment, parameterised ``r' = b_1 z' + c_1``, the
antiderivative of the azimuthal integral in ``z'`` is, with ``gamma = z' - z``,
``r_1 = c_1 + b_1 z`` and ``a_1^2 = 1 + b_1^2``,

    a_1^3 f(phi) = a_1 b_1 D + (r_1 + b_1^2 r cos phi) arsinh(beta_1)

with ``D = sqrt(gamma^2 + r^2 + r'^2 - 2 r r' cos phi)``, ``beta_1 = Gamma_1/B_1``,
``Gamma_1 = a_1^2 gamma + b_1 (r_1 - r cos phi)`` and the exact squared
denominator ``B_1^2 = r_1^2 - 2 r r_1 cos phi + r^2 (1 + b_1^2 sin^2 phi)``.
This reproduces the paper's eq (8) and reduces to eq (24) at ``b_1 = 0``.

The complete elliptic kinds enter through the closed form of the ``D`` part.  The
``arsinh`` part is log-singular at the azimuth where the denominator ``B_1``
vanishes, which is a target level with the segment end, and a grid across the
section hits it by alignment.  It is therefore taken on the graded two-panel
rule of :mod:`nova.biot.polygonanalytic`, which subtracts a model of the
logarithm in closed form, grades each panel with a sinh map from its own end and
integrates the bounded remainder, with the polynomial weight
``cos(phi) (r_1 + b_1^2 r cos(phi))`` the sheet's own integrand needs.
The ``D`` term is smooth but has a square-root layer of width ``D_0`` at
``phi = 0``, so it is integrated on a sinh map anchored there, which is exact for
``D = sqrt(D_0^2 + 4 r r_1 sin^2(phi/2))``.  The straight-cylinder limit
``b_1 = 0`` is eq (24) and is evaluated in closed form.

That graded rule is not the only route.  :meth:`SheetFlux.antiderivative_closed`
carries the same antiderivative with the ``arsinh`` block reduced by parts: the
oscillatory part of the weight has an antiderivative vanishing at both ends, so
the derivative of ``arsinh(beta_1)`` is what remains, and it is rational over
``B_1^2 D``.  Splitting that bracket onto ``B_1^2``'s pole pair leaves complete
integrals of the first, second and third kinds and one smooth ``arsinh``
integral, the remainder the paper itself leaves to quadrature.  The two routes
share no arithmetic in the ``R`` block, so they check each other there;
``route="closed"`` is the default and ``route="quadrature"`` selects the graded
one.

The flat sheet's own closed form is the paper's eqs (26) and (27), transcribed in
:meth:`SheetFlux.pancake_antiderivative` and :meth:`SheetFlux._pancake_field`.
They are the ``b_2 = 0`` case of the same antiderivative, carrying complete
integrals of all three kinds plus the paper's one remaining ``arsinh`` integral
``g_0^(2)(pi/2)``.  That integral is log-singular at the azimuth where the target
is level with the sheet's plane, which is where eight of the section's twelve
fixed targets sit, so it is evaluated by :func:`nova.biot.zeta.zeta`, this
repository's owner of that integral, which carries the tanh-sinh rule its
near-plane case needs.  The flat sheet's surface-current integral
(:func:`nova.biot.greens.greens_psi` over the segment) is kept beside it as the
independent check, taken by ``route="quadrature"``.

The block feeding eq (22a) (``S``, ``M``, ``N`` and ``P_phi`` of eqs 13a and 18a)
could not be reconstructed from the available copy of the paper: assembled as
read it disagrees with the verified azimuthal quadrature by a factor, not by
rounding.  It is therefore not used; the sloped case is reduced as above.
"""

from __future__ import annotations

import functools
from functools import lru_cache

import numpy as np

import jax
import jax.numpy as jnp

from nova.biot.completeelliptic import (
    complete_kind,
    complete_kind_traced,
    complete_pole,
)
from nova.biot.gradedresidual import QUARTER, graded_residual
from nova.biot.greens import (
    MU0,
    _filament_elliptic_combinations,
    greens_bz_br,
    greens_psi,
)
from nova.biot.zeta import traced_zeta, zeta
from nova.jax.config import configure_dtypes

__all__ = [
    "ROUTES",
    "SheetFlux",
    "packed_sheet_end_scale",
    "packed_sheet_field",
    "sheet_field",
    "sheet_flux",
]

# The closed form is built from jitted programs, and a compiled program cannot
# carry single precision past the geometry it was traced on.  Enabling x64 once,
# here, is what `complete_kind_traced` is then entitled to assume.
configure_dtypes()
assert jax.config.jax_enable_x64, "the closed form needs double arithmetic"

ROUTES = ("closed", "quadrature")
"""The two evaluation routes :meth:`SheetFlux.flux` and :meth:`SheetFlux.field` offer.

``"closed"`` is the production route at every slope: Part II eqs (22a)/(23a)
for a sloped segment, Snow's eqs (23)/(24)/(26)/(27) form for a vertical or flat
one.  ``"quadrature"`` is the independent cross-check, the same antiderivative
taken by Gauss-Legendre quadrature, and it shares no arithmetic with the closed
route in the block it is compared on.
"""


def _checked_route(route: str) -> str:
    """Return ``route`` if it names an evaluation family, refusing anything else."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {route!r}")
    return route


_PANELS = 8
"""Sub-intervals of the segment rule, placed symmetrically."""

_NODES = 16
"""Gauss-Legendre nodes per panel; the rule is fixed, never adapted."""


@lru_cache(maxsize=None)
def _gauss_rule():
    """The panelised rule's nodes and weights, built once on first use."""
    return np.polynomial.legendre.leggauss(_NODES)


def _divide(numerator, denominator):
    """Quotient and remainder of two polynomials in ascending coefficients.

    The bracketed weight ``N(cos phi)`` is quartic and the sheet's squared
    denominator is quadratic in ``cos phi``, so splitting the integrand onto the
    denominator leaves a quadratic quotient and a linear remainder.  ``polydiv``
    trims the remainder's high coefficients, so it is padded back to one
    coefficient per divisor degree: the reductions index the remainder by degree,
    and a zero linear term is still a linear term.
    """
    divisor = np.asarray(denominator, dtype=np.float64)
    quotient, remainder = np.polynomial.polynomial.polydiv(
        np.asarray(numerator, dtype=np.float64), divisor
    )
    rest = np.zeros(divisor.size - 1)
    rest[: remainder.size] = remainder
    return quotient, rest


_ARSINH_NODES = 128
"""Nodes of the graded arsinh rule, the Part V module's own count."""


@lru_cache(maxsize=None)
def _graded_nodes():
    """The sinh map's nodes; it is exact for the layer the square root carries."""
    return np.polynomial.legendre.leggauss(64)


_FIELD_PANELS = 140
_FIELD_RATIO = 1.25
_FIELD_NODES = 24
"""The azimuthal rule of the field kernels, graded geometrically toward the
singular azimuth.  A target level with a segment end puts ``B_1`` at zero on
``phi = 0``, where the field's integrands carry a logarithmic end; geometric
grading in the sinh variable resolves it, and the count is the one measured to
hold the section's floor."""


@lru_cache(maxsize=None)
def _field_rule():
    """The field kernels' azimuthal nodes and weights, built once on first use."""
    return np.polynomial.legendre.leggauss(_FIELD_NODES)


_FLAT_PANELS = 64
_FLAT_NODES = 32
"""The flat annulus's radial rule; its kernel varies on the standoff scale."""


def _root_expression(height, radius, source):
    """``R``: the closed form's complete first- and second-kind ``D`` block.

    ``2a[2(1 - k^2)K + (k^2 - 2)E] / (3k^2)`` of eq (22a), with the modulus
    ``k^2 = 4 r r' / a^2`` and its complement both formed from the geometry
    rather than by subtraction.
    """
    distance2 = height**2 + radius**2 + source**2
    q = 2.0 * radius * source / distance2
    use_series = q <= 0.25
    power = jnp.where(use_series, q, 0.0)
    coefficient = 1.0
    moment = 1.0
    series = 0.0
    for order in range(1, 33):
        coefficient *= (1.5 - order) / order
        if order % 2:
            moment *= order / (order + 1)
            series += -coefficient * moment * power**order
    series *= np.pi * jnp.sqrt(distance2)
    height = jnp.where(use_series, 1.0, height)
    radius = jnp.where(use_series, 1.0, radius)
    source = jnp.where(use_series, 0.5, source)
    span = height * height + (radius + source) ** 2
    k2 = 4.0 * radius * source / span
    complement = (height * height + (radius - source) ** 2) / span
    first, second = complete_kind_traced(complement)
    regular = (
        2.0 * span**0.5 * (2.0 * (1.0 - k2) * first + (k2 - 2.0) * second) / (3.0 * k2)
    )
    return jnp.where(use_series, series, regular)


_closed_root = jax.jit(_root_expression)


_closed_root_gradient = jax.jit(jax.value_and_grad(_closed_root, argnums=(0, 1)))


def _zero_source_terms(xp, gamma, radius, slope):
    """Primitive and target derivatives at a source end of zero radius.

    At zero modulus the root integral vanishes. Fourier integration of the
    remaining atanh gives the interior primitive below; outside |b gamma|=r
    it is linear in r and independent of gamma.
    """
    distance = xp.hypot(gamma, radius)
    square = slope**2
    inside = xp.abs(slope * gamma) <= radius
    magnitude = xp.where(slope != 0.0, xp.abs(slope), 1.0)
    exterior = (
        np.pi
        / 2
        * xp.sign(gamma)
        * (xp.sqrt(1 + square) + square * xp.arcsinh(1 / magnitude))
    )
    inner = np.pi * square / 2
    angle = xp.arcsinh(gamma / radius)
    value = xp.where(
        inside, inner * (gamma * distance / radius + radius * angle), radius * exterior
    )
    height = xp.where(inside, 2 * inner * distance / radius, 0.0)
    radial = xp.where(inside, inner * (angle - gamma * distance / radius**2), exterior)
    return value, height, radial


def _axis_field_ends(xp, target_z, start_r, start_z, end_r, end_z):
    """Elementary on-axis B_Z antiderivatives in arclength, per sheet end."""
    length = xp.hypot(end_r - start_r, end_z - start_z)
    u = (end_r - start_r) / length
    v = (end_z - start_z) / length
    cross = v * start_r - u * (start_z - target_z)
    line = cross == 0.0
    held_cross = xp.where(line, 1.0, cross)
    ends = []
    for source, gamma in ((end_r, end_z - target_z), (start_r, start_z - target_z)):
        t = u * source + v * gamma
        held_t = xp.where(line, 1.0, t)
        distance = xp.hypot(held_t, held_cross)
        regular = (
            u**2 * xp.arcsinh(held_t / xp.abs(held_cross))
            + ((v**2 - u**2) * held_t - 2 * u * v * held_cross) / distance
        )
        line_t = xp.where(line, t, 1.0)
        collinear = u**2 * xp.sign(line_t) * xp.log(xp.abs(line_t))
        ends.append(MU0 / 2 * xp.where(line, collinear, regular))
    return ends


def _axis_expansion(radius, target_z, start_r, start_z, end_r, end_z):
    """Fields and flux from the axis primitive's convergent radial expansion."""
    from jax.experimental.jet import jet

    def ends(z):
        return jnp.stack(_axis_field_ends(jnp, z, start_r, start_z, end_r, end_z))

    tangent = (jnp.ones_like(target_z),) + (jnp.zeros_like(target_z),) * 4
    value, derivatives = jet(ends, (target_z,), (tangent,))
    first, second, third, fourth, fifth = derivatives
    height = value - radius**2 / 4 * second + radius**4 / 64 * fourth
    radial = -radius / 2 * first + radius**3 / 16 * third - radius**5 / 384 * fifth
    flux = (
        np.pi * radius**2 * (value - radius**2 / 8 * second + radius**4 / 192 * fourth)
    )
    return radial, height, flux


_axis_expansion = jax.jit(_axis_expansion)


def _near_axis(xp, radius, target_z, start_r, start_z, end_r, end_z):
    """Select a uniformly convergent radial series away from its line limit."""
    dr, dz = end_r - start_r, end_z - start_z
    length2 = dr**2 + dz**2
    foot = -(start_r * dr + (start_z - target_z) * dz) / length2
    foot = xp.minimum(1.0, xp.maximum(0.0, foot))
    distance = xp.hypot(start_r + foot * dr, start_z + foot * dz - target_z)
    cross = (dz * start_r - dr * (start_z - target_z)) / xp.sqrt(length2)
    return (
        (radius > 0) & (radius <= 0.01 * distance) & (xp.abs(cross) > 0.01 * distance)
    )


def _cylinder_radial(xp, gamma, radius, source):
    """The cylinder's radial endpoint is the stable ring flux over 2 pi r."""
    at_zero = source == 0.0
    source = xp.where(at_zero, 0.5 * radius, source)
    span = gamma**2 + (radius + source) ** 2
    complement = (gamma**2 + (radius - source) ** 2) / span
    first, second = complete_kind(complement, xp=xp)
    _, flux, _ = _filament_elliptic_combinations(
        xp, 4 * radius * source / span, complement, first, second
    )
    value = MU0 * xp.sqrt(span) * flux / (2 * np.pi * radius)
    return xp.where(at_zero, 0.0, value)


def _source_potential_series(xp, gamma, radius, source, slope, *, exterior_only=True):
    """Source multipoles of psi/(mu0 pi r²) and its target derivatives.

    The addition theorem integrates cos(phi) P_n onto
    pi sin(theta) sin(theta_s) P_n'(t) P_n'(v)/(n(n+1)).
    Integrating arclength adds the factor 1/(n+2). Each term retains its
    source-radius power before the two endpoint primitives are differenced.
    """
    held_slope = xp.where(slope != 0.0, slope, 1.0)
    a = xp.sqrt(1 + slope**2)
    intercept = gamma - source / held_slope
    distance = xp.hypot(radius, intercept)
    ratio = source * a / (xp.abs(held_slope) * distance)
    eligible = (
        (slope != 0.0)
        & (
            (not exterior_only)
            | (xp.abs(source - slope * gamma) > radius * (1 + 1e-12))
        )
        & (ratio <= 0.5)
    )
    if xp is np and not np.any(eligible):
        zero = np.zeros_like(distance)
        return eligible, intercept, zero, zero, zero
    ratio = xp.where(eligible, ratio, 0.0)
    t = -intercept / distance
    v = xp.sign(held_slope) / a
    zero, one = xp.zeros_like(distance), xp.ones_like(distance)

    def step(index, state):
        (
            prev,
            current,
            dprev,
            derivative,
            vprev,
            vcurrent,
            power,
            value,
            height,
            radial,
        ) = state
        n = index + 1
        common = power * vcurrent / (n * (n + 1) * (n + 2))
        value = value + common * current
        height = height + common * (
            -derivative * radius**2 / distance**3
            - (n + 2) * current * intercept / distance**2
        )
        radial = radial + common * (
            derivative * intercept * radius / distance**3
            - (n + 2) * current * radius / distance**2
        )
        following = ((2 * n + 1) * t * current - (n + 1) * prev) / n
        dfollowing = ((2 * n + 1) * (current + t * derivative) - (n + 1) * dprev) / n
        vfollowing = ((2 * n + 1) * v * vcurrent - (n + 1) * vprev) / n
        return (
            current,
            following,
            derivative,
            dfollowing,
            vcurrent,
            vfollowing,
            power * ratio,
            value,
            height,
            radial,
        )

    *_, value, height, radial = _sheet_series_loop(
        xp, 80, step, (zero, one, zero, zero, zero, one, ratio**3, zero, zero, zero)
    )
    factor = xp.sign(held_slope) * slope**2 / a**2
    return eligible, intercept, factor * value, factor * height, factor * radial


def _source_series_ends(xp, radius, target_z, start_r, start_z, end_r, end_z):
    """Difference physical source multipoles before adding a primitive gauge.

    Both ends must lie inside the convergence sphere about the source line's
    axis intercept. The common zero-source primitive then cancels identically.
    Holding inactive geometry before the series keeps its derivative defined.
    """
    dz = end_z - start_z
    slope = (end_r - start_r) / xp.where(dz != 0, dz, 1.0)
    held_slope = xp.where(slope != 0, slope, 1.0)
    intercept = start_z - target_z - start_r / held_slope
    distance = xp.hypot(radius, intercept)
    eligible = (
        (dz != 0)
        & (slope != 0)
        & (
            xp.maximum(start_r, end_r) * xp.sqrt(1 + slope**2)
            <= 0.5 * xp.abs(held_slope) * distance
        )
    )
    r = xp.where(eligible, radius, 1.0)
    b = xp.where(eligible, slope, 1.0)
    fields, fluxes = [], []
    for source, height in ((end_r, end_z), (start_r, start_z)):
        source = xp.where(eligible, source, 0.0)
        gamma = xp.where(eligible, height - target_z, 1.0)
        _, _, value, vertical, radial = _source_potential_series(
            xp, gamma, r, source, b, exterior_only=False
        )
        orientation = xp.where(dz > 0, 1.0, -1.0)
        fields.append(
            (
                orientation * MU0 / 2 * r * vertical,
                orientation * MU0 / 2 * (2 * value + r * radial),
            )
        )
        fluxes.append(orientation * MU0 * np.pi * r**2 * value)
    return eligible, fields, fluxes


def _source_gradient_series(xp, gamma, radius, source, slope, regular):
    eligible, intercept, value, height, radial = _source_potential_series(
        xp, gamma, radius, source, slope
    )
    base = _zero_source_terms(xp, intercept, radius, slope)
    factor = np.pi * (1 + slope**2)
    terms = (
        base[0] + factor * radius * value,
        base[1] + factor * radius * height,
        base[2] + factor * (value + radius * radial),
    )
    return tuple(xp.where(eligible, a, b) for a, b in zip(terms, regular))


def _sheet_series_loop(xp, count, body, initial):
    """Carry a fixed analytic series without unrolling its traced program."""
    if xp is np:
        state = initial
        for index in range(count):
            state = body(index, state)
        return state
    return jax.lax.fori_loop(0, count, body, initial)


@functools.lru_cache(maxsize=None)
def _pancake_series_constants():
    """Full-turn even cosine moments and even binomial coefficients."""
    moment = [np.pi]
    for order in range(1, 65):
        moment.append(moment[-1] * (2 * order - 1) / (2 * order))
    indices = np.arange(32)
    moments = np.asarray(moment)[indices[:, None] + indices[None, :] + 1]
    coefficient = [1.0]
    for order in range(1, 64):
        coefficient.append(coefficient[-1] * (2 * order - 1) / (2 * order))
    return moments, np.asarray(coefficient)[2 * indices]


def _pancake_end_series(xp, gamma, radius, source, regular, regular_height):
    """Factor small endpoint fields before the complete poles and zeta cancel.

    The source multipoles converge in source/distance or its inverse; the
    intermediate source range uses the combined pole integrand's near-axis
    series. The small-source branches factor out source cubed; the outer
    branch uses a rationalised infinite-radius limit.
    """
    gamma, radius, source = map(xp.asarray, (gamma, radius, source))
    distance = xp.hypot(gamma, radius)
    t2 = (gamma / distance) ** 2
    inner = source <= 0.8 * distance
    outer = distance <= 0.8 * source
    vin = xp.where(inner, source / distance, 0.0)
    vout = xp.where(outer, distance / xp.where(source > 0.0, source, 1.0), 0.0)
    zero = xp.zeros_like(distance)
    one = xp.ones_like(distance)

    def step(index, state):
        even, odd, coefficient, pin, pout, inside, outside, inside_height = state
        term = coefficient * odd / ((2 * index + 2) * (2 * index + 3))
        following = -(index + 1.5) / (index + 1)
        inside = inside + pin * term
        inside_height = inside_height + pin * coefficient * (t2 * odd - even) / (
            (2 * index + 2) * (2 * index + 3)
        )
        outside = outside + pout * following * term
        even = ((4 * index + 5) * t2 * odd - (2 * index + 3) * even) / (2 * index + 2)
        odd = ((4 * index + 7) * even - (2 * index + 4) * odd) / (2 * index + 3)
        return (
            even,
            odd,
            coefficient * following,
            pin * vin**2,
            pout * vout**2,
            inside,
            outside,
            inside_height,
        )

    *_, inside, outside, inside_height = _sheet_series_loop(
        xp, 96, step, (one, 3.0 * one, one, one, vout**2, zero, zero, zero)
    )
    common = np.pi * (radius / distance) * (gamma / distance)
    inside = -common * vin**3 * inside
    outside = (
        -np.pi * xp.sign(gamma) * radius / (distance + xp.abs(gamma)) - common * outside
    )

    near_axis = (radius / distance) ** 2 <= 0.25
    use_axis = near_axis & ~inner & ~outer
    h = xp.where(use_axis, (radius / distance) ** 2, 0.0)
    span = distance**2 + source**2
    q = xp.where(use_axis, 2.0 * radius * source / span, 0.0)
    moments, coefficients = _pancake_series_constants()
    order = xp.arange(32)
    integrals = (h[..., None] ** order) @ xp.asarray(moments)
    weights = (
        xp.asarray(coefficients)
        * q[..., None] ** (2 * xp.maximum(order - 1, 0))
        * (4.0 * radius**2 / span**2)[..., None]
        * (
            (2.0 * order / (2 * order + 1)) * distance[..., None] ** 2
            - source[..., None] ** 2
        )
    )
    weights = xp.where(order == 0, -1.0, weights)
    axis = (
        gamma
        * radius
        * source**3
        / (distance**2 * span**1.5)
        * xp.sum(weights * integrals, axis=-1)
    )
    radial = xp.where(
        inner, inside, xp.where(outer, outside, xp.where(use_axis, axis, regular))
    )
    height = xp.where(inner, np.pi * vin**3 * inside_height, regular_height)
    return radial, height


def _cylinder_height_series(xp, gamma, radius, source):
    """Exterior cylinder height endpoint with its source-radius square factored.

    The disk-potential multipoles give the endpoint as
    ``-pi*t*v**2 sum (2k+1)/(2k+2) P_2k(0) [P_(2k+1)(t)/t] v**(2k)``.
    Carrying the odd polynomial divided by ``t`` keeps the sheet plane regular.
    Here ``t=gamma/distance`` and ``v=source/distance``.
    """
    distance = xp.hypot(gamma, radius)
    eligible = (radius > source) & (source <= 0.8 * distance)
    t = gamma / distance
    v = xp.where(eligible, source / distance, 0.0)
    one = xp.ones_like(distance)
    zero = xp.zeros_like(distance)

    def step(index, state):
        even, odd, moment, power, value = state
        value = value + power * moment * (2 * index + 1) / (2 * index + 2) * odd
        even = ((4 * index + 3) * t**2 * odd - (2 * index + 1) * even) / (2 * index + 2)
        odd = ((4 * index + 5) * even - (2 * index + 2) * odd) / (2 * index + 3)
        moment = -moment * (2 * index + 1) / (2 * index + 2)
        return even, odd, moment, power * v**2, value

    *_, value = _sheet_series_loop(xp, 96, step, (one, one, one, one, zero))
    return eligible, -np.pi * t * v**2 * value


def _sheet_cylinder_height(xp, gamma, radius, source):
    """Cylinder height endpoint through complete kinds or exterior multipoles.

    Interior complete kinds have the same sign. For a small source outside it,
    the source-radius series combines their opposing leading terms analytically.
    """
    gamma, radius, source = map(xp.asarray, (gamma, radius, source))
    span = gamma**2 + (radius + source) ** 2
    complement = (gamma**2 + (radius - source) ** 2) / span
    pole = ((source - radius) / (source + radius)) ** 2
    first, _ = complete_kind(complement, xp=xp)
    third = complete_pole(pole, complement, xp=xp)
    regular = (
        gamma / xp.sqrt(span) * (first + (source - radius) / (source + radius) * third)
    )
    eligible, series = _cylinder_height_series(xp, gamma, radius, source)
    return xp.where(eligible, series, regular)


class SheetFlux:
    """Poloidal flux of one sheet in the closed form of Part II."""

    def __init__(self, start, end, *, sheet_current: float = 1.0):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.sheet_current = float(sheet_current)
        if self.start.shape != (2,) or self.end.shape != (2,):
            raise ValueError("segment endpoints must be (r, z) pairs")
        if self.start[1] > self.end[1]:
            self.start, self.end = self.end, self.start
        span = self.end - self.start
        self._height = float(span[1])
        self._slope = float(span[0] / span[1]) if span[1] != 0.0 else np.inf
        self._flat = span[1] == 0.0

    def _rule(self, lower: float, upper: float):
        """Panelised Gauss-Legendre nodes and weights on ``[lower, upper]``."""
        nodes, weights = _gauss_rule()
        edges = np.linspace(lower, upper, _PANELS + 1)
        half = 0.5 * np.diff(edges)
        centre = 0.5 * (edges[1:] + edges[:-1])
        x = (centre[:, None] + half[:, None] * nodes[None, :]).ravel()
        w = (half[:, None] * weights[None, :]).ravel()
        return x, w

    def _root_term(self, gamma: float, r: float, rp: float) -> float:
        """``a_1 b_1`` times the azimuthal integral of ``cos(phi) D``.

        ``D^2 = (r - r_p)^2 + gamma^2 + 4 r r_p sin^2(phi/2)``, so the map
        ``phi = width sinh(s)`` from ``phi = 0`` is exact for the square-root
        layer there and leaves a smooth remainder.  The two squares are formed
        from positive terms rather than by subtracting nearly equal radii, which
        is what keeps the layer's own digits at a near standoff.
        """
        if rp == 0.0:
            return 0.0
        width = np.hypot(gamma, r - rp) / np.sqrt(r * rp)
        width = width if width > 0.0 else 1.0
        node, weight = _graded_nodes()
        span = np.arcsinh(np.pi / width)
        stretch = 0.5 * span * (node + 1.0)
        phi = width * np.sinh(stretch)
        jacobian = 0.5 * span * width * np.cosh(stretch)
        dense = np.sqrt(
            (r - rp) ** 2 + gamma**2 + 4.0 * r * rp * np.sin(0.5 * phi) ** 2
        )
        return float(np.sum(jacobian * weight * np.cos(phi) * dense))

    def _arsinh_term(self, gamma: float, r: float, r1: float, b1: float) -> float:
        """The azimuthal integral of ``cos(phi) (r_1 + b_1^2 r cos(phi)) arsinh``.

        The azimuth halves at ``a = phi/2 = pi/4``, and each half is graded from
        its own end: ``a = phi/2`` runs over the first half, and the offset
        ``pi/2 - a`` over the second.  ``c = cos^2 a - sin^2 a`` is ``-cos(phi)``,
        in which the weight is a quadratic and the denominator is quadratic in
        ``c`` and in the panel's own ``sin^2``.
        """
        panels, pieces = self._arsinh_rule(gamma, r, r1, b1)

        # cos(phi) = -c and d phi = 2 da, so the weight is -2 r_1 c + 2 b_1^2 r c^2;
        # the endpoint coefficients are its values at c = -1 and c = +1
        weight = (
            [-8.0 * b1 * b1 * r],
            np.full(1, 2.0 * (r1 + b1 * b1 * r)),
            np.full(1, 2.0 * (b1 * b1 * r - r1)),
        )
        value = graded_residual(panels, pieces, _ARSINH_NODES, np, weight=weight)
        return float(np.asarray(value).sum())

    def _arsinh_rule(self, gamma: float, r: float, r1: float, b1: float):
        """The two graded panels and the integrand closures of the arsinh terms.

        The azimuth halves at ``a = phi/2 = pi/4``, and each half is graded from
        its own end: ``a = phi/2`` runs over the first half, and the offset
        ``pi/2 - a`` over the second.  ``c = cos^2 a - sin^2 a`` is ``-cos(phi)``,
        and both the numerator and the denominator are quadratic in it.
        """
        a1sq = 1.0 + b1 * b1
        spread = r1 - r
        # The denominator's end value and curvature at each graded end, read off
        # the exact squares (r_1 + r)^2 - 4 r (r_1 - r b_1^2) b^2 on the phi = pi
        # panel and (r_1 - r)^2 + 4 r (r_1 + r b_1^2) b^2 on the phi = 0 one.  A
        # target beyond the segment's ends or on the far side of its line puts
        # r_1 below -r, where the phi = 0 curvature is negative and r_1 + r is
        # negative; both enter the log model as magnitudes, the model being the
        # square root of a squared term, so they are taken unsigned rather than
        # left to an imaginary square root.
        upper = 2.0 * np.sqrt(r * np.abs(r1 - r * b1 * b1))
        lower = 2.0 * np.sqrt(r * np.abs(r1 + r * b1 * b1))
        far = a1sq * gamma + b1 * (r1 + r)
        near = a1sq * gamma + b1 * spread
        one = np.ones(1)
        panels = (
            (one * abs(r1 + r), one * far, one * upper, 0.0, QUARTER),
            (one * abs(spread), one * near, one * lower, 0.0, QUARTER),
        )

        def pieces(x, y):
            return (
                near + 2.0 * b1 * r * y,
                np.sqrt(spread**2 + 4.0 * r * r1 * y + 4.0 * r * r * b1 * b1 * x * y),
            )

        return panels, pieces

    def _residual_arsinh(self, gamma: float, r: float, r1: float, b1: float) -> float:
        """``int_0^pi arsinh(beta_1) d phi``: the paper's eq (12b) remainder.

        This is the one piece of the arsinh block with no elementary form.  It is
        smooth away from a target level with the segment end, where it is
        logarithmically integrable, so it runs on the same graded rule.  The
        factor of two is ``d phi = 2 da``.
        """
        panels, pieces = self._arsinh_rule(gamma, r, r1, b1)
        value = graded_residual(panels, pieces, _ARSINH_NODES, np)
        return 2.0 * float(np.asarray(value).sum())

    def antiderivative(self, gamma: float, r: float, rp: float) -> float:
        """The sloped-segment antiderivative, integrated over the azimuth.

        ``gamma = z' - z`` is the source's height above the target and ``rp`` its
        radius.  The paper's function is the antiderivative in the source
        coordinate, so the segment enters as the difference between the two ends.
        ``rp`` must lie on the segment's supporting line, ``rp = b_1 z' + c_1``.
        """
        if rp == 0.0:
            return float(
                MU0
                * r
                / (1 + self._slope**2)
                * _zero_source_terms(np, gamma, r, self._slope)[0]
            )
        b1 = self._slope
        a1sq = 1.0 + b1 * b1
        r1 = rp - b1 * gamma
        total = np.sqrt(a1sq) * b1 * self._root_term(gamma, r, rp)
        total += self._arsinh_term(gamma, r, r1, b1)
        return float(MU0 * r / a1sq * total)

    def _arc_remainder(self, gamma, r, r1, b1, a, k2, K, E, complement):
        """``int_0^pi N(cos phi) / (B_1^2 D) d phi``: the pole reduction's remainder.

        Integrating the arsinh term by parts moves the derivative onto
        ``arsinh(beta_1)``, and that derivative is rational:

            d arsinh(beta_1)/d phi
                = a_1 r sin(phi)
                  [b_1 r (r - r_1 cos phi) - gamma (r_1 + b_1^2 r cos phi)]
                  / (B_1^2 D).

        The oscillatory part of the weight has an antiderivative that vanishes at
        both ends of the half turn, so no boundary term survives and the whole
        remainder is the bracket ``N`` above over ``B_1^2 D``.

        ``N`` is quartic in ``cos phi`` and ``B_1^2`` is quadratic, so the
        quotient integrates onto ``K`` and ``E`` through the Jacobi moments.  The
        linear remainder sits over ``B_1^2``, which is a quadratic in
        ``y = sin^2 a = sn^2 u``; its two roots are the paper's own characteristic
        magnitudes, and each ``1/(y - y_p)`` is a complete third-kind integral of
        characteristic ``n_p = 1/y_p``.
        """
        a1 = np.sqrt(1.0 + b1 * b1)
        bracket = np.array([r1, 0.5 * b1 * b1 * r, -r1, -0.5 * b1 * b1 * r])
        derivative = np.array(
            [b1 * r * r - gamma * r1, -b1 * b1 * r * gamma - b1 * r * r1]
        )
        weight = a1 * r * np.convolve(bracket, derivative)
        squared = np.array(
            [r1 * r1 + (1.0 + b1 * b1) * r * r, -2.0 * r * r1, -b1 * b1 * r * r]
        )
        # A target level with the segment end puts a root of B_1^2 on phi = 0,
        # where the bracket carries the same (1 - cos phi).  Cancelling it first
        # keeps the split a difference of finite integrals: left in, the two
        # quarter-turn integrals diverge together and lose every digit bought.
        gap = r1 - r
        if abs(gap) <= 1e-12 * r:
            weight, _ = _divide(weight, np.array([1.0, -1.0]))
            squared, _ = _divide(squared, np.array([1.0, -1.0]))
        quotient, remainder = _divide(weight, squared)

        polynomial = np.zeros(3)
        polynomial[: quotient.size] = quotient
        moment = K
        square = (K - E) / k2
        quartic = ((2.0 + k2) * K - 2.0 * (1.0 + k2) * E) / (3.0 * k2 * k2)
        # cos(phi) = 2 sn^2 u - 1, so cos^2 = 4 sn^4 - 4 sn^2 + 1
        folded = (
            polynomial[2] * (4.0 * quartic - 4.0 * square + moment)
            + polynomial[1] * (2.0 * square - moment)
            + polynomial[0] * moment
        )

        if squared.size == 3:
            leading = squared[2]
            # roots in cos(phi), carried to y = sin^2 u = (1 + cos(phi))/2
            centre_roots = -squared[1] / leading
            product_roots = squared[0] / leading
            discriminant = np.sqrt(0.25 * centre_roots * centre_roots - product_roots)
            roots = np.array(
                [
                    0.5 * (1.0 + 0.5 * centre_roots - discriminant),
                    0.5 * (1.0 + 0.5 * centre_roots + discriminant),
                ]
            )
            offset = remainder[0] - remainder[1]
            slope = 2.0 * remainder[1]
            # cos(phi) = 2 y - 1, so the y-quadratic's leading coefficient is 4 x this
            weights = (offset + slope * roots) / (4.0 * leading) / (roots[0] - roots[1])
            weights[1] = (
                (offset + slope * roots[1]) / (4.0 * leading) / (roots[1] - roots[0])
            )
            third = complete_pole(1.0 - 1.0 / roots, np.full(2, complement))
            third_kind = float(np.sum(-(weights / roots) * third))
        else:
            # one factor cancelled, so the denominator is linear in cos(phi) and
            # leaves a single characteristic
            pole = (1.0 + (-squared[0] / squared[1])) / 2.0
            third = complete_pole(
                np.asarray([1.0 - 1.0 / pole]), np.asarray([complement])
            )
            third_kind = -remainder[0] / (2.0 * squared[1] * pole) * third[0]
        return float(2.0 / a * (folded + third_kind))

    def _arsinh_term_closed(self, gamma, r, r1, b1, a, k2, K, E, complement):
        """The arsinh part of the antiderivative in closed form.

        The part-integration leaves the paper's eq (12b) remainder multiplied by
        the weight's mean and the rational pole reduction of
        :meth:`_arc_remainder`.
        """
        mean = 0.5 * b1 * b1 * r * self._residual_arsinh(gamma, r, r1, b1)
        return mean - self._arc_remainder(gamma, r, r1, b1, a, k2, K, E, complement)

    def antiderivative_closed(self, gamma: float, r: float, rp: float) -> float:
        """The sloped-segment antiderivative in closed form.

        The ``D`` part is the complete first- and second-kind combination
        ``2a[2(1 - k^2)K + (k^2 - 2)E]/(3k^2)``; the ``arsinh`` part is its
        by-parts reduction, which leaves one smooth ``arsinh`` quadrature and
        complete third-kind integrals at the two pole seeds.  A straight segment
        (``b_1 = 0``) is eq (24) and must take the earlier route instead.
        """
        if rp == 0.0:
            return float(
                MU0
                * r
                / (1 + self._slope**2)
                * _zero_source_terms(np, gamma, r, self._slope)[0]
            )
        eligible, intercept, value, _, _ = _source_potential_series(
            np, gamma, r, rp, self._slope
        )
        if eligible:
            base = _zero_source_terms(np, intercept, r, self._slope)[0]
            return float(
                MU0 * r / (1 + self._slope**2) * base + MU0 * np.pi * r**2 * value
            )
        b1 = self._slope
        a1sq = 1.0 + b1 * b1
        r1 = rp - b1 * gamma
        span = gamma * gamma + (r + rp) ** 2
        a = np.sqrt(span)
        k2 = 4.0 * r * rp / span
        complement = (gamma * gamma + (r - rp) ** 2) / span
        first, second = complete_kind(np.asarray([complement]))
        K = float(first[0])
        E = float(second[0])
        root = float(_closed_root(gamma, r, rp))
        total = np.sqrt(a1sq) * b1 * root
        total += self._arsinh_term_closed(gamma, r, r1, b1, a, k2, K, E, complement)
        return float(MU0 * r / a1sq * total)

    def _azimuth_rule(self, width: float):
        """Nodes and weights on ``[0, pi]``, graded toward the singular end.

        The map ``phi = width sinh(s)`` reproduces the square-root layer of the
        ring distance and turns the logarithm at ``phi = 0`` into a logarithm of
        the grading variable, which geometric panels in ``s`` resolve.
        """
        node, weight = _field_rule()
        span = np.arcsinh(np.pi / width)
        index = np.arange(_FIELD_PANELS + 1)
        edges = span * (_FIELD_RATIO**index - 1.0) / (_FIELD_RATIO**_FIELD_PANELS - 1.0)
        half = 0.5 * np.diff(edges)
        centre = 0.5 * (edges[1:] + edges[:-1])
        stretch = (centre[:, None] + half[:, None] * node[None, :]).ravel()
        rule = (half[:, None] * weight[None, :]).ravel() * width * np.cosh(stretch)
        return width * np.sinh(stretch), rule

    def _gradients(
        self, gamma: float, r: float, rp: float
    ) -> tuple[float, float, float]:
        """``(S, dS/dgamma, dS/dr)`` of the antiderivative's azimuthal integral.

        ``S = a_1 b_1 R + A`` is the bracket :meth:`antiderivative` carries, so
        the field is the total derivative of the same closed form rather than a
        numerical difference of its values.  The ``z`` derivative is taken in
        ``gamma = z' - z``, the radial one at fixed source; ``R``'s partials are
        the derivative kernels on the shared graded rule, and ``A``'s come from
        :meth:`_arsinh_gradient`.
        """
        if rp == 0.0:
            return tuple(
                float(x) for x in _zero_source_terms(np, gamma, r, self._slope)
            )
        b1 = self._slope
        a1 = np.sqrt(1.0 + b1 * b1)
        width = np.hypot(gamma, r - rp) / np.sqrt(r * rp)
        width = width if width > 0.0 else 1.0
        phi, rule = self._azimuth_rule(width)
        cosine = np.cos(phi)
        dense = np.sqrt(
            (r - rp) ** 2 + gamma**2 + 4.0 * r * rp * np.sin(0.5 * phi) ** 2
        )
        root = self._root_term(gamma, r, rp)
        root_gamma = float(np.sum(rule * cosine * gamma / dense))
        root_radial = float(np.sum(rule * cosine * (r - rp * cosine) / dense))
        arsinh_term, arsinh_gamma, arsinh_radial = self._arsinh_gradient(gamma, r, rp)
        value = a1 * b1 * root + arsinh_term
        gamma_value = a1 * b1 * root_gamma + arsinh_gamma
        radial_value = a1 * b1 * root_radial + arsinh_radial
        return float(value), gamma_value, radial_value

    def _arsinh_gradient(
        self, gamma: float, r: float, rp: float
    ) -> tuple[float, float, float]:
        """``(A, dA/dgamma, dA/dr)``: the arsinh remainder and its partials.

        ``A`` is the one piece of the antiderivative the paper leaves to
        quadrature.  Its partials are the pole coefficient of
        ``d(arsinh beta_1)/dx`` contracted onto the same graded rule, and
        ``d(r_1)/dgamma = -b_1`` enters the height component WITHOUT the field
        weight, so the weighted flux integral cannot stand in for it.
        """
        if rp == 0.0:
            return tuple(
                float(x) for x in _zero_source_terms(np, gamma, r, self._slope)
            )
        b1 = self._slope
        a1sq = 1.0 + b1 * b1
        r1 = rp - b1 * gamma
        a1 = np.sqrt(a1sq)
        phi, rule = _gradient_azimuth_rule(
            np, *[np.asarray([value]) for value in (gamma, r, rp, b1)]
        )
        phi, rule = phi.ravel(), rule.ravel()
        cosine = np.cos(phi)
        sine_sq = np.sin(phi) ** 2
        dense = np.sqrt(
            (r - rp) ** 2 + gamma**2 + 4.0 * r * rp * np.sin(0.5 * phi) ** 2
        )
        # ``W`` vanishes with the standoff at the singular azimuth, so it is
        # formed from positive terms: written as the difference of two O(r)
        # values it loses every digit the grading buys there.
        half_sine = np.sin(0.5 * phi) ** 2
        # a target on the segment's supporting line stands off by rounding
        # alone; taken at face value that standoff is what the graded nodes
        # below it divide by, so the geometry is snapped to the line it is on
        gap = r1 - r
        if abs(gap) <= 1e-12 * r:
            gap = 0.0
        W = gap + 2.0 * r * half_sine
        denominator = W * W + a1sq * r * r * sine_sq
        Gamma = a1sq * gamma + b1 * W
        arsinh = np.arcsinh(Gamma / np.sqrt(denominator))
        smear = r1 + b1 * b1 * r * cosine
        # the pole coefficient of d(arsinh beta_1)/dx, contracted so the target
        # level with the segment end does not divide zero by zero; ``E_1`` is
        # ``a_1 D``, so this over the shared azimuth scale is exact
        gamma_pole = a1sq * (W * (rp - r * cosine) + r * r * sine_sq) / denominator
        radial_pole = (
            -b1 * cosine * denominator
            - Gamma * (-gap + 2.0 * r1 * half_sine + r * b1 * b1 * sine_sq)
        ) / denominator
        arsinh_plain = float(np.sum(rule * cosine * arsinh))
        gamma_value = -b1 * arsinh_plain + float(
            np.sum(rule * cosine * smear * gamma_pole / (a1 * dense))
        )
        radial_value = b1 * b1 * float(np.sum(rule * cosine * cosine * arsinh)) + float(
            np.sum(rule * cosine * smear * radial_pole / (a1 * dense))
        )
        return self._arsinh_term(gamma, r, r1, b1), gamma_value, radial_value

    def _sloped_field(self, r: float, r_z: float) -> tuple[float, float]:
        """``(B_R, B_Z)`` from the gradient of the flux, exact in the closed form."""
        a1sq = 1.0 + self._slope**2
        # the antiderivative carries ``MU0 r / a_1^2`` and the field is
        # ``(1 / 2 pi r)`` times the gradient of the flux, so the radius cancels
        # in the height component and survives in the radial one
        b_r = b_z = 0.0
        ends = (
            (float(self.end[1]) - r_z, float(self.end[0]), 1.0),
            (float(self.start[1]) - r_z, float(self.start[0]), -1.0),
        )
        for height, radius, sign in ends:
            value, gamma_value, radial_value = self._gradients(height, r, radius)
            b_r += sign * gamma_value
            b_z += sign * (value + r * radial_value)
        return (
            float(MU0 / (2.0 * np.pi * a1sq) * b_r),
            float(MU0 / (2.0 * np.pi * r * a1sq) * b_z),
        )

    def _closed_gradient(self, gamma: float, r: float, rp: float):
        """``(S, dS/dgamma, dS/dr)`` with ``R`` closed and ``A`` shared.

        ``R`` is the complete first- and second-kind block, and its partials are
        the forward-mode tangents of those same expressions through
        :mod:`nova.biot.completeelliptic`'s own descent.  ``A``, the arsinh
        remainder the paper leaves to quadrature, is taken whole from
        :meth:`_arsinh_gradient`: one kernel, evaluated the same way on both
        routes, so the two fields cannot disagree about a piece neither
        computes in closed form.
        """
        if rp == 0.0:
            return tuple(
                float(x) for x in _zero_source_terms(np, gamma, r, self._slope)
            )
        b1 = self._slope
        a1 = np.sqrt(1.0 + b1 * b1)
        root, gradient = _closed_root_gradient(
            jnp.asarray(gamma), jnp.asarray(r), jnp.asarray(rp)
        )
        arsinh_term, arsinh_gamma, arsinh_radial = self._arsinh_gradient(gamma, r, rp)
        regular = (
            float(a1 * b1) * float(root) + arsinh_term,
            float(a1 * b1) * float(gradient[0]) + arsinh_gamma,
            float(a1 * b1) * float(gradient[1]) + arsinh_radial,
        )
        return _source_gradient_series(np, gamma, r, rp, b1, regular)

    def _sloped_field_closed(self, r: float, r_z: float) -> tuple[float, float]:
        """``(B_R, B_Z)`` from the closed form's own partial derivatives."""
        a1sq = 1.0 + self._slope**2
        b_r = b_z = 0.0
        ends = (
            (float(self.end[1]) - r_z, float(self.end[0]), 1.0),
            (float(self.start[1]) - r_z, float(self.start[0]), -1.0),
        )
        for height, radius, sign in ends:
            value, gamma_value, radial_value = self._closed_gradient(height, r, radius)
            if self._slope == 0.0:
                gamma_value = float(
                    2 * np.pi / MU0 * _cylinder_radial(np, height, r, radius)
                )
            b_r += sign * gamma_value
            if self._slope == 0.0:
                b_z += sign * r * float(_sheet_cylinder_height(np, height, r, radius))
            else:
                b_z += sign * (value + r * radial_value)
        return (
            float(MU0 / (2.0 * np.pi * a1sq) * b_r),
            float(MU0 / (2.0 * np.pi * r * a1sq) * b_z),
        )

    def _pancake_terms(
        self, gamma: float, r: float, rp: float
    ) -> tuple[float, float, float]:
        """``(A_phi, H_r, H_z)`` of eqs (26) and (27) at one segment end.

        ``A_phi`` is the antiderivative of the flux, ``H_r`` and ``H_z`` the
        paper's field kernels, whose difference between the two ends times
        ``mu0 / 2 pi`` is the field.  A target level with the sheet's plane is the
        one place the printed blocks need their limit taken rather than their
        arithmetic: the third-kind sum carries ``1/(r - c)`` against a ``Pi`` that
        falls off like ``r - c``, so the radial kernel's block cancels to zero and
        the potential's third-kind term vanishes with ``gamma^2``.
        """
        if rp == 0.0:
            return 0.0, 0.0, 0.0
        span = gamma**2 + (r + rp) ** 2
        a = np.sqrt(span)
        complement = (gamma**2 + (r - rp) ** 2) / span
        first, second = complete_kind(np.asarray([complement]))
        big_k = float(np.ravel(first)[0])
        big_e = float(np.ravel(second)[0])
        c = np.sqrt(gamma**2 + r**2)
        # ``g_0^(2)(pi/2)`` is ``zeta`` with the arc half-angle folded to
        # pi/2, so the flat sheet calls the repository's owner of this
        # integral rather than a second transcription of its rule.
        remainder = float(
            np.ravel(zeta(np.asarray(rp), np.asarray(r), np.asarray(gamma), np.pi / 2))[
                0
            ]
        )
        potential = (
            -rp
            / (2.0 * a * r)
            * ((r * r - rp * rp - 3.0 * gamma**2) * big_k + span * big_e)
        )
        potential += r * remainder
        height = 2.0 * remainder - 2.0 * rp / a * big_k
        if abs(gamma) <= 1e-12 * r:
            return potential, 0.0, height
        third = 0.0
        radial = 0.0
        for order in (1, 2):
            if order == 1:
                ratio = (rp + c) / (r - c)
                characteristic = 2.0 * r / (r - c)
            else:
                ratio = (rp - c) / (r + c)
                characteristic = 2.0 * r / (r + c)
            pole = float(
                np.ravel(
                    complete_pole(
                        np.asarray([1.0 - characteristic]), np.asarray([complement])
                    )
                )[0]
            )
            third += (-1) ** order * ratio * pole
            radial += (-1) ** order * c * ratio * pole
        potential -= c * gamma**2 / (2.0 * a * r) * third
        regular = gamma / (a * r) * (2.0 * rp * big_k - radial)
        radial, height = _pancake_end_series(np, gamma, r, rp, regular, height)
        return potential, float(radial), float(height)

    def pancake_antiderivative(self, gamma: float, r: float, rp: float) -> float:
        """Eq (26): the flat sheet's antiderivative in the source radius ``r'``.

        ``gamma = z' - z`` is the target's depth below the sheet's plane and
        ``rp`` the source radius at this end.  The flux is ``mu0 r`` times the
        difference between the two ends.
        """
        return self._pancake_terms(gamma, r, rp)[0]

    def _pancake_ends(self) -> tuple[float, float]:
        """The segment's two source radii, lower first.

        A flat sheet's current runs azimuthally, so the antiderivative is
        differenced in the source radius and the direction the endpoints were
        given in carries no sign.
        """
        return tuple(sorted((float(self.start[0]), float(self.end[0]))))

    def _pancake_flux(self, r: float, z: float) -> float:
        """Eq (26) over the segment: ``mu0 r`` times the two ends' difference."""
        gamma = float(self.start[1]) - z
        lower, upper = self._pancake_ends()
        return (
            MU0
            * r
            * (
                self.pancake_antiderivative(gamma, r, upper)
                - self.pancake_antiderivative(gamma, r, lower)
            )
        )

    def _pancake_field(self, r: float, z: float) -> tuple[float, float]:
        """``(B_R, B_Z)`` of eq (27) at ``(r, z)``."""
        gamma = float(self.start[1]) - z
        lower, upper = self._pancake_ends()
        ends = ((upper, 1.0), (lower, -1.0))
        radial = height = 0.0
        for radius, sign in ends:
            _, end_radial, end_height = self._pancake_terms(gamma, r, radius)
            radial += sign * end_radial
            height += sign * end_height
        return (
            float(MU0 / (2.0 * np.pi) * radial),
            float(MU0 / (2.0 * np.pi) * height),
        )

    def _flat_field(self, r: float, z: float) -> tuple[float, float]:
        """The surface-current integral of the ring field over a flat annulus."""
        z0 = float(self.start[1])
        lower, upper = sorted(float(v) for v in (self.start[0], self.end[0]))
        node, weight = np.polynomial.legendre.leggauss(_FLAT_NODES)
        edges = np.linspace(lower, upper, _FLAT_PANELS + 1)
        half = 0.5 * np.diff(edges)
        centre = 0.5 * (edges[1:] + edges[:-1])
        rp = (centre[:, None] + half[:, None] * node[None, :]).ravel()
        w = (half[:, None] * weight[None, :]).ravel()
        bz, br = greens_bz_br(np.full_like(rp, r), np.full_like(rp, z), rp, z0)
        return float(np.sum(w * br)), float(np.sum(w * bz))

    def field(
        self, r: float, z: float, *, route: str = "closed"
    ) -> tuple[float, float]:
        """``(B_R, B_Z)`` [T] at ``(r, z)``, per the sheet current set at build.

        ``route`` names the evaluation family and means the same thing at every
        slope.  ``"closed"`` is the production route: eq (27) on a flat sheet,
        the closed-form antiderivative's own partial derivatives on a sloped
        one.  ``"quadrature"`` is the independent cross-check, the
        surface-current integral of the ring Green's function, which shares no
        arithmetic with the closed route in the ``R`` block, so a disagreement
        between them is a defect in one.
        """
        _checked_route(route)
        if self.start[0] == self.end[0] == 0.0:
            return 0.0, 0.0
        if r == 0.0:
            ends = _axis_field_ends(np, z, *self.start, *self.end)
            return 0.0, float(self.sheet_current * (ends[0] - ends[1]))
        if _near_axis(np, r, z, *self.start, *self.end):
            radial, height, flux = _axis_expansion(r, z, *self.start, *self.end)
            return float(self.sheet_current * (radial[0] - radial[1])), float(
                self.sheet_current * (height[0] - height[1])
            )
        series, ends, _ = _source_series_ends(np, r, z, *self.start, *self.end)
        if series and route == "closed":
            return tuple(
                float(self.sheet_current * (ends[0][i] - ends[1][i])) for i in (0, 1)
            )
        if self._flat:
            b_r, b_z = (
                self._pancake_field(r, z)
                if route == "closed"
                else self._flat_field(r, z)
            )
        elif route == "closed":
            b_r, b_z = self._sloped_field_closed(r, z)
        else:
            b_r, b_z = self._sloped_field(r, z)
        return float(b_r * self.sheet_current), float(b_z * self.sheet_current)

    def _vertical_flux(self, r: float, z: float) -> float:
        """Eq (24): the exact closed form of a straight cylinder."""
        source = float(self.start[0])
        if _near_axis(np, source, z, r, self.start[1], r, self.end[1]):
            _, _, flux = _axis_expansion(source, z, r, self.start[1], r, self.end[1])
            return float(flux[0] - flux[1])
        lower = self.antiderivative_uniform(
            float(self.start[1]) - z, r, float(self.start[0])
        )
        upper = self.antiderivative_uniform(
            float(self.end[1]) - z, r, float(self.end[0])
        )
        return MU0 * r * (upper - lower)

    def antiderivative_uniform(self, gamma: float, r: float, rp: float) -> float:
        """Eq (24) as a function of the source height, for the ``b_1 = 0`` case."""
        if rp == 0.0:
            return 0.0
        span = gamma**2 + (rp + r) ** 2
        a = np.sqrt(span)
        complement = 1.0 - 4.0 * r * rp / span
        first, second = complete_kind(np.asarray(complement))
        third = complete_pole(
            np.asarray(1.0 - 4.0 * r * rp / (rp + r) ** 2), np.asarray(complement)
        )
        return float(
            gamma
            / (2.0 * r)
            * (
                (gamma**2 + 2.0 * rp**2 + 2.0 * r**2) / a * first
                - a * second
                - (rp - r) ** 2 / a * third
            )
        )

    def _flat_flux(self, r: float, z: float) -> float:
        """The surface-current integral of the ring flux over a flat annulus."""
        z0 = float(self.start[1])
        lower, upper = sorted(float(v) for v in (self.start[0], self.end[0]))
        rp, w = self._rule(lower, upper)
        return float(
            np.sum(w * greens_psi(np.full_like(rp, r), np.full_like(rp, z), rp, z0))
        )

    def flux(self, r: float, z: float, *, route: str = "closed") -> float:
        """Poloidal flux [Wb] at ``(r, z)``, per the sheet current set at build.

        ``route`` names the evaluation family and means the same thing at every
        slope.  ``"closed"`` is the production route: eq (26) on a flat sheet,
        eq (24) on a vertical one, and the pole-reduced antiderivative on a
        sloped one.  ``"quadrature"`` is the independent cross-check, the
        surface-current integral of the ring flux, which shares no arithmetic
        with the closed route in the ``R`` block, so a disagreement between them
        is a defect in one.
        """
        _checked_route(route)
        if r == 0.0 or self.start[0] == self.end[0] == 0.0:
            return 0.0
        if _near_axis(np, r, z, *self.start, *self.end):
            radial, height, flux = _axis_expansion(r, z, *self.start, *self.end)
            return float(self.sheet_current * (flux[0] - flux[1]))
        series, _, fluxes = _source_series_ends(np, r, z, *self.start, *self.end)
        if series and route == "closed":
            return float(self.sheet_current * (fluxes[0] - fluxes[1]))
        if self._flat:
            value = (
                self._pancake_flux(r, z) if route == "closed" else self._flat_flux(r, z)
            )
        elif self._slope == 0.0 and route == "closed":
            value = self._vertical_flux(r, z)
        else:
            evaluate = (
                self.antiderivative_closed if route == "closed" else self.antiderivative
            )
            upper = evaluate(float(self.end[1]) - z, r, float(self.end[0]))
            lower = evaluate(float(self.start[1]) - z, r, float(self.start[0]))
            value = upper - lower
        return float(value * self.sheet_current)


def sheet_flux(
    r: float, z: float, start, end, *, sheet_current: float = 1.0, route: str = "closed"
) -> float:
    """Poloidal flux [Wb] of a conic sheet at ``(r, z)``."""
    return SheetFlux(start, end, sheet_current=sheet_current).flux(r, z, route=route)


def sheet_field(
    r: float, z: float, start, end, *, sheet_current: float = 1.0, route: str = "closed"
) -> tuple[float, float]:
    """``(B_R, B_Z)`` [T] of a conic sheet at ``(r, z)``."""
    return SheetFlux(start, end, sheet_current=sheet_current).field(r, z, route=route)


# --- the batched edge-sheet closed form -------------------------------
# Every helper below evaluates one whole block of (target, edge) pairs, so the
# tile route and the scalar route above are one implementation rather than two.
# The scalar class is the reference the packed form is checked against; the
# packed form is what the tiled build launches.
_SHEET_ARSINH_NODES = 128
_SHEET_FIELD_PANELS = 140
_SHEET_FIELD_RATIO = 1.25
_SHEET_FIELD_NODES = 24


@functools.lru_cache(maxsize=None)
def _sheet_graded_rule() -> tuple:
    """Return the graded arsinh rule's nodes and its azimuthal rule's nodes."""
    return np.polynomial.legendre.leggauss(64), np.polynomial.legendre.leggauss(
        _SHEET_FIELD_NODES
    )


def _sheet_root(gamma, radius, source):
    """``R``: the closed form's complete first- and second-kind ``D`` block.

    The same expression :func:`nova.biot.sheet` carries, written against the
    traced elliptic descent so one call evaluates a whole block of pairs.
    """
    return _root_expression(gamma, radius, source)


def _sheet_root_gradient(gamma, radius, source):
    """Return ``(R, dR/dgamma, dR/dradius)`` per pair.

    Each pair's ``R`` depends only on its own geometry, so the elementwise
    derivative is a ``vmap`` of JAX's own ``value_and_grad`` over the traced
    elliptic descent rather than a full Jacobian.
    """
    import jax

    value, gradient = jax.vmap(
        jax.value_and_grad(_sheet_root, argnums=(0, 1)), in_axes=(0, 0, 0)
    )(gamma, radius, source)
    return value, gradient


def _sheet_azimuth_rule(
    xp, width, *, angle=np.pi, panels=_SHEET_FIELD_PANELS, ratio=_SHEET_FIELD_RATIO
):
    """Return the field kernels' azimuth nodes and weights, shape ``(pairs, nodes)``.

    The rule is graded geometrically toward the singular azimuth and its width
    follows the target's own standoff, so it is a function of the traced geometry
    rather than a constant. The node count is fixed, which is what keeps the
    program one shape.
    """
    _, (node, weight) = _sheet_graded_rule()
    node = xp.asarray(node)
    weight = xp.asarray(weight)
    span = xp.arcsinh(angle / width)
    index = np.arange(panels + 1)
    edges = span[:, None] * (ratio**index - 1.0) / (ratio**panels - 1.0)
    half = 0.5 * (edges[:, 1:] - edges[:, :-1])
    centre = 0.5 * (edges[:, 1:] + edges[:, :-1])
    stretch = (centre[:, :, None] + half[:, :, None] * node[None, None, :]).reshape(
        width.shape[0], -1
    )
    rule = (half[:, :, None] * weight[None, None, :]).reshape(width.shape[0], -1)
    rule = rule * width[:, None] * xp.cosh(stretch)
    return width[:, None] * xp.sinh(stretch), rule


def _gradient_azimuth_rule(xp, gamma, radius, source, slope):
    """Resolve the ring distance and both pole layers of the same remainder.

    Splitting at pi/2 preserves the integral while grading each end. The
    inverse squared width sums the curvatures of D² and B² at both endpoints.
    A zero pole gap is removable; hold its inactive denominator before selecting.
    """
    intercept = source - slope * gamma
    square = 1 + slope**2
    inverse = radius * source / (gamma**2 + (radius - source) ** 2)
    for gap, curvature in (
        (intercept - radius, radius * (intercept - radius) + square * radius**2),
        (intercept + radius, -radius * (intercept + radius) + square * radius**2),
    ):
        held = xp.where(gap != 0, gap, radius)
        inverse = inverse + xp.where(gap != 0, xp.abs(curvature) / held**2, 0.0)
    width = 1 / xp.sqrt(inverse)
    stretch = xp.arcsinh(np.pi / (2 * width))
    resolved = (xp.abs(intercept - radius) > 1e-12 * radius) & (
        xp.abs(intercept + radius) > 1e-12 * radius
    )
    ring_pole = 2 * xp.arcsinh(
        xp.hypot(gamma, radius - source) / (2 * xp.sqrt(radius * source))
    )
    split = resolved & (stretch <= 10.0) & (width <= 1.0) & (ring_pole >= width)
    pole_root = xp.sqrt(square) * xp.hypot(intercept, slope * radius)
    for gap, sign in ((intercept - radius, 1), (intercept + radius, -1)):
        denominator = radius * (slope**2 * radius + sign * intercept + pole_root)
        present = denominator > 0
        held = xp.where(present, denominator, 1.0)
        pole_distance = 2 * xp.arcsinh(xp.abs(gap) / xp.sqrt(2 * held))
        split = split & (~present | (pole_distance >= width))
    held_width = xp.where(split, width, 1.0)
    phi, rule = _sheet_azimuth_rule(
        xp, held_width, angle=np.pi / 2, panels=_SHEET_FIELD_PANELS // 2, ratio=1.1
    )
    phi = xp.concatenate((phi, np.pi - phi), axis=1)
    rule = xp.concatenate((rule, rule), axis=1)
    distance_width = xp.hypot(gamma, radius - source) / xp.sqrt(radius * source)
    distance_width = xp.where(distance_width > 0, distance_width, 1.0)
    held_width = xp.where(split, 1.0, distance_width)
    single_phi, single_rule = _sheet_azimuth_rule(xp, held_width)
    single_phi = xp.where((intercept < 0)[:, None], np.pi - single_phi, single_phi)
    return xp.where(split[:, None], phi, single_phi), xp.where(
        split[:, None], rule, single_rule
    )


def _sheet_arsinh_rule(xp, gamma, radius, r1, b1):
    """Return the two graded panels and the integrand closure of the arsinh term."""
    a1sq = 1.0 + b1 * b1
    spread = r1 - radius
    upper = 2.0 * xp.sqrt(radius * xp.abs(r1 - radius * b1 * b1))
    # a target beyond the segment's ends or on the far side of its line puts
    # r_1 below -r, where the phi = 0 curvature is negative and r_1 + r is
    # negative; both enter the log model as magnitudes, the model being the
    # square root of a squared term, so they are taken unsigned rather than
    # left to an imaginary square root
    lower = 2.0 * xp.sqrt(radius * xp.abs(r1 + radius * b1 * b1))
    far = a1sq * gamma + b1 * (r1 + radius)
    near = a1sq * gamma + b1 * spread
    zero = xp.zeros_like(radius)
    panels = (
        (xp.abs(r1 + radius), far, upper, zero, QUARTER),
        (xp.abs(spread), near, lower, zero, QUARTER),
    )

    def pieces(x, y):
        return (
            near[:, None] + 2.0 * b1[:, None] * radius[:, None] * y,
            xp.sqrt(
                spread[:, None] ** 2
                + 4.0 * radius[:, None] * r1[:, None] * y
                + 4.0 * radius[:, None] ** 2 * b1[:, None] ** 2 * x * y
            ),
        )

    return panels, pieces


def _sheet_arsinh_term(xp, gamma, radius, r1, b1):
    """``int cos(phi) (r_1 + b_1^2 r cos phi) arsinh d phi``, log removed."""
    panels, pieces = _sheet_arsinh_rule(xp, gamma, radius, r1, b1)
    weight = (
        [-8.0 * b1 * b1 * radius],
        xp.asarray(2.0 * (r1 + b1 * b1 * radius)),
        xp.asarray(2.0 * (b1 * b1 * radius - r1)),
    )
    return graded_residual(panels, pieces, _SHEET_ARSINH_NODES, xp, weight=weight)


def _sheet_arsinh_gradient(xp, gamma, radius, source, b1):
    """Return ``(A, dA/dgamma, dA/dradius)`` of the arsinh remainder per pair."""
    a1sq = 1.0 + b1 * b1
    a1 = xp.asarray(1.0 + b1 * b1) ** 0.5
    r1 = source - b1 * gamma
    phi, rule = _gradient_azimuth_rule(xp, gamma, radius, source, b1)
    cosine = xp.cos(phi)
    sine_sq = xp.sin(phi) ** 2
    dense = xp.sqrt(
        (radius - source)[:, None] ** 2
        + gamma[:, None] ** 2
        + 4.0 * radius[:, None] * source[:, None] * xp.sin(0.5 * phi) ** 2
    )
    half_sine = xp.sin(0.5 * phi) ** 2
    gap = r1 - radius
    gap = xp.where(xp.abs(gap) <= 1e-12 * radius, 0.0, gap)
    W = gap[:, None] + 2.0 * radius[:, None] * half_sine
    denominator = W * W + a1sq[:, None] * radius[:, None] ** 2 * sine_sq
    gamma_bracket = a1sq[:, None] * gamma[:, None] + b1[:, None] * W
    arsinh = xp.arcsinh(gamma_bracket / xp.sqrt(denominator))
    smear = r1[:, None] + b1[:, None] ** 2 * radius[:, None] * cosine
    gamma_pole = (
        a1sq[:, None]
        * (
            W * (source[:, None] - radius[:, None] * cosine)
            + radius[:, None] ** 2 * sine_sq
        )
        / denominator
    )
    radial_pole = (
        -b1[:, None] * cosine * denominator
        - gamma_bracket
        * (
            -gap[:, None]
            + 2.0 * r1[:, None] * half_sine
            + radius[:, None] * b1[:, None] ** 2 * sine_sq
        )
    ) / denominator
    arsinh_plain = xp.sum(rule * cosine * arsinh, axis=1)
    gamma_value = -b1 * arsinh_plain + xp.sum(
        rule * cosine * smear * gamma_pole / (a1[:, None] * dense), axis=1
    )
    radial_value = b1 * b1 * xp.sum(rule * cosine * cosine * arsinh, axis=1) + xp.sum(
        rule * cosine * smear * radial_pole / (a1[:, None] * dense), axis=1
    )
    return _sheet_arsinh_term(xp, gamma, radius, r1, b1), gamma_value, radial_value


def _sheet_closed_gradient(xp, gamma, radius, source, b1):
    """Return ``(S, dS/dgamma, dS/dradius)`` of the closed-form antiderivative."""
    at_zero = source == 0.0
    limit = _zero_source_terms(xp, gamma, radius, b1)
    gamma = xp.where(at_zero, 1.0, gamma)
    radius = xp.where(at_zero, 1.0, radius)
    source = xp.where(at_zero, 0.5, source)
    b1 = xp.where(at_zero, 0.0, b1)
    a1 = xp.sqrt(1.0 + b1 * b1)
    root, gradient = _sheet_root_gradient(gamma, radius, source)
    value, gamma_value, radial_value = _sheet_arsinh_gradient(
        xp, gamma, radius, source, b1
    )
    regular = _source_gradient_series(
        xp,
        gamma,
        radius,
        source,
        b1,
        (
            a1 * b1 * root + value,
            a1 * b1 * gradient[0] + gamma_value,
            a1 * b1 * gradient[1] + radial_value,
        ),
    )
    return tuple(xp.where(at_zero, a, b) for a, b in zip(limit, regular))


def _sheet_pancake_terms(xp, gamma, radius, source):
    """``(A_phi, H_r, H_z)`` of Snow's eqs (26) and (27) at one flat-sheet end.

    The flat sheet's remainder is the paper's ``g_0^(2)(pi/2)``, taken through
    :func:`nova.biot.zeta.traced_zeta` so the same rule serves a block of pairs.
    A target level with the sheet's plane is the one place the printed blocks
    need their limit taken rather than their arithmetic, and it is selected
    arithmetically rather than by a branch.
    """
    at_zero = source == 0.0
    source = xp.where(at_zero, 0.5 * radius, source)
    span = gamma**2 + (radius + source) ** 2
    a = xp.sqrt(span)
    complement = (gamma**2 + (radius - source) ** 2) / span
    first, second = complete_kind(complement, xp=xp)
    c = xp.sqrt(gamma**2 + radius**2)
    remainder = traced_zeta(xp, source, radius, gamma, np.pi / 2)
    potential = (-source / (2.0 * a * radius)) * (
        (radius**2 - source**2 - 3.0 * gamma**2) * first + span * second
    )
    potential = potential + radius * remainder
    height = 2.0 * remainder - 2.0 * source / a * first
    level = xp.abs(gamma) <= 1e-12 * radius
    # a target level with the plane puts the first pole on the range end; the
    # block cancels to zero there, so the characteristic is held off the pole
    # and the term is zeroed rather than divided by zero
    held = xp.where(level, radius + radius, c)
    third = 0.0
    radial = 0.0
    for order in (1, 2):
        if order == 1:
            ratio = (source + c) / (radius - held)
            characteristic = 2.0 * radius / (radius - held)
        else:
            ratio = (source - c) / (radius + c)
            characteristic = 2.0 * radius / (radius + held)
        pole = complete_pole(
            xp.asarray(1.0 - characteristic), xp.asarray(complement), xp=xp
        )
        third = third + (-1) ** order * ratio * pole
        radial = radial + (-1) ** order * c * ratio * pole
    potential = potential - xp.where(
        level, 0.0, c * gamma**2 / (2.0 * a * radius) * third
    )
    radial = xp.where(
        level, 0.0, gamma / (a * radius) * (2.0 * source * first - radial)
    )
    radial, height = _pancake_end_series(xp, gamma, radius, source, radial, height)
    radial = xp.where(level, 0.0, radial)
    return tuple(xp.where(at_zero, 0.0, value) for value in (potential, radial, height))


def _sheet_pancake_field(xp, target_r, target_z, start_r, start_z, end_r, end_z):
    """``(B_R, B_Z)`` [T per A/m] of a flat edge, eq (27) over the segment."""
    gamma = xp.asarray(start_z) - target_z
    lower = xp.minimum(xp.asarray(start_r), xp.asarray(end_r))
    upper = xp.maximum(xp.asarray(start_r), xp.asarray(end_r))
    radial = 0.0
    height = 0.0
    for source, sign in ((upper, 1.0), (lower, -1.0)):
        _, end_radial, end_height = _sheet_pancake_terms(xp, gamma, target_r, source)
        radial = radial + sign * end_radial
        height = height + sign * end_height
    return MU0 / (2.0 * np.pi) * radial, MU0 / (2.0 * np.pi) * height


def _binary_field_geometry(xp, *coordinates):
    """Canonicalise a homogeneous field with an exact binary dilation.

    frexp chooses a shared exponent; ldexp changes exponent bits exactly for
    normal finite coordinates. A common input dilation therefore has identical
    canonical operands and follows identical floating-point arithmetic.
    """
    magnitude = xp.abs(xp.asarray(coordinates[0]))
    for coordinate in coordinates[1:]:
        magnitude = xp.maximum(magnitude, xp.abs(xp.asarray(coordinate)))
    _, exponent = xp.frexp(magnitude)
    return tuple(
        xp.ldexp(xp.asarray(coordinate), -exponent) for coordinate in coordinates
    )


def _sheet_end_terms(xp, target_r, target_z, start_r, start_z, end_r, end_z):
    """Return the per-end field terms of both the sloped and the flat route.

    Each component of :func:`packed_sheet_field` is a signed difference of two
    segment-end terms, so the terms themselves are what both the value and its
    rounding scale are built from. Returning them from one place is what keeps
    the two from drifting: the value is their signed sum and the scale is their
    absolute sum, per component, and neither restates the branch structure.

    The flat route's ends sit at the segment's radial extremes, the sloped
    route's at its ordered endpoints. ``flat`` selects between them.
    """
    target_r, target_z, start_r, start_z, end_r, end_z = _binary_field_geometry(
        xp, target_r, target_z, start_r, start_z, end_r, end_z
    )
    target_r = xp.asarray(target_r)
    target_z = xp.asarray(target_z)
    start_r = xp.asarray(start_r)
    start_z = xp.asarray(start_z)
    end_r = xp.asarray(end_r)
    end_z = xp.asarray(end_z)
    on_axis = target_r == 0.0
    zero_area = (start_r == 0.0) & (end_r == 0.0)
    near_axis = _near_axis(xp, target_r, target_z, start_r, start_z, end_r, end_z)
    expansion_inputs = (
        xp.where(near_axis, target_r, 0.001),
        xp.where(near_axis, target_z, -1.0),
        xp.where(near_axis, start_r, 0.5),
        xp.where(near_axis, start_z, 0.0),
        xp.where(near_axis, end_r, 0.5),
        xp.where(near_axis, end_z, 1.0),
    )
    if xp is np:
        expansion = (
            _axis_expansion(*expansion_inputs)
            if np.any(near_axis)
            else (np.zeros((2,) + target_r.shape),) * 3
        )
    else:
        expansion = jax.lax.cond(
            xp.any(near_axis),
            lambda _: _axis_expansion(*expansion_inputs),
            lambda _: (xp.zeros((2,) + target_r.shape),) * 3,
            operand=None,
        )
    axis_active = on_axis & ~zero_area
    axis_ends = _axis_field_ends(
        xp,
        xp.where(axis_active, target_z, -1.0),
        xp.where(axis_active, start_r, 0.5),
        xp.where(axis_active, start_z, 0.0),
        xp.where(axis_active, end_r, 0.5),
        xp.where(axis_active, end_z, 1.0),
    )
    inactive = on_axis | zero_area | near_axis
    target_r = xp.where(inactive, 2.0, target_r)
    target_z = xp.where(inactive, -1.0, target_z)
    start_r = xp.where(inactive, 0.5, start_r)
    start_z = xp.where(inactive, 0.0, start_z)
    end_r = xp.where(inactive, 0.5, end_r)
    end_z = xp.where(inactive, 1.0, end_z)
    source_series, source_ends, _ = _source_series_ends(
        xp, target_r, target_z, start_r, start_z, end_r, end_z
    )
    # The sheet current is an azimuthal density, so the field depends on the
    # segment and the sign of K alone. Ordering the endpoints by height makes
    # that independence arithmetic rather than a convention on the caller.
    swap = start_z > end_z
    lower_r = xp.where(swap, end_r, start_r)
    lower_z = xp.where(swap, end_z, start_z)
    upper_r = xp.where(swap, start_r, end_r)
    upper_z = xp.where(swap, start_z, end_z)
    drop_r = upper_r - lower_r
    drop_z = upper_z - lower_z
    flat = drop_z == 0.0
    held_z = xp.where(flat, xp.ones_like(drop_z), drop_z)
    b1 = drop_r / held_z
    a1sq = 1.0 + b1 * b1
    radial_scale = MU0 / (2.0 * np.pi * a1sq)
    height_scale = MU0 / (2.0 * np.pi * target_r * a1sq)
    sloped_radial = []
    sloped_height = []
    for start in (1.0, 0.0):
        gamma = (upper_z if start else lower_z) - target_z
        source = (upper_r if start else lower_r) + xp.zeros_like(target_z)
        value, gamma_value, radial_value = _sheet_closed_gradient(
            xp, gamma, target_r, source, b1
        )
        cylinder_radial = _cylinder_radial(xp, gamma, target_r, source)
        gamma_value = xp.where(
            b1 == 0.0, 2 * np.pi / MU0 * cylinder_radial, gamma_value
        )
        sloped_radial.append(radial_scale * gamma_value)
        cylinder_height = (
            MU0 / (2.0 * np.pi) * _sheet_cylinder_height(xp, gamma, target_r, source)
        )
        sloped_height.append(
            xp.where(
                b1 == 0.0,
                cylinder_height,
                height_scale * (value + target_r * radial_value),
            )
        )
    # A flat sheet's two ends are its radial extremes; its antiderivative is the
    # same difference, taken at those ends with the target's own height offset.
    gamma = start_z - target_z
    lower = xp.minimum(start_r, end_r)
    upper = xp.maximum(start_r, end_r)
    pancake_radial = []
    pancake_height = []
    for source in (upper, lower):
        _, end_radial, end_height = _sheet_pancake_terms(xp, gamma, target_r, source)
        pancake_radial.append(MU0 / (2.0 * np.pi) * end_radial)
        pancake_height.append(MU0 / (2.0 * np.pi) * end_height)
    for values, radial in (
        (sloped_radial, True),
        (sloped_height, False),
        (pancake_radial, True),
        (pancake_height, False),
    ):
        for index in range(2):
            values[index] = xp.where(
                source_series, source_ends[index][0 if radial else 1], values[index]
            )
            values[index] = xp.where(
                near_axis, expansion[0 if radial else 1][index], values[index]
            )
            value = xp.where(
                on_axis, 0.0 if radial else axis_ends[index], values[index]
            )
            values[index] = xp.where(zero_area, 0.0, value)
    return sloped_radial, sloped_height, pancake_radial, pancake_height, flat


def packed_sheet_field(xp, target_r, target_z, start_r, start_z, end_r, end_z):
    """Return ``(B_R, B_Z)`` [T per A/m] per pair from the edge-sheet closed form.

    The packed transcription of :meth:`SheetFlux._sloped_field_closed` and, for a
    horizontal edge, of :meth:`SheetFlux._pancake_field`: one call evaluates a
    whole block of (target, edge) pairs, each pair against its own segment, with
    no branch on a value. An edge's own slope selects its route arithmetically, so
    a tile mixing sloped and horizontal edges is one program.
    """
    sloped_radial, sloped_height, pancake_radial, pancake_height, flat = (
        _sheet_end_terms(xp, target_r, target_z, start_r, start_z, end_r, end_z)
    )
    return (
        xp.where(
            flat,
            pancake_radial[0] - pancake_radial[1],
            sloped_radial[0] - sloped_radial[1],
        ),
        xp.where(
            flat,
            pancake_height[0] - pancake_height[1],
            sloped_height[0] - sloped_height[1],
        ),
    )


def packed_sheet_end_scale(xp, target_r, target_z, start_r, start_z, end_r, end_z):
    """Return the absolute per-end scale of each field component, per pair.

    Two correctly rounded evaluations of a component disagree by at most the
    last bits of the terms it is summed from, so the largest disagreement either
    can carry is the absolute sum of those terms rather than the net value,
    which cancellation can leave orders of magnitude smaller. The terms are the
    ones :func:`packed_sheet_field` sums, taken from the same helper, so a
    component's scale can never be built from another component's terms.
    """
    sloped_radial, sloped_height, pancake_radial, pancake_height, flat = (
        _sheet_end_terms(xp, target_r, target_z, start_r, start_z, end_r, end_z)
    )
    return (
        xp.where(
            flat,
            xp.abs(pancake_radial[0]) + xp.abs(pancake_radial[1]),
            xp.abs(sloped_radial[0]) + xp.abs(sloped_radial[1]),
        ),
        xp.where(
            flat,
            xp.abs(pancake_height[0]) + xp.abs(pancake_height[1]),
            xp.abs(sloped_height[0]) + xp.abs(sloped_height[1]),
        ),
    )
