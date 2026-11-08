"""Poloidal flux of a conic toroidal current sheet.

A magnetised cell's edge carries a thin sheet: a straight segment in the poloidal
plane swept through a full turn, bearing a uniform azimuthal surface current
``K``.  This module evaluates the poloidal flux ``psi`` threading a circle at
``(r, z)`` from the closed form of L. K. Urankar, "Current-carrying finite arc
segment, Part II", IEEE Trans. Magn. MAG-18(3), 911-917, 1982.

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
from nova.biot.greens import MU0, greens_bz_br, greens_psi
from nova.biot.zeta import zeta
from nova.jax.config import configure_dtypes

__all__ = ["ROUTES", "SheetFlux", "sheet_field", "sheet_flux"]

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


@jax.jit
def _closed_root(height, radius, source):
    """``R``: the closed form's complete first- and second-kind ``D`` block.

    ``2a[2(1 - k^2)K + (k^2 - 2)E] / (3k^2)`` of eq (22a), with the modulus
    ``k^2 = 4 r r' / a^2`` and its complement both formed from the geometry
    rather than by subtraction.
    """
    span = height * height + (radius + source) ** 2
    k2 = 4.0 * radius * source / span
    complement = (height * height + (radius - source) ** 2) / span
    first, second = complete_kind_traced(complement)
    return (
        2.0 * span**0.5 * (2.0 * (1.0 - k2) * first + (k2 - 2.0) * second) / (3.0 * k2)
    )


_closed_root_gradient = jax.jit(jax.value_and_grad(_closed_root, argnums=(0, 1)))


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
        root = 2.0 * a * (2.0 * (1.0 - k2) * K + (k2 - 2.0) * E) / (3.0 * k2)
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
        b1 = self._slope
        a1sq = 1.0 + b1 * b1
        r1 = rp - b1 * gamma
        a1 = np.sqrt(a1sq)
        width = np.hypot(gamma, r - rp) / np.sqrt(r * rp)
        width = width if width > 0.0 else 1.0
        # The arsinh argument's denominator vanishes at the azimuth where the
        # source ring passes through the target, which is phi = 0 when
        # r_1 = r and phi = pi when r_1 = -r.  The rule grades only one end, so
        # it is placed at whichever end is nearer that confluence: a target on
        # the far side of the line puts r_1 below -r, and the end carrying the
        # logarithmic layer is then the far one.
        phi, rule = self._azimuth_rule(width)
        if np.abs(r1 + r) < np.abs(r1 - r):
            phi = np.pi - phi
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
        b1 = self._slope
        a1 = np.sqrt(1.0 + b1 * b1)
        root, gradient = _closed_root_gradient(
            jnp.asarray(gamma), jnp.asarray(r), jnp.asarray(rp)
        )
        arsinh_term, arsinh_gamma, arsinh_radial = self._arsinh_gradient(gamma, r, rp)
        return (
            float(a1 * b1) * float(root) + arsinh_term,
            float(a1 * b1) * float(gradient[0]) + arsinh_gamma,
            float(a1 * b1) * float(gradient[1]) + arsinh_radial,
        )

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
            b_r += sign * gamma_value
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
        return potential, gamma / (a * r) * (2.0 * rp * big_k - radial), height

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
        lower = self.antiderivative_uniform(
            float(self.start[1]) - z, r, float(self.start[0])
        )
        upper = self.antiderivative_uniform(
            float(self.end[1]) - z, r, float(self.end[0])
        )
        return MU0 * r * (upper - lower)

    def antiderivative_uniform(self, gamma: float, r: float, rp: float) -> float:
        """Eq (24) as a function of the source height, for the ``b_1 = 0`` case."""
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
