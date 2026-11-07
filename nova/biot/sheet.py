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
integrates the bounded remainder: ``_weighted_residual`` carries the polynomial
weight ``cos(phi) (r_1 + b_1^2 r cos(phi))`` the sheet's own integrand needs.
The ``D`` term is smooth but has a square-root layer of width ``D_0`` at
``phi = 0``, so it is integrated on a sinh map anchored there, which is exact for
``D = sqrt(D_0^2 + 4 r r_1 sin^2(phi/2))``.  The straight-cylinder limit
``b_1 = 0`` is eq (24) and is evaluated in closed form.

Two printed coefficient blocks could not be reconstructed from the available
copy of the paper: the block feeding eq (22a) (``S``, ``M``, ``N`` and ``P_phi``
of eqs 13a and 18a) and the flat-segment coefficients of eq (26).  Both are
therefore not used; the sloped case is reduced as above, and the flat case
evaluates the surface-current integral of the ring flux
(:func:`nova.biot.greens.greens_psi`) on a fixed panelised rule.
"""

from __future__ import annotations

import numpy as np

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.gradedresidual import QUARTER
from nova.biot.greens import MU0, greens_psi

# ``_weighted_residual`` is private to the Part V module, which owns the graded
# rule and its endpoint-log removal.  Importing it keeps one implementation of
# that machinery; copying it here would let the two drift.
from nova.biot.polygonanalytic import _weighted_residual

__all__ = ["SheetFlux", "sheet_flux"]

_PANELS = 8
"""Sub-intervals of the segment rule, placed symmetrically."""

_NODES = 16
"""Gauss-Legendre nodes per panel; the rule is fixed, never adapted."""

_GAUSS = np.polynomial.legendre.leggauss(_NODES)

_ARSINH_NODES = 128
"""Nodes of the graded arsinh rule, the Part V module's own count."""

_GRADED_NODES = np.polynomial.legendre.leggauss(64)
"""The sinh map's nodes; it is exact for the layer the square root carries."""


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
        nodes, weights = _GAUSS
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
        node, weight = _GRADED_NODES
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
        a1sq = 1.0 + b1 * b1
        spread = r1 - r
        # the denominator's curvature at each graded end, read off the exact
        # squares (r_1 + r)^2 - 4 r (r_1 - r b_1^2) b^2 on the phi = pi panel and
        # (r_1 - r)^2 + 4 r (r_1 + r b_1^2) b^2 on the phi = 0 one
        upper = 2.0 * np.sqrt(r * np.abs(r1 - r * b1 * b1))
        lower = 2.0 * np.sqrt(r * (r1 + r * b1 * b1))
        far = a1sq * gamma + b1 * (r1 + r)
        near = a1sq * gamma + b1 * spread
        one = np.ones(1)
        panels = (
            (one * (r1 + r), one * far, one * upper, 0.0, QUARTER),
            (one * abs(spread), one * near, one * lower, 0.0, QUARTER),
        )

        def pieces(x, y):
            return (
                near + 2.0 * b1 * r * y,
                np.sqrt(spread**2 + 4.0 * r * r1 * y + 4.0 * r * r * b1 * b1 * x * y),
            )

        # cos(phi) = -c and d phi = 2 da, so the weight is -2 r_1 c + 2 b_1^2 r c^2;
        # the endpoint coefficients are its values at c = -1 and c = +1
        weight = (
            [-8.0 * b1 * b1 * r],
            np.full(1, 2.0 * (r1 + b1 * b1 * r)),
            np.full(1, 2.0 * (b1 * b1 * r - r1)),
        )
        value = _weighted_residual(panels, pieces, weight, _ARSINH_NODES, np)
        return float(np.asarray(value).sum())

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

    def flux(self, r: float, z: float) -> float:
        """Poloidal flux [Wb] at ``(r, z)``, per the sheet current set at build."""
        if self._flat:
            value = self._flat_flux(r, z)
        elif self._slope == 0.0:
            value = self._vertical_flux(r, z)
        else:
            upper = self.antiderivative(float(self.end[1]) - z, r, float(self.end[0]))
            lower = self.antiderivative(
                float(self.start[1]) - z, r, float(self.start[0])
            )
            value = upper - lower
        return float(value * self.sheet_current)


def sheet_flux(r: float, z: float, start, end, *, sheet_current: float = 1.0) -> float:
    """Poloidal flux [Wb] of a conic sheet at ``(r, z)``."""
    return SheetFlux(start, end, sheet_current=sheet_current).flux(r, z)
