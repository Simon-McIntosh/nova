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

The complete elliptic kinds enter through the closed form of the ``D`` part; the
``arsinh`` part is evaluated on a fixed panelised Gauss-Legendre rule, the same
class of rule the paper's eq (12b) leaves its residual integral to.  The
straight-cylinder limit ``b_1 = 0`` is eq (24) and is evaluated in closed form.

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
from nova.biot.greens import MU0, greens_psi

__all__ = ["SheetFlux", "sheet_flux"]

_PANELS = 8
"""Sub-intervals of the azimuthal and segment rules, placed symmetrically."""

_NODES = 16
"""Gauss-Legendre nodes per panel; the rule is fixed, never adapted."""

_GAUSS = np.polynomial.legendre.leggauss(_NODES)


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

    def antiderivative(self, gamma: float, r: float, rp: float) -> float:
        """The sloped-segment antiderivative, integrated over the azimuth.

        ``gamma = z' - z`` is the source's height above the target and ``rp`` its
        radius.  The paper's function is the antiderivative in the source
        coordinate, so the segment enters as the difference between the two ends.
        ``rp`` must lie on the segment's supporting line, ``rp = b_1 z' + c_1``.
        """
        b1 = self._slope
        a1sq = 1.0 + b1 * b1
        a1 = np.sqrt(a1sq)
        r1 = rp - b1 * gamma
        phi, w = self._rule(0.0, np.pi)
        cos = np.cos(phi)
        sin = np.sin(phi)
        gap = gamma**2 + r * r + rp * rp - 2.0 * r * rp * cos
        dense = np.sqrt(gap)
        squared = r1 * r1 - 2.0 * r * r1 * cos + r * r * (1.0 + b1 * b1 * sin * sin)
        gamma1 = a1sq * gamma + b1 * (r1 - r * cos)
        integrand = (
            a1 * b1 * dense
            + (r1 + b1 * b1 * r * cos) * np.arcsinh(gamma1 / np.sqrt(squared))
        ) / a1**3
        return float(MU0 * r * a1 * np.sum(w * cos * integrand))

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
