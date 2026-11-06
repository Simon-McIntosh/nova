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
ends (verified for the straight-cylinder limit against the 1e-13 ring reference
in ``scripts/edge_sheet/reference.py``).  The paper's eq (24) is the ``b_1 = 0``
form, a straight cylinder of radius ``r'`` and height ``|Z_2 - Z_1|``.

Only the straight-cylinder case is implemented here.  The sloped (``b_1 != 0``)
and flat-pancake (``b_2 = 0``) forms, Part II eqs (22a) to (27), need the
``arsinh`` integral and the third-kind branch bookkeeping; a segment that is not
parallel to the axis raises an error rather than returning a wrong number.
"""

from __future__ import annotations

import numpy as np

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.greens import MU0

__all__ = ["SheetFlux", "sheet_flux"]


class SheetFlux:
    """Poloidal flux of one sheet in the closed form of Part II eq (24)."""

    def __init__(self, start, end, *, sheet_current: float = 1.0):
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.sheet_current = float(sheet_current)
        if self.start.shape != (2,) or self.end.shape != (2,):
            raise ValueError("segment endpoints must be (r, z) pairs")
        span = self.end - self.start
        if span[1] == 0.0:
            raise NotImplementedError(
                "flat sheet (b_2 = 0) is Part II eq (26) and is not implemented"
            )
        if span[0] != 0.0:
            raise NotImplementedError(
                "sloped sheet (b_1 != 0) is Part II eq (22a) and is not implemented"
            )
        if span[1] < 0.0:
            self.start, self.end = self.end, self.start

    def antiderivative(self, gamma: float, r: float, rp: float):
        """Part II eq (24) as a function of the source height.

        ``gamma = z' - z`` is the source's height above the target and ``rp`` is
        the cylinder radius.  The paper's function is the antiderivative in the
        source coordinate, so the sheet's flux is this evaluated at the two ends
        and subtracted.
        """
        span = gamma**2 + (rp + r) ** 2
        a = np.sqrt(span)
        parameter = 4.0 * r * rp / span
        complement = 1.0 - parameter
        first, second = complete_kind(np.asarray(complement))
        characteristic = 4.0 * r * rp / (rp + r) ** 2
        third = complete_pole(np.asarray(1.0 - characteristic), np.asarray(complement))
        return (
            gamma
            / (2.0 * r)
            * (
                (gamma**2 + 2.0 * rp**2 + 2.0 * r**2) / a * first
                - a * second
                - (rp - r) ** 2 / a * third
            )
        )

    def flux(self, r: float, z: float) -> float:
        """Poloidal flux [Wb] at ``(r, z)``, per the sheet current set at build."""
        rp = self.start[0]
        upper = self.antiderivative(float(self.end[1]) - z, r, rp)
        lower = self.antiderivative(float(self.start[1]) - z, r, rp)
        return float(MU0 * r * (upper - lower) * self.sheet_current)


def sheet_flux(r: float, z: float, start, end, *, sheet_current: float = 1.0) -> float:
    """Poloidal flux [Wb] of a conic sheet at ``(r, z)``."""
    return SheetFlux(start, end, sheet_current=sheet_current).flux(r, z)
