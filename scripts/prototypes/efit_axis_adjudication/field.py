"""Vectorised access to a record's own not-a-knot bicubic spline.

The field is the scipy RectBivariateSpline (kx=ky=3, s=0) of the record's own
65x65 grid, the same object the rebuild check uses.  Its coefficient matrix and
knots are reused here to evaluate psi, its gradient and its Hessian on many
points at once, so a dense Newton census is cheap.  No adapter code is used.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, RectBivariateSpline


class FastField:
    def __init__(self, rs, zs, psi):
        self.rbf = RectBivariateSpline(rs, zs, psi, kx=3, ky=3, s=0)
        tr, tz = self.rbf.get_knots()
        self.tr, self.tz = tr, tz
        nr, nz = len(tr) - 4, len(tz) - 4
        self.C = self.rbf.get_coeffs().reshape(nr, nz)
        self._br = BSpline(tr, np.eye(nr), 3)
        self._bz = BSpline(tz, np.eye(nz), 3)
        self._br1 = self._br.derivative(1)
        self._br2 = self._br.derivative(2)
        self._bz1 = self._bz.derivative(1)
        self._bz2 = self._bz.derivative(2)

    def jet(self, R, Z):
        """Return (psi, gr, gz, hrr, hrz, hzz) at points R, Z (1-D arrays)."""
        br = self._br(R)
        br1 = self._br1(R)
        br2 = self._br2(R)
        bz = self._bz(Z)
        bz1 = self._bz1(Z)
        bz2 = self._bz2(Z)
        C = self.C
        value = np.einsum("mi,ij,mj->m", br, C, bz)
        gr = np.einsum("mi,ij,mj->m", br1, C, bz)
        gz = np.einsum("mi,ij,mj->m", br, C, bz1)
        hrr = np.einsum("mi,ij,mj->m", br2, C, bz)
        hzz = np.einsum("mi,ij,mj->m", br, C, bz2)
        hrz = np.einsum("mi,ij,mj->m", br1, C, bz1)
        return value, gr, gz, hrr, hrz, hzz

    def value_gradient(self, R, Z):
        value, gr, gz, _, _, _ = self.jet(R, Z)
        return value, np.column_stack((gr, gz))