"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cache, cached_property, wraps
from typing import ClassVar

import dask.array as da
import numpy as np
import scipy.special

Array = da.Array | np.ndarray

# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


@dataclass
class BiotConstants:
    """Manage biot intergration constants."""

    rs: Array = field(default_factory=lambda: da.zeros_like([]))
    zs: Array = field(default_factory=lambda: da.zeros_like([]))
    r: Array = field(default_factory=lambda: da.zeros_like([]))
    z: Array = field(default_factory=lambda: da.zeros_like([]))

    eps: ClassVar[np.float64] = 1e4 * np.finfo(float).eps

    def sign(self, x):
        """Return sign of array -1 if x < 0 else 1."""
        return np.where(x < -self.eps, -1, 1)
        return 2*(x >= 0) - 1

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @property
    def a2(self):
        """Return a**2 coefficient."""
        return self.gamma**2 + (self.rs + self.r)**2

    @property
    def a(self):
        """Return a coefficient."""
        return np.sqrt(self.a2)

    @property
    def b(self):
        """Return b coefficient."""
        return self.rs + self.r

    @property
    def c2(self):
        """Return c**2 coefficient."""
        return self.gamma**2 + self.r**2

    @property
    def c(self):
        """Return c coefficient."""
        return np.sqrt(self.c2)

    @cached_property
    def k2(self):
        """Return k2 coefficient."""
        return (1-self.eps) * 4*self.r*self.rs / self.a2

    @property
    def ck2(self):
        """Return complementary modulus."""
        return 1 - self.k2

    @cached_property
    def K(self):
        """Return complete elliptic intergral of the 1st kind."""
        return self.ellipk(self.k2)

    @cached_property
    def E(self):
        """Return complete elliptic intergral of the 2nd kind."""
        return self.ellipe(self.k2)

    @cached_property
    def U(self):
        """Return U coefficient."""
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)

    @staticmethod
    def ellipk(m):
        """Return complete elliptic intergral of the 1st kind."""
        return scipy.special.ellipk(m)

    @staticmethod
    def ellipe(m):
        """Return complete elliptic intergral of the 1st kind."""
        return scipy.special.ellipe(m)

    @staticmethod
    def ellippi(n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Taken from https://github.com/scipy/scipy/issues/4452.
        """
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = scipy.special.elliprf(x, y, z)
        rf[(y == 0) | (y == 1)] = scipy.special.elliprc(0, 1)
        rj = scipy.special.elliprj(x, y, z, p)
        rj[y == p] = scipy.special.elliprd(x, z, p[y == p])
        rj[p == 1] = scipy.special.elliprd(x, y[p == 1], 1)
        return rf + rj * n / 3

    @cached_property
    def np2(self) -> dict[int, np.ndarray]:
        """Return np**2 constant."""
        return {1: 2*self.r / (self.r - self.c - self.eps),
                2: (1-self.eps) * 2*self.r / (self.r + self.c),
                3: (1-self.eps) * 4*self.r*self.rs / self.b**2}

    @cached_property
    def Pphi(self) -> dict[int, np.ndarray]:
        """Return Pphi coefficient, q in [1, 2, 3]."""
        Pphi = {p: (self.rs - (-1)**p * self.c) * self.np2[p] * self.c *
                (3*self.r**2 - self.c2) / (2*self.r) for p in [1, 2]}
        Pphi[3] = -self.rs / self.b * (self.rs - self.r) * \
            (3*self.r**2 - self.rs**2)
        return Pphi

    @cached_property
    def Pi(self) -> dict[int, np.ndarray]:
        """Return complete elliptc intergral of the 3rd kind."""
        return {p: self.ellippi(self.np2[p], self.k2) for p in range(1, 4)}

    def p_sum(self, func):
        """Return p sum."""
        result = np.zeros_like(self.r)
        for p in range(1, 4):
            result += (-1)**p * func[p] * self.Pi[p]
        return result
