"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
import scipy.special

# pylint: disable=W0631  # disable short names


@dataclass
class BiotConstants:
    """Manage biot intergration constants."""

    rs: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    zs: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    r: np.ndarray = field(default_factory=lambda: np.zeros_like([]))
    z: np.ndarray = field(default_factory=lambda: np.zeros_like([]))

    eps: ClassVar[np.float64] = 1.5 * np.finfo(float).eps

    def sign(self, x):
        """Return sign of array -1 if x < 0 else 1."""
        return np.where(abs(x) > self.eps, np.sign(x), 0)

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
    def _ellip(kind: str, /, *args, out=None, shape=None, where=True):
        if out is None:
            out = np.zeros_like(args[0], dtype=float, shape=shape)
        func = getattr(scipy.special, f'ellip{kind}')
        return func(*args, out=out, where=where)

    @classmethod
    def ellipk(cls, m):
        """Return complete elliptic intergral of the 1st kind."""
        return cls._ellip('k', m)

    @classmethod
    def ellipe(cls, m):
        """Return complete elliptic intergral of the 2nd kind."""
        return cls._ellip('e', m)

    @classmethod
    def ellippi(cls, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Adapted from https://github.com/scipy/scipy/issues/4452.
        """
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = cls._ellip('rf', x, y, z, shape=m.shape, where=(m < 1))
        #cls._ellip('rc', 0, 1, out=rf, where=np.isclose(y, 1))
        rj = cls._ellip('rj', x, y, z, p, shape=m.shape, where=(m < 1))
        #cls._ellip('rd', x, z, p, out=rj, where=np.isclose(y, p))
        #cls._ellip('rd', x, y, p, out=rj, where=np.isclose(p, 1))
        return rf + rj * n / 3

    @cached_property
    def np2(self) -> dict[int, np.ndarray]:
        """Return np**2 constant."""
        return {1: 2*self.r / (self.r - self.c - self.eps),
                2: 2*self.r / (self.r + self.c),
                3: 4*self.r*self.rs / self.b**2}

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
