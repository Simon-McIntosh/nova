"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cached_property, wraps

import dask.array as da
import numpy as np
import scipy.special

Array = da.Array | np.ndarray

# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


def gamma_zero(func):
    """Return result protected against degenerate values as gamma -> 0."""
    @wraps(func)
    def wrapper(self, r, c):
        index = self.gamma_zero_index
        result = np.zeros_like(self.r)
        result[~index] = func(self, r[~index], c[~index])
        return result
    return wrapper


def unit_m(func):
    """Return result protected against degenerate values as m -> 1."""
    @wraps(func)
    def wrapper(self, m):
        index = np.isclose(m-1, 0)
        result = np.zeros_like(self.r)
        result[~index] = func(self, m[~index])
        return result
    return wrapper


def unit_nm(func):
    """Return result protected against degenerate values as n | m -> 1."""
    @wraps(func)
    def wrapper(self, n, m):
        index = np.isclose(n-1, 0) | np.isclose(m-1, 0)
        result = np.zeros_like(self.r)
        result[~index] = func(self, n[~index], m[~index])
        return result
    return wrapper


@dataclass
class BiotConstants:
    """Manage biot intergration constants."""

    rs: Array = field(default_factory=lambda: da.zeros_like([]))
    zs: Array = field(default_factory=lambda: da.zeros_like([]))
    r: Array = field(default_factory=lambda: da.zeros_like([]))
    z: Array = field(default_factory=lambda: da.zeros_like([]))

    @property
    def b(self):
        """Return b coefficient."""
        return self.rs + self.r

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @cached_property
    def gamma_zero_index(self):
        """Return gamma==zero index."""
        return np.isclose(self.gamma, 0)

    @property
    def a2(self):
        """Return a**2 coefficient."""
        return self.gamma**2 + (self.rs + self.r)**2

    @property
    def a(self):
        """Return a coefficient."""
        return np.sqrt(self.a2)

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
        return 4*self.r*self.rs / self.a2

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

    @unit_m
    def ellipk(self, m):
        """Return complete elliptic intergral of the 1st kind."""
        return scipy.special.ellipk(m)

    @unit_m
    def ellipe(self, m):
        """Return complete elliptic intergral of the 1st kind."""
        return scipy.special.ellipe(m)

    @unit_nm
    def ellippi(self, n, m):
        """
        Return complete elliptic intergral of the 3rd kind.

        Taken from https://github.com/scipy/scipy/issues/4452.
        """
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = scipy.special.elliprf(x, y, z)
        rj = scipy.special.elliprj(x, y, z, p)
        return rf + rj * n / 3

    @gamma_zero
    def np2_1(self, r, c):
        """Return np**2(p=1) constant."""
        return 2*r / (r - c)

    def np2(self, p: int):
        """Return np**2 constant."""
        if p == 1:
            return self.np2_1(self.r, self.c)
        if p == 2:
            return 2*self.r / (self.r + self.c)
        if p == 3:
            return 4*self.r*self.rs / self.b**2

    def Pphi(self, p: int):
        """Return Pphi coefficient, q=2."""
        if p == 3:
            return -self.rs / self.b * (self.rs - self.r) * \
                (3*self.r**2 - self.rs**2)
        return (self.rs - (-1)**p * self.c) * self.np2(p) * self.c * \
            (3*self.r**2 - self.c2) / (2*self.r)

    def Pi_p(self, p: int):
        """Return complete elliptc intergral of the 3rd kind."""
        return self.ellippi(self.np2(p), self.k2)

    @cached_property
    def Pi_1(self):
        """Return Pi(1)."""
        return self.ellippi(self.np2(1), self.k2)

    @cached_property
    def Pi_2(self):
        """Return Pi(2)."""
        return self.ellippi(self.np2(2), self.k2)

    @cached_property
    def Pi_3(self):
        """Return Pi(3)."""
        return self.ellippi(self.np2(3), self.k2)

    def p_sum(self, func):
        """Return p sum."""
        result = np.zeros_like(self.r)
        for p in range(1, 4):
            result += (-1)**p * func(p) * getattr(self, f'Pi_{p}')
        return result

    @cached_property
    def U(self):
        """Return U coefficient."""
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)
