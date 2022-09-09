"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cached_property

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

    def __getitem__(self, attr):
        """Provide dict-like access to attributes."""
        return getattr(self, attr)

    @property
    def b(self):
        """Return b coefficient."""
        return self.rs + self.r

    @property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @property
    def gamma_zero(self):
        """Return gamma zero index."""
        return np.isclose(self.gamma, 0, atol=1e-12)

    @property
    def pi_zero(self):
        """Return rs-r zero index."""
        return np.isclose(self.rs - self.r, 0)

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

    @property
    def k2(self):
        """Return k2 coefficient."""
        return 4*self.r*self.rs / self.a2

    @property
    def ck2(self):
        """Return complementary modulus."""
        return 1 - self.k2

    @property
    def K(self):
        """Return elliptic intergral of the 1st kind."""
        return scipy.special.ellipk(self.k2)

    @property
    def E(self):
        """Return elliptic intergral of the 2nd kind."""
        return scipy.special.ellipe(self.k2)

    @staticmethod
    def ellippi(n, m):
        """Taken from https://github.com/scipy/scipy/issues/4452."""
        atol = 1e-18
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = scipy.special.elliprf(x, y, z)
        index = (np.isclose(x - z, 0, atol=atol) |
                 np.isclose(y - z, 0, atol=atol))
        if index.any():
            rf[index] = scipy.special.elliprc(x, y[index])
        rj = scipy.special.elliprj(x, y, z, p)
        index = (np.isclose(x - p, 0, atol=atol) |
                 np.isclose(y - p, 0, atol=atol) |
                 np.isclose(z - p, 0, atol=atol))
        if index.any():
            rj[index] = scipy.special.elliprd(x, y[index], z)
        return rf + rj * n / 3

    def np2(self, p: int):
        """Return np**2 constant."""
        if p == 1:
            return 2*self.r / self.zero_offset(self.r - self.c)
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

    def Pi(self, p: int):
        """Return complete elliptc intergral of the 3rd kind."""
        return self.ellippi(self.np2(p), self.k2)

    @property
    def U(self):
        """Return U coefficient."""
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)

    def zero_offset(self, array):
        """Return array with values close to zero offset to atol."""
        if (index := np.isclose(array, 0, atol=1e-12)).any():
            array[index] = 1e-12
        return array
