"""Biot-Savart intergration constants."""
from dataclasses import dataclass, field
from functools import cached_property

import dask.array as da
import numpy as np
import scipy.special


# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


@dataclass
class BiotConstants:
    """Manage biot intergration constants."""

    rs: da.Array = field(default_factory=lambda: da.zeros_like([]))
    zs: da.Array = field(default_factory=lambda: da.zeros_like([]))
    r: da.Array = field(default_factory=lambda: da.zeros_like([]))
    z: da.Array = field(default_factory=lambda: da.zeros_like([]))

    def __getitem__(self, attr):
        """Provide dict-like access to attributes."""
        return getattr(self, attr)

    def phi(self, alpha):
        """Return sysrem invariant angle transformation."""
        phi = np.pi - 2*alpha
        if np.isclose(phi, 0, atol=1e-16):
            return 1e-16
        return phi

    @cached_property
    def b(self):
        """Return b coefficient."""
        return self.rs + self.r

    @cached_property
    def gamma(self):
        """Return gamma coefficient."""
        return self.zs - self.z

    @cached_property
    def a2(self):
        """Return a**2 coefficient."""
        return self.gamma**2 + (self.rs + self.r)**2

    @cached_property
    def a(self):
        """Return a coefficient."""
        return da.sqrt(self.a2)

    @cached_property
    def c2(self):
        """Return c**2 coefficient."""
        return self.gamma**2 + self.r**2

    @cached_property
    def c(self):
        """Return c coefficient."""
        return da.sqrt(self.c2)

    @cached_property
    def k2(self):
        """Return k2 coefficient."""
        return 4*self.r*self.rs / self.a2

    @cached_property
    def ck2(self):
        """Return complementary modulus."""
        return 1 - self.k2

    @cached_property
    def K(self):
        """Return elliptic intergral of the 1st kind."""
        return self.k2.map_blocks(scipy.special.ellipk, dtype=float)

    @cached_property
    def E(self):
        """Return elliptic intergral of the 2nd kind."""
        return self.k2.map_blocks(scipy.special.ellipe, dtype=float)

    @staticmethod
    def ellippi(n, m):
        """Taken from https://github.com/scipy/scipy/issues/4452."""
        x, y, z, p = 0, 1-m, 1, 1-n
        rf = scipy.special.elliprf(x, y, z)
        rj = scipy.special.elliprj(x, y, z, p)
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
        return da.map_blocks(self.ellippi, self.np2(p), self.k2, dtype=float)

    @cached_property
    def U(self):
        """Return U coefficient."""
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)

    def zero_offset(self, array):
        """Return array with values close to zero offset to atol."""
        if (index := da.isclose(array, 0, atol=1e-16)).any():
            array[index] = 1e-16
        return array
