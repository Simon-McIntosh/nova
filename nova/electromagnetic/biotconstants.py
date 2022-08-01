"""Biot-Savart intergration constants."""
from dataclasses import dataclass
from functools import cached_property

import dask.array as da
import numpy as np
import scipy.integrate
import scipy.special


# pylint: disable=no-member  # disable scipy.special module not found
# pylint: disable=W0631  # disable short names


@dataclass
class BiotConstants:
    """Manage biot intergration constants."""

    rs: da.Array
    zs: da.Array
    r: da.Array
    z: da.Array

    def __getitem__(self, attr):
        """Provide dict-like access to attributes."""
        return getattr(self, attr)

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
        return self.unit_offset(4*self.r*self.rs / self.a2)

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
        y = 1 - m
        rf = scipy.special.elliprf(0, y, 1)
        rj = scipy.special.elliprj(0, y, 1, 1 - n)
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
        return da.map_blocks(
            self.ellippi, self.unit_offset(self.np2(p)), self.k2)

    @cached_property
    def U(self):
        """Return U coefficient."""
        return self.k2 * (4*self.gamma**2 +
                          3*self.rs**2 - 5*self.r**2) / (4*self.r)

    def zero_offset(self, array, atol=1e-16):
        """Return array with values close to zero offset to atol."""
        if (index := da.isclose(array, 0, atol=atol)).any():
            array[index] = atol
        return array

    def unit_offset(self, array, atol=1e-16):
        #return array
        """Return array with values close to one offset to 1-atol."""
        if (index := da.isclose(array, 1, atol=atol)).any():
            array[index] = 1-atol
        return array

    def phi(self, alpha, atol=1e-16):
        """Return sysrem invariant angle transformation."""
        phi = np.pi - 2*alpha
        if np.isclose(phi, 0, atol=atol):
            phi = atol
        return phi

    def B2(self, alpha):
        """Return B2 coefficient."""
        phi = self.phi(alpha)
        return self.zero_offset(self.rs**2 + self.r**2 -
                                2*self.r*self.rs*np.cos(phi))

    def D2(self, alpha):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2(alpha)

    def G2(self, alpha):
        """Return G2 coefficient."""
        phi = self.phi(alpha)
        return self.gamma**2 + self.r**2 * np.sin(phi)**2

    def beta1(self, alpha):
        """Return beta1 coefficient."""
        phi = self.phi(alpha)
        return (self.rs - self.r * np.cos(phi)) / np.sqrt(self.G2(alpha))

    def beta2(self, alpha):
        """Return beta2 coefficient."""
        return self.gamma / np.sqrt(self.B2(alpha))

    def beta3(self, alpha):
        """Return beta3 coefficient."""
        phi = self.phi(alpha)
        return self.gamma*(self.rs - self.r * np.cos(phi)) / \
            (self.r * np.sin(phi) * np.sqrt(self.D2(alpha)))

    def Cphi_coef(self, alpha: float):
        """Return Cphi(alpha) coefficient."""
        return 1/2 * self.gamma*self.a * \
            np.sqrt(1 - self.k2 * np.sin(alpha)**2) * \
            -np.sin(2*alpha) - 1/6*np.arcsinh(self.beta2(alpha)) * \
            np.sin(2*alpha) * (2*self.r**2 * np.sin(2*alpha)**2 +
                               3*(self.rs**2 - self.r**2)) - \
            1/4*self.gamma*self.r * \
            np.arcsinh(self.beta1(alpha)) * \
            -np.sin(4*alpha) - 1/3*self.r**2 * \
            np.arctan(self.beta3(alpha)) * -np.cos(2*alpha)**3

    def Cphi(self, alpha: float):
        """Return Cphi intergration constant."""
        return self.Cphi_coef(alpha) - self.Cphi_coef(0)

    def zeta(self, alpha, k=8):
        """Return zeta coefficient calculated using Romberg integration."""
        alpha, dalpha = da.linspace(0, alpha, 2**k + 1, retstep=True)
        beta1 = da.stack([self.beta1(_alpha) for _alpha in alpha])
        asinh_beta1 = np.arcsinh(beta1).rechunk({0: -1, 1: 'auto', 2: 'auto'},
                                                block_size_limit=1e8)
        return asinh_beta1.map_blocks(scipy.integrate.romb, dx=dalpha, axis=0,
                                      dtype=float, drop_axis=0)
