"""Biot-Savart intergration constants."""
from dataclasses import dataclass
from functools import cached_property

import dask.array as da
import numpy as np
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
        """Return a2 coefficient."""
        return self.gamma**2 + (self.rs + self.r)**2

    @cached_property
    def a(self):
        """Return a coefficient."""
        return da.sqrt(self.a2)

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

    def phi(self, alpha):
        """Return sysrem invariant angle transformation."""
        return np.pi - 2*alpha

    def B2(self, alpha):
        """Return B2 coefficient."""
        phi = self.phi(alpha)
        return self.rs**2 + self.r**2 - 2*self.r*self.rs*np.cos(phi)

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
        if np.isclose(phi, 0):  # arctan(1/0)
            phi += 1e-16
        return self.gamma*(self.rs - self.r * np.cos(phi)) / \
            (self.r * np.sin(phi) * np.sqrt(self.D2(alpha)))
