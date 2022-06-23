"""Biot-Savart intergration constants."""
from dataclasses import dataclass
from functools import cached_property

import dask.array as da
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
