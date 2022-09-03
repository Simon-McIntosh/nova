"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import ClassVar

import dask.array as da
import numpy as np
import scipy.integrate

from nova.biot.biotconstants import BiotConstants
from nova.biot.biotmatrix import BiotMatrix


def gamma_zero(func):
    """Return result protected against degenerate values as gamma -> 0."""
    @wraps(func)
    def wrapper(self, i: int):
        result = func(self, i)
        if (index := np.isclose(self.gamma, 0)).any():
            result[index] = 0
        return result
    return wrapper


@dataclass
class CylinderConstants(BiotConstants):
    """Extend BiotConstants class."""

    romberg_k: int = 8

    alpha: ClassVar[float] = np.pi/2

    def __post_init__(self):
        """Build intergration parameters."""
        self.phi_zeta, self.dphi_zeta = \
            da.linspace(np.pi, np.pi - 2*self.alpha,
                        2**self.romberg_k + 1, retstep=True)
        print(self.dphi_zeta)

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
        return self.gamma*(self.rs - self.r * np.cos(phi)) / \
            (self.r * np.sin(phi) * np.sqrt(self.D2(alpha)))

    def Cphi_coef(self, alpha):
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

    def Cphi(self, alpha):
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


@dataclass
class BiotCylinder(BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    _corner: int = field(init=False, default=0, repr=False)

    name: ClassVar[str] = 'cylinder'  # element name
    attrs: ClassVar[dict[str, str]] = dict(
        rs='x', zs='z', dx='dx', dz='dz', r='x', z='z',
        area='area')

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.constant = [[] for _ in range(4)]
        for i, (unit_x, unit_z) in enumerate(zip([-1, 1, 1, -1],
                                                 [-1, -1, 1, 1])):
            self.constant[i] = CylinderConstants(
                self['rs'] + unit_x/2 * self['dx'],
                self['zs'] + unit_z/2 * self['dz'],
                self['r'], self['z'])

    @property
    def corner(self):
        """Return corner index."""
        return self._corner

    @corner.setter
    def corner(self, i: int):
        """Update corner index."""
        self._corner = i

    def __getattr__(self, attr):
        """Return coefficent evaluated at self.corner."""
        return self.constant[self.corner][attr]

    @gamma_zero
    def Aphi_hat(self, i: int):
        """Return vector potential intergration coefficient."""
        self.corner = i
        return self.Cphi(np.pi/2) + \
            self.gamma*self.r*self.zeta(np.pi/2) + \
            self.gamma*self.a / (6*self.r) * \
            (self.U*self.K - 2*self.rs*self.E) + \
            self.gamma / (6*self.a*self.r) * \
            da.sum(da.stack([(-1)**p * self.Pphi(p) * self.Pi(p) for
                             p in range(1, 4)]), axis=0)

    @gamma_zero
    def Br_hat(self, i: int):
        """Return radial magnetic field intergration coefficient."""
        self.corner = i

        return da.zeros_like(self['r'])

    @gamma_zero
    def Bz_hat(self, i: int):
        """Return vertical magnetic field intergration coefficient."""
        self.corner = i
        return da.zeros_like(self['r'])

    def _intergrate(self, func):
        """Return corner intergration."""
        return 1 / (4*np.pi*self['area']) * \
            ((func(2) - func(1)) - (func(3) - func(0)))

    @cached_property
    def Aphi(self):
        """Return Aphi dask array."""
        return self._intergrate(self.Aphi_hat)

    @property
    def Psi(self):
        """Return Psi dask array."""
        return 2 * np.pi * self.mu_o * self['r'] * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return self.mu_o * self._intergrate(self.Br_hat)

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return self.mu_o * self._intergrate(self.Bz_hat)


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=1, dplasma=-150)
    coilset.coil.insert(5, 0.5, 0.01, 0.8, section='r', turn='r',
                        nturn=300, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, section='r', turn='r',
                        nturn=300, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, section='r', turn='r',
                        nturn=300, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, section='r', turn='r',
                        nturn=300, segment='cylinder')
    coilset.saloc['Ic'] = 5e3

    coilset.grid.solve(2000, 1)
    coilset.grid.plot(colors='C1')
    coilset.plot()
