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
        #if (index := da.isclose(self.gamma, 0)).any():
        #    result[index] = 0
        result[da.isclose(self.gamma, 0)] = 0
        return result
    return wrapper


@dataclass
class CylinderConstants(BiotConstants):
    """Extend BiotConstants class."""

    romberg_k: int = 8
    alpha: float = np.pi/2

    def __post_init__(self):
        """Build intergration parameters."""
        if self.alpha >= np.pi/2:
            self.alpha -= 1e-14
        self.phi_zeta = da.linspace(np.pi, np.pi - 2*self.alpha,
                                    2**self.romberg_k + 1)
        self.dalpha_zeta = self.alpha / 2**self.romberg_k

    def B2(self, phi):
        """Return B2 coefficient."""
        return self.rs**2 + self.r**2 - 2*self.r*self.rs*da.cos(phi)

    def D2(self, phi):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2(phi)

    def G2(self, phi):
        """Return G2 coefficient."""
        return self.gamma**2 + self.r**2 * da.sin(phi)**2

    def beta1(self, phi):
        """Return beta1 coefficient."""
        return (self.rs - self.r * da.cos(phi)) / da.sqrt(self.G2(phi))

    def beta2(self, phi):
        """Return beta2 coefficient."""
        return self.gamma / da.sqrt(self.B2(phi))

    def beta3(self, phi):
        """Return beta3 coefficient."""
        return self.gamma*(self.rs - self.r * da.cos(phi)) / \
            (self.r * da.sin(phi) * da.sqrt(self.D2(phi)))

    def Cphi_coef(self, alpha):
        """Return Cphi(alpha) coefficient."""
        phi = np.pi - 2*alpha
        return 1/2 * self.gamma*self.a * \
            da.sqrt(1 - self.k2 * da.sin(alpha)**2) * \
            -da.sin(2*alpha) - 1/6*da.arcsinh(self.beta2(phi)) * \
            da.sin(2*alpha) * (2*self.r**2 * da.sin(2*alpha)**2 +
                               3*(self.rs**2 - self.r**2)) - \
            1/4*self.gamma*self.r * \
            da.arcsinh(self.beta1(phi)) * \
            -da.sin(4*alpha) - 1/3*self.r**2 * \
            da.arctan(self.beta3(phi)) * -da.cos(2*alpha)**3

    @property
    def Cphi(self):
        """Return Cphi intergration constant."""
        return self.Cphi_coef(self.alpha) - self.Cphi_coef(0)

    @property
    def zeta(self):
        """Return zeta coefficient calculated using Romberg integration."""
        asinh_beta1 = da.stack([da.arcsinh(self.beta1(phi))
                                for phi in self.phi_zeta])
        return asinh_beta1.map_blocks(
            scipy.integrate.romb, dx=self.dalpha_zeta, axis=0, dtype=float,
            drop_axis=0)


@dataclass
class BiotCylinder(BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    romberg_k: int = 8
    _corner: int = field(init=False, default=0, repr=False)

    name: ClassVar[str] = 'cylinder'  # element name
    attrs: ClassVar[dict[str, str]] = dict(
        rs='x', zs='z', dx='dx', dz='dz', r='x', z='z',
        area='area')

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.constant = [[] for _ in range(4)]
        print('r', self.target('x'))
        print('rs', self.source('x'))

        for i, (unit_x, unit_z) in enumerate(zip([-1, 1, 1, -1],
                                                 [-1, -1, 1, 1])):
            self.constant[i] = CylinderConstants(
                self['rs'] + unit_x/2 * self['dx'],
                self['zs'] + unit_z/2 * self['dz'],
                self['r'], self['z'], romberg_k=self.romberg_k)

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
        return self.Cphi + self.gamma*self.r*self.zeta +\
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
