"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
import quadpy.c1

from nova.biot.biotconstants import BiotConstants
from nova.biot.biotmatrix import BiotMatrix


@dataclass
class CylinderConstants(BiotConstants):
    """Extend BiotConstants class."""

    alpha: ClassVar[float] = np.pi/2

    def __post_init__(self):
        """Build intergration parameters."""
        scheme = quadpy.c1.gauss_patterson(4)
        self.phi_points = np.pi - self.alpha * (scheme.points + 1)
        self.phi_weights = scheme.weights * self.alpha / 2
        super().__post_init__()

    @cached_property
    def v(self):
        """Return v coefficient."""
        return 1 + self.k2*(self.gamma**2 - self.b*self.r) / (2*self.r*self.rs)

    def B2(self, phi):
        """Return B2 coefficient."""
        return self.rs**2 + self.r**2 - 2*self.r*self.rs*np.cos(phi)

    def D2(self, phi):
        """Return D2 coefficient."""
        return self.gamma**2 + self.B2(phi)

    def G2(self, phi):
        """Return G2 coefficient."""
        return self.gamma**2 + self.r**2 * np.sin(phi)**2

    def beta1(self, phi):
        """Return beta1 coefficient."""
        return (self.rs - self.r * np.cos(phi)) / np.sqrt(self.G2(phi))

    def beta2(self, phi):
        """Return beta2 coefficient."""
        return self.gamma / np.sqrt(self.B2(phi))

    def beta3(self, phi):
        """Return beta3 coefficient."""
        return self.gamma*(self.rs - self.r * np.cos(phi)) / \
            (self.r * np.sin(phi) * np.sqrt(self.D2(phi)))

    def Cphi_alpha(self, alpha):
        """Return Cphi(alpha) coefficient."""
        phi = np.pi - 2*alpha
        return 1/2 * self.gamma*self.a * \
            np.sqrt(1 - self.k2 * np.sin(alpha)**2) * \
            -np.sin(2*alpha) - 1/6*np.arcsinh(self.beta2(phi)) * \
            np.sin(2*alpha) * (2*self.r**2 * np.sin(2*alpha)**2 +
                               3*(self.rs**2 - self.r**2)) - \
            1/4*self.gamma*self.r * \
            np.arcsinh(self.beta1(phi)) * \
            -np.sin(4*alpha) - 1/3*self.r**2 * \
            np.arctan(self.beta3(phi)) * -np.cos(2*alpha)**3

    @cached_property
    def Cphi_0(self):
        """Return Cphi(alpha=0) coefficient."""
        #return -1/3*self.r**2 * np.arctan(self.beta3(np.pi))
        return 1/3*self.r**2 * np.pi/2 * self.sign(self.gamma)

    @cached_property
    def Cphi_pi_2(self):
        """Return Cphi(alpha=pi/2) coefficient."""
        #return -1/3*self.r**2 * np.arctan(self.beta3(1e-6))
        return -1/3*self.r**2 * np.pi/2 * \
            self.sign(self.gamma * (self.rs - self.r))

    @cached_property
    def Cphi(self):
        """Return Cphi intergration constant evaluated between 0 and pi/2."""
        #return self.Cphi_alpha(np.pi/2-1e-6) - self.Cphi_alpha(1e-6)
        return self.Cphi_pi_2 - self.Cphi_0

    @cached_property
    def zeta(self):
        """Return zeta coefficient calculated using Romberg integration."""
        result = np.zeros_like(self.r)
        for phi, weight in zip(self.phi_points, self.phi_weights):
            result += weight * np.arcsinh(self.beta1(phi))
        return result

    @property
    def Qr(self) -> dict[int, np.ndarray]:
        """Return Qr(p) coefficient."""
        Qr = {p: (self.rs - (-1)**p * self.c) * self.np2[p] *
              self.gamma**2 * self.c / self.r for p in [1, 2]}
        Qr[3] = np.zeros_like(self.r)
        return Qr

    @property
    def Qz(self) -> dict[int, np.ndarray]:
        """Return Qz(p) coefficient."""
        Qz = {p: (self.rs - (-1)**p * self.c) *
              -2*self.gamma*self.c*self.np2[p] for p in [1, 2]}
        Qz[3] = self.gamma*self.b*(self.rs - self.r)*self.np2[3]
        return Qz

    @property
    def Dz(self):
        """Return Dz coefficient."""
        return 3/self.r*self.Cphi


@dataclass
class BiotCylinder(CylinderConstants, BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name: ClassVar[str] = 'cylinder'  # element name

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.rs = np.stack(
            [self.source('x') + delta/2 * self.source('dx')
             for delta in [-1, 1, 1, -1]], axis=-1)
        self.zs = np.stack(
            [self.source('z') + delta/2 * self.source('dz')
             for delta in [-1, -1, 1, 1]], axis=-1)
        self.r = np.stack([self.target('x') for _ in range(4)], axis=-1)
        self.z = np.stack([self.target('z') for _ in range(4)], axis=-1)

    def Aphi_hat(self):
        """Return vector potential intergration coefficient."""
        return self.Cphi + self.gamma*self.r*self.zeta + \
            self.gamma*self.a / (6*self.r) * \
            (self.U*self.K - 2*self.rs*self.E) + \
            self.gamma / (6*self.a*self.r) * self.p_sum(self.Pphi)

    def Br_hat(self):
        """Return radial magnetic field intergration coefficient."""
        return self.r*self.zeta - self.a / (2*self.r) * self.rs*(
            self.E - self.v*self.K) - 1/(4*self.a*self.r) * \
            self.p_sum(self.Qr)

    def Bz_hat(self):
        """Return vertical magnetic field intergration coefficient."""
        return self.Dz + 2*self.gamma*self.zeta - self.a / (2*self.r) * \
            3/2 * self.gamma*self.k2*self.K - 1/(4*self.a*self.r) * \
            self.p_sum(self.Qz)

    def _intergrate(self, data):
        """Return corner intergration."""
        return 1 / (2*np.pi*self.source('area')) * \
            ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0]))

    @cached_property
    def Aphi(self):
        """Return Aphi dask array."""
        return self._intergrate(self.Aphi_hat())

    @property
    def Psi(self):
        """Return Psi dask array."""
        return 2 * np.pi * self.mu_0 * self.target('x') * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return self.mu_0 * self._intergrate(self.Br_hat())

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return self.mu_0 * self._intergrate(self.Bz_hat())


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-2, nplasma=25**2)
    '''
    coilset.coil.insert(5, 0.5, 0.01, 0.8, segment='cylinder')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, segment='cylinder')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, segment='cylinder')
    '''
    coilset.firstwall.insert(5, 2, 1, 1,
                             turn='s', tile=False, segment='cylinder')

    #coilset.coil.insert(0.3, 0, 0.15, 0.15, segment='cylinder', delta=-1000)

    coilset.saloc['Ic'] = 5e3
    coilset.grid.solve(100, 1)

    coilset = CoilSet(dcoil=-2, nplasma=25**2)

    coilset = CoilSet(dcoil=-5, nplasma=5)
    #coilset.coil.insert(5, [-1.1, 1.1], 0.75, 0.75, Ic=[1, 1])
    coilset.firstwall.insert(dict(o=[5.25, 0, 0.5]), Ic=0.5)
    coilset.grid.solve(3e2, 1.5, 'plasma')  # generate plasma grid


    print()
    print(np.isfinite(coilset.grid.psi).all())
    print(np.sum(~np.isfinite(coilset.grid.psi)))

    coilset.plot()
    levels = coilset.grid.plot('psi', colors='C1', nulls=False)

    '''
    coilset = CoilSet()
    coilset.coil.insert(2, 0, 0.5, 0.5, section='r', turn='r',
                        delta=-8**2, tile=False, segment='ring')
    coilset.saloc['Ic'] = 5e3

    coilset.grid.solve(50**2, 0)
    levels = coilset.grid.plot('psi', colors='C0', nulls=False,
                               levels=levels)
    '''
