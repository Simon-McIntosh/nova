"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nova.electromagnetic.biotconstants import BiotConstants
from nova.electromagnetic.biotmatrix import BiotMatrix


@dataclass
class BiotCylinder(BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name: ClassVar[str] = 'cylinder'  # element name
    attrs: ClassVar[list[str]] = dict(
        rs='x', zs='z', dx='dx', dz='dz', r='x', z='z')

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        self.const = [[] for _ in range(4)]
        for i, (_dx, _dz) in enumerate(zip([-0.5, 0.5, 0.5, -0.5],
                                           [-0.5, -0.5, 0.5, 0.5])):
            self.const[i] = BiotConstants(
                self['rs']+_dx*self['dx'], self['zs']+_dz*self['dz'],
                self['r'], self['z'])

    def _Cphi(self, i: int, alpha: float):
        """Return Cphi(alpha) constant evaluated at corner i."""
        return 0.5*self.const[i]['gamma']*self.const[i].a * \
            np.sqrt(1 - self.const[i].k2 * np.sin(alpha)**2) * \
            -np.sin(2*alpha) - 1/6*np.arcsinh(self.const[i].beta2(alpha)) * \
            np.sin(2*alpha) * (2*self['r']**2 * np.sin(2*alpha)**2 +
                               3*(self['rs']**2 - self['r']**2)) - \
            1/4*self.const[i].gamma*self['r'] * \
            np.arcsinh(self.const[i].beta1(alpha))

    @property
    def Aphi(self):
        """Return Aphi dask array."""
        return 1 / (2*np.pi) * self.const['a']/self['r'] * \
            ((1 - self.const['k2']/2) * self.const['K'] - self.const['E'])

    @property
    def Psi(self):
        """Return Psi dask array."""
        return 2 * np.pi * self.mu_o * self['r'] * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return self.mu_o / (2*np.pi) * self.const['gamma'] * \
            (self.const['K'] - (2-self.const['k2']) / (2*self.const['ck2']) *
             self.const['E']) / (self.const['a'] * self['r'])

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return self.mu_o / (2*np.pi) * \
            (self['r']*self.const['K'] -
             (2*self['r'] - self.const['b']*self.const['k2']) /
             (2*self.const['ck2']) * self.const['E']) / \
            (self.const['a']*self['r'])


if __name__ == '__main__':

    from nova.electromagnetic.framespace import FrameSpace

    radius, height = np.meshgrid(np.linspace(4, 7, 5),
                                 np.linspace(-1, 1, 10))
    frame = FrameSpace(dict(x=radius.flatten(),
                            z=height.flatten(), segment='cylinder'))

    cylinder = BiotCylinder(frame, frame)
    print(cylinder._Cphi(0, np.pi).compute())
