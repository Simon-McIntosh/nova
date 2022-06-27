"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass, field
from typing import ClassVar

import dask.array as da
import numpy as np

from nova.electromagnetic.biotconstants import BiotConstants
from nova.electromagnetic.biotmatrix import BiotMatrix


@dataclass
class BiotCylinder(BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    _corner: int = field(init=False, default=0, repr=False)

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
        return self.const[self.corner][attr]

    def Aphi_hat(self, i: int):
        """Return Aphi intergration coefficient."""
        self.corner = i
        return self.Cphi(np.pi/2) + self.gamma*self.r*self.zeta(np.pi/2) + \
            self.gamma*self.a / (6*self.r) * \
            (self.U*self.K - 2*self.rs*self.E) + \
            self.gamma / (6*self.a*self.r) * \
            da.sum(da.stack([(-1)**p * self.Pphi(p) * self.Pi(p) for
                             p in range(1, 4)]), axis=0)

    '''
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
    '''


if __name__ == '__main__':

    from nova.electromagnetic.framespace import FrameSpace

    radius, height = np.meshgrid(np.linspace(4, 7, 5),
                                 np.linspace(-1, 1, 10))
    frame = FrameSpace(dict(x=radius.flatten(),
                            z=height.flatten(), segment='cylinder'))

    cylinder = BiotCylinder(frame, frame)

    print(cylinder.Aphi_hat(0).compute())
