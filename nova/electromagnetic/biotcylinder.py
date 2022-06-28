"""Biot-Savart calculation for complete circular cylinders."""
from dataclasses import dataclass, field
from functools import cached_property
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
        self.constant = [[] for _ in range(4)]
        for i, (dx, dz) in enumerate(zip([-0.5, 0.5, 0.5, -0.5],
                                         [-0.5, -0.5, 0.5, 0.5])):
            self.constant[i] = BiotConstants(
                self.data['rs'] + dx*self.data['dx'],
                self.data['zs'] + dz*self.data['dz'],
                self.data['r'], self.data['z'])

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
        print('attr', attr)
        return self.constant[self.corner][attr]

    def Aphi_hat(self, i: int):
        """Return Aphi intergration coefficient."""
        self.corner = i

        rs = self.constant[i]['r'] # TODO fix this ****
        return self.Cphi(np.pi/2) + \
            self.gamma*self.r*self.zeta(np.pi/2) + \
            self.gamma*self.a / (6*self.r) * \
            (self.U*self.K - 2*self.rs*self.E) + \
            self.gamma / (6*self.a*self.r) * \
            da.sum(da.stack([(-1)**p * self.Pphi(p) * self.Pi(p) for
                             p in range(1, 4)]), axis=0)

    @cached_property
    def Aphi(self):
        """Return Aphi dask array."""
        return 1 / (4*np.pi) * ((self.Aphi_hat(2) - self.Aphi_hat(1)) -
                                (self.Aphi_hat(3) - self.Aphi_hat(0)))

    @property
    def Psi(self):
        """Return Psi dask array."""
        print('data', self.data['r'])
        print('data2', self['r'])
        print('data3', self.r)
        return 2 * np.pi * self.mu_o * self.data['r'] * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return da.zeros_like(self['r'])

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return da.zeros_like(self['r'])


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-1, dplasma=-150)
    coilset.coil.insert(5, 0.5, 0.4, 0.8, section='r', turn='r',
                        nturn=300, segment='cylinder', name='CS1')
    coilset.saloc['Ic'] = 5e3

    coilset.grid.solve(1000, limit=[4.5, 6, 0, 1])
    coilset.grid.plot()
    coilset.plot()
    '''
    coilset = CoilSet(dcoil=-50, dplasma=-150)
    coilset.coil.insert(5, 0.5, 0.4, 0.4, section='r', turn='r',
                        nturn=300, segment='ring')
    coilset.saloc['Ic'] = 5e3

    coilset.grid.solve(1000, limit=[4.5, 6, 0, 1])
    coilset.grid.plot()
    coilset.plot()
    '''




    '''
    from nova.electromagnetic.framespace import FrameSpace
    radius, height = np.meshgrid(np.linspace(4, 7, 5),
                                 np.linspace(-1, 1, 10))
    frame = FrameSpace(dict(x=radius.flatten(),
                            z=height.flatten(), segment='cylinder'))

    cylinder = BiotCylinder(frame, frame)

    print(cylinder.Aphi_hat(0).compute())
    '''
