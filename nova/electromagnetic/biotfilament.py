"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import ClassVar

import dask.array as da
from dask.cache import Cache
import numpy as np
import scipy.special

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotbase import BiotBase


# pylint: disable=no-member  # disable scipy.special module not found


@dataclass
class BiotFilament(BiotBase):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name = 'filament'  # element name
    attrs: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])
    _attrs: ClassVar[list[str]] = ['Aphi', 'Psi', 'Br', 'Bz']

    def calculate_coefficients(self):
        """Return interaction coefficients."""
        self.coef = Coefficients(self.source('rms'), self.source('z'),
                                 self.target('x'), self.target('z'))

    @cached_property
    def Aphi(self):
        """Return Aphi dask array."""
        return 1 / (2*np.pi) * self.coef.a / self.coef.target_radius * \
            ((1 - self.coef.k2/2) * self.coef.K - self.coef.E)

    @property
    def Psi(self):
        """Return Psi dask array."""
        return 2 * np.pi * self.mu_o * self.coef.target_radius * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return self.mu_o / (2*np.pi) * self.coef.gamma * \
            (self.coef.K - (2-self.coef.k2) / (2*self.coef.ck2) *
             self.coef.E) / (self.coef.a * self.coef.target_radius)

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return self.mu_o / (2*np.pi) * \
            (self.coef.target_radius*self.coef.K -
             (2*self.coef.target_radius - self.coef.b*self.coef.k2) /
             (2*self.coef.ck2) * self.coef.E) / \
            (self.coef.a*self.coef.target_radius)

    def calculate_vector_potential(self):
        """Calculate target vector potential (r, phi, z), Wb/Amp-turn-turn."""
        self.vector['Aphi'] = self.Aphi.compute()

    def calculate_scalar_potential(self):
        """Calculate scalar potential."""
        self.vector['Psi'] = self.Psi.compute()

    def calculate_magnetic_field(self):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Br'] = self.Br.compute()
        self.vector['Bz'] = self.Bz.compute()


if __name__ == '__main__':

    from nova.electromagnetic.framespace import FrameSpace

    radius, height = np.meshgrid(np.linspace(4, 7, 100),
                                 np.linspace(-1, 1, 10))
    frame = FrameSpace(dict(x=radius.flatten(),
                            z=height.flatten(), segment='ring'))
    source = BiotFrame(frame)
    target = BiotFrame(frame)
    source.set_target(len(target))
    target.set_source(len(source))

    coef = Coefficients(source('x'), source('z'), target('x'), target('z'))

    print(coef.E.shape)

    filament = BiotFilament(source, target)
