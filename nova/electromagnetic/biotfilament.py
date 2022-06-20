"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import scipy.special

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotbase import BiotBase


# pylint: disable=no-member  # disable scipy.special module not found


@dataclass
class PolidalCoordinates:
    """Manage poloidal coordinates."""

    source: BiotFrame
    target: BiotFrame

    def __post_init__(self):
        """Extract source and target coordinates."""
        self.source_radius = self.source('rms')
        self.source_height = self.source('z')
        self.target_radius = self.target('x')
        self.target_height = self.target('z')


@dataclass
class BiotFilament(BiotBase):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name = 'ring'  # element name
    attrs: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])
    _attrs: ClassVar[list[str]] = ['Aphi', 'Psi', 'Br', 'Bz']

    def calculate_coefficients(self):
        """Return interaction coefficients."""
        pass

    '''
        offset = PoloidalOffset(self.source, self.target)
        coeff = {'rs': offset.source_radius, 'zs': offset.source_height,
                 'r': offset.target_radius, 'z': offset.target_height}
        coeff['b'] = coeff['rs'] + coeff['r']
        coeff['gamma'] = coeff['zs'] - coeff['z']
        coeff['a2'] = coeff['gamma']**2 + (coeff['r'] + coeff['rs'])**2
        coeff['a'] = np.sqrt(coeff['a2'])
        coeff['k2'] = 4 * coeff['r'] * coeff['rs'] / coeff['a2']
        coeff['ck2'] = 1 - coeff['k2']  # complementary modulus
        coeff['K'] = scipy.special.ellipk(coeff['k2'])  # ellip integral - 1st
        coeff['E'] = scipy.special.ellipe(coeff['k2'])  # ellip integral - 2nd
        return coeff
    '''

    def calculate_vector_potential(self, coeff):
        """Calculate target vector potential (r, phi, z), Wb/Amp-turn-turn."""
        self.vector['Aphi'] = 1 / (2*np.pi) * coeff['a']/coeff['r'] * \
            ((1 - coeff['k2']/2) * coeff['K'] - coeff['E'])

    def calculate_scalar_potential(self, coeff):
        """Calculate scalar potential."""
        self.vector['Psi'] = 2 * np.pi * self.mu_o * \
            coeff['r'] * self.vector['Aphi']

    def calculate_magnetic_field(self, coeff):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Br'] = self.mu_o / (2*np.pi) * \
            coeff['gamma'] * (coeff['K'] - (2-coeff['k2']) / (2*coeff['ck2']) *
                              coeff['E']) / (coeff['a'] * coeff['r'])
        self.vector['Bz'] = self.mu_o / (2*np.pi) * \
            (coeff['r']*coeff['K'] - (2*coeff['r'] - coeff['b']*coeff['k2']) /
             (2*coeff['ck2']) * coeff['E']) / (coeff['a']*coeff['r'])
