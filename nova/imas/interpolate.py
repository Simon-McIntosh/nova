"""Interpolate equilibria within separatrix."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import mu_0


from nova.imas.machine import Machine
from nova.imas.profile import Profile


@dataclass
class Interpolate(Machine, Profile):
    """Extract coil and plasma currents from ids and apply to CoilSet."""

    filename: str = field(default='interpolate')
    pf_passive: bool = False

    def itime_update(self):
        """Extend itime update."""
        super().itime_update()
        self.ionize()

    def ionize(self):
        """Ionize plasma filaments and set turn number."""
        self.plasma.separatrix = self.boundary
        self.sloc['plasma', 'Ic'] = self.get['ip']
        ionize = self.aloc['ionize']
        radius = self.aloc['x'][ionize]
        height = self.aloc['z'][ionize]
        psi = self.psi_rbs(radius, height)
        psi_norm = self.normalize(psi)
        current_density = radius * self.p_prime(psi_norm) + \
            self.ff_prime(psi_norm) / (mu_0 * radius)
        current_density *= -2*np.pi
        current = current_density * self.aloc['area'][ionize]
        self.aloc['nturn'][ionize] = current / current.sum()


if __name__ == '__main__':

    interpolate = Interpolate(105028, 1)
