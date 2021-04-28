"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass

import numpy as np

from nova.electromagnetic.biotsavart import BiotSavart
from nova.electromagnetic.biotframe import BiotFrame


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
class PoloidalOffset(PolidalCoordinates):
    """Offset source and target filaments."""

    fold_number: int = 0  # Number of e-foling lenghts within filament
    merge_number: int = 1  # Merge radius, multiple of filament widths
    rms_offset: bool = True  # Maintain rms offset for filament pairs

    def __post_init__(self):
        """Calculate source turn effective radius and source-target span."""
        self.turn_radius = np.max([self.source('dx'),
                                   self.source('dz')], axis=0) / 2  # df

        self.calculate_seperation()

    def calculate_seperation(self):
        """Calculate source-target seperation."""
        self.span = np.array([(self.target_radius-self.source_radius),
                              (self.target_height-self.source_height)])  # dL
        self.span_norm = np.linalg.norm(self.span, axis=0)  # dL_mag

    def calculate(self):
        # extract interaction


        # select filaments within merge radius
        idx = np.where(dL_mag <= df*n_merge)[0]

        # reduce
        dL_mag = dL_mag[idx]
        dL = dL[:, idx]
        df = df[idx]
        ro = source._dx_[idx]*source._cs_factor_[idx]/2  # self seperation

        # interacton orientation
        index = np.isclose(dL_mag, 0)
        dL_norm = np.zeros((2, len(index)))
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]

        if n_fold == 0:
            factor = (1 - dL_mag / (df*n_merge))  # linear blending
        else:
            factor = np.exp(-n_fold*(dL_mag/df)**2)  # exponential blending

        dr = factor*ro*dL_norm[0, :]  # radial offset
        dz = factor*ro*dL_norm[1, :]  # vertical offset

        if rms_offset:
            drms = -(self.r[idx]+self.rs[idx])/4 + np.sqrt(
                (self.r[idx]+self.rs[idx])**2 -
                8*dr*(self.r[idx] - self.rs[idx] + 2*dr))/4
            self.rs[idx] += drms
            self.r[idx] += drms
        # offset source filaments
        self.rs[idx] -= dr/2
        self.zs[idx] -= dz/2
        # offset target filaments
        self.r[idx] += dr/2
        self.z[idx] += dz/2


@dataclass
class BiotCircle(BiotSavart):
    """
    Extend BiotSavart.

    Compute interaction for complete circular filaments.

    """

    name = 'circle'  # applicable cross section type


    def __post_init__(self):
        super().__post_init__()


        #self.initialize_filaments(source, target)
        #self.offset_filaments(source)
        #self.calculate_coefficients()

    def calculate(self):
        pass

    def initialize_filaments(self, source, target):
        self.rs, self.zs = source._rms_, source._z_  # source
        self.r, self.z = target._x_, target._z_  # target


    def calculate_coefficients(self):
        self.b = self.rs + self.r
        self.gamma = self.zs - self.z
        self.a2 = self.gamma**2 + (self.r + self.rs)**2
        self.a = np.sqrt(self.a2)
        self.k2 = 4 * self.r * self.rs / self.a2
        self.ck2 = 1 - self.k2  # complementary modulus
        self.K = ellipk(self.k2)  # first complete elliptic integral
        self.E = ellipe(self.k2)  # second complete elliptic integral

    def scalar_potential(self):
        'vector and scalar potential'
        Aphi = 1 / (2*np.pi) * self.a/self.r * \
            ((1 - self.k2/2) * self.K - self.E)  # Wb/Amp-turn-turn
        psi = 2 * np.pi * mu_o * self.r * Aphi  # scalar potential
        return psi

    def radial_field(self):
        Br = mu_o / (2*np.pi) * self.gamma * (
            self.K - (2-self.k2) / (2*self.ck2) * self.E) / (self.a*self.r)
        return Br  # T / Amp-turn-turn

    def vertical_field(self):  # T / Amp-turn-turn
        Bz = mu_o / (2*np.pi) * (self.r*self.K - \
            (2*self.r - self.b*self.k2) /
            (2*self.ck2) * self.E) / (self.a*self.r)
        return Bz  # T / Amp-turn-turn


if __name__ == '__main__':

    source = {'x': [3, 3.4, 3.6], 'z': [3.1, 3, 3.3],
          'dl': 0.3, 'dt': 0.3, 'section': 'hex'}
    biotfilament = BiotFilament(source, source, update=['ps'])
