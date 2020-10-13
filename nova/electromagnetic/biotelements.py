import numpy as np
from scipy.special import ellipk, ellipe

mu_o = 4e-7*np.pi  # magnetic constant [Vs/Am]


class Filament:
    """Compute interaction using complete circular filaments."""

    _cross_section = 'filament'  # applicable cross section type

    def __init__(self, source, target):
        self.initialize_filaments(source, target)
        self.offset_filaments()
        self.calculate_coefficients()

    def initialize_filaments(self, source, target):
        self.rs, self.zs = source._rms_, source._z_  # source
        self.r, self.z = target._x_, target._z_  # target
        #self.dl = np.linalg.norm([source._dx_, source._dz_],
        #                         axis=0)  # filament characteristic length
        self.dl = np.max([source._dx_, source._dz_], axis=0)
        self.cross_section_factor = source._cs_factor_  # cross-section factor

    def offset_filaments(self):
        'offset source and target points'
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
        idx = np.where(dL_mag < self.dl/2)[0]  # seperation < dl/2
        # reduce
        dL = dL[:, idx]
        dL_mag = dL_mag[idx]
        dr = self.dl[idx]/2  # filament characteristic radius
        dL_norm = np.zeros((2, len(idx)))
        index = np.isclose(dL_mag, 0)
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]
        ro = dr*self.cross_section_factor[idx]  # self seperation
        factor = 1 - dL_mag/dr
        dr = factor*ro*dL_norm[0]  # radial offset
        dz = factor*ro*dL_norm[1]  # vertical offset
        # rms offset
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

