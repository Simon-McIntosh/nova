import numpy as np
import scipy.interpolate
from scipy.special import ellipk, ellipe

mu_o = 4e-7*np.pi  # magnetic constant [Vs/Am]


class geometric_mean_radius:
    # On the geometrical mean distances of rectangular areas and the
    # calculation of self-inductance, E. B. Rosa

    def __init__(self):
        self.construct()

    def construct(self):
        nx, nz = np.arange(0, 6), np.arange(0, 6)
        c = np.zeros((6, 6))
        c[:, 0] = [0, 6528.5, 510.4, 102.5, 32.5, 13.6]
        c[:, 1] = [6528.5, -2301.9, -89.5, 23.6, 16.0, 8.4]
        c[:, 2] = [510.4, -89.5, -130.8, -34.8, -5.8, 0.7]
        c[:, 3] = [102.5, 23.6, -34.8, -25.6, -11.3, -4.0]
        c[:, 4] = [32.5, 16.0, -5.8, -11.3, -8.1, -4.5]
        c[:, 5] = [13.6, 8.4, 0.7, -4.0, -4.5, -3.1]
        ratio = np.exp(1e-6*c)  # ratio of gmr / r(centroid)
        self.ratio = scipy.interpolate.RectBivariateSpline(nx, nz, ratio)

    def evaluate(self, nx, nz):  # evaluate gmr/r ratio
        r = self.ratio.ev(nx, nz)
        index = (np.array(nx) > 5) | (np.array(nz) > 5)
        r[index] = 1
        return r


class Filament:
    """Compute interaction using complete circular filaments."""

    _cross_section = 'filament'  # applicable cross section type

    def __init__(self, source, target):
        self.initialize_filaments(source, target)
        self.offset_filaments(source)
        self.calculate_coefficients()

    def initialize_filaments(self, source, target):
        self.rs, self.zs = source._rms_, source._z_  # source
        self.r, self.z = target._x_, target._z_  # target

    def offset_filaments(self, source, n_fold=0, n_merge=1,
                         rms_offset=True):
        """
        Offset source and target filaments.

        Parameters
        ----------
        source : nova.BiotFrame
            Source filament biotframe.
        n_fold : float, optional
            Number of e-foling lenghts within filament. The default is 1.
        n_merge : float, optional
            Merge radius, multiple of filament widths. The default is 1.25.
        rms_offset : bool, optional
            Maintain rms offset for filament pairs. The default is False.

        Returns
        -------
        None.

        """
        # extract interaction
        df = np.max([source._dx_, source._dz_], axis=0) / 2
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
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

    def _offset_filaments(self):
        'offset source and target points'
        # point seperation
        dL = np.array([(self.r-self.rs), (self.z-self.zs)])
        dL_mag = np.linalg.norm(dL, axis=0)
        dr = self.dl/2  # filament characteristic radius
        ro = dr*self.cross_section_factor  # self seperation

        # zero-seperation
        index = np.isclose(dL_mag, 0)
        dL_norm = np.zeros((2, self.nI))
        dL_norm[0, index] = 1  # radial offset
        dL_norm[:, ~index] = dL[:, ~index] / dL_mag[~index]
        # initalize offsets
        dr, dz = np.zeros(self.nI), np.zeros(self.nI)

        # mutual offset
        nx = dL[0] / self.drs
        nz = dL[1] / self.dzs
        mutual_index = np.where((nx <= 5) & (nz <= 5))  # mutual index
        mutual_factor = self.gmr.evaluate(nx[mutual_index], nz[mutual_index])
        dr[mutual_index] = (mutual_factor-1) * dL[0, mutual_index]
        dz[mutual_index] = (mutual_factor-1) * dL[1, mutual_index]

        # self inductance index
        self_index = np.where(dL_mag <= ro)  # seperation < dl/2
        #self_dr = self.dl[self_index]/2  # filament characteristic radius
        #self_ro = self_dr*self.cross_section_factor[self_index]  # seperation
        self_ro = ro[self_index]
        self_factor = 1 - dL_mag[self_index]/self_ro
        dr[self_index] = self_factor*self_ro*dL_norm[0, self_index]  # radial
        dz[self_index] = self_factor*self_ro*dL_norm[1, self_index]  # vertical

        # rms offset
        drms = -(self.r+self.rs)/4 + np.sqrt((self.r+self.rs)**2 -
                                             8*dr*(self.r - self.rs + 2*dr))/4
        self.rs += drms
        self.r += drms
        # offset source filaments
        self.rs -= dr/2
        self.zs -= dz/2
        # offset target filaments
        self.r += dr/2
        self.z += dz/2

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

