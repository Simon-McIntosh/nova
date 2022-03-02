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


class SingleTurn:
    """Self-inductance methods for a single turn circular coil."""

    def __init__(self, x, cross_section='circle'):
        self.x = x  # coil major radius
        self.cross_section = cross_section  # coil cross_section
        self.cross_section_factor = \
            geometric_mean_radius.gmr_factor[self.cross_section]

    def minor_radius(self, L, bounds=(0, 1)):
        '''
        inverse method, solve coil minor radius for given inductance

        Attributes:
            L (float): target inductance Wb
            bounds (tuple of floats): bounds fraction of major radius

        Returns:
            dr (float): coil minor radius
        '''
        self.Lo = L
        r = minimize_scalar(self.flux_err, method='bounded',
                            bounds=bounds, args=(self.Lo),
                            options={'xatol': 1e-12}).x
        gmr = self.x * r
        dr = gmr / self.cross_section_factor
        return dr

    def flux_err(self, r, *args):
        gmr = r * self.x
        L_target = args[0]
        L = self.flux(gmr)
        return (L-L_target)**2

    def flux(self, gmr):
        '''
        calculate self-induced flux though a single-turn coil

        Attributes:
            a (float): coil major radius
            gmr (float): coil cross-section geometric mean radius

        Retuns:
            L (float): self inductance of coil
        '''
        if self.x > 0:
            L = self.x * ((1 + 3 * gmr**2 / (16 * self.x**2)) *
                          np.log(8 * self.x / gmr) -
                          (2 + gmr**2 / (16 * self.x**2)))
        else:
            L = 0
        return biot_savart.mu_o * L  # Wb

