import numpy as np
from scipy.interpolate import RectBivariateSpline
from pandas.api.types import is_list_like
from collections.abc import Iterable


class geometric_mean_radius:
    # On the geometrical mean distances of rectangular areas and the
    # calculation of self-inductance, E. B. Rosa

    gmr_factor = {'circle': np.exp(-0.25),  # circle-circle
                  'square': 2*0.447049,  # square-square
                  'skin': 1}  # skin-skin

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
        self.ratio = RectBivariateSpline(nx, nz, ratio)

    def evaluate(self, nx, nz):  # evaluate gmr/r ratio
        r = self.ratio.ev(nx, nz)
        index = (np.array(nx) > 5) | (np.array(nz) > 5)
        r[index] = 1
        return r

    def calculate_self(self, dx, dz, cross_section='circle'):
        ''' calculate self geometric mean radius '''
        ro = np.array([dx, dz]).mean(axis=0) / 2
        ro *= self.cross_section_factor(cross_section)
        return ro

    def cross_section_factor(self, cross_section):
        if not is_list_like(cross_section):
            cross_section = [cross_section]
        cross_section = np.array(cross_section)
        factor = np.zeros(len(cross_section))
        factor[cross_section == 'circle'] = self.gmr_factor['circle']
        factor[cross_section == 'square'] = self.gmr_factor['square']
        factor[cross_section == 'skin'] = self.gmr_factor['skin']
        factor[cross_section == 'polygon'] = self.gmr_factor['square']
        if np.array(factor == 0).any():
            cs_isnull = np.unique(cross_section[factor == 0])
            raise IndexError(f'cross section: {cs_isnull} not defined')
        return factor
    
    @staticmethod
    def gmr(x, dx, turn_section='circle'):
        '''
        Attributes:
            x (float or itterable): geometric coil center
            dx (float): coil radial width
        '''
        ro = dx * geometric_mean_radius.gmr_factor[turn_section]
        if not isinstance(x, Iterable):
            x = [x]
        _gmr = np.zeros(len(x))
        for i, x_ in enumerate(x):
            _gmr[i] = np.sqrt(ro**2 + 4 * x_**2) / 2 
        return _gmr


def gmr_offset(x_gmr, *args):
    x, dx = args  # coil geometry
    g = np.exp(1/dx * ((x_gmr + dx/2) * np.log(x_gmr + dx/2)
                       - dx - (x_gmr - dx/2) * np.log(x_gmr - dx/2)))
    offset = np.abs(g - x)
    return offset


if __name__ == '__main__':
    gmr = geometric_mean_radius()
    
    print(gmr.calculate_self([1, 2, 3], [2, 2, 2],
                             ['circle', 'square', 'circle']))
    print(gmr.evaluate([0, 1], [0, 0]))
    print(gmr.gmr(3, 1.5))

