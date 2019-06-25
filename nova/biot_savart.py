from scipy.special import ellipk, ellipe
from nova.inductance.geometric_mean_radius import geometric_mean_radius
from amigo.geom import shape, grid
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


class biot_savart:

    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]

    def __init__(self, coilset=None, **kwargs):
        self.gmr = geometric_mean_radius()  # mutual gmr factors
        self.mutual = kwargs.pop('mutual', False)  # mutual inductance offset
        self.load_coilset(coilset)

    @staticmethod
    def gmd(x, Nt):
        '''
        geometric mean distance
        '''
        return np.exp(np.sum(abs(Nt) * np.log(x)) / np.sum(abs(Nt)))

    @staticmethod
    def amd(x, Nt):
        '''
        arithmetic mean distance
        '''
        return np.sum(abs(Nt) * x) / np.sum(abs(Nt))

    def load_coilset(self, coilset):
        self.coil_index = coilset.coil.index
        self.subcoil_index = coilset.subcoil.index
        self.subindex = coilset.coil['subindex']
        self.nC = coilset.subcoil.nC
        self.coil = coilset.subcoil.loc[:, ['x', 'z', 'dx', 'dz', 'Nt']]
        self.coil['ro'] = self.gmr.calculate_self(coilset.subcoil)
        self.coil = self.coil.to_dict('list')
        self.reshape_coil()

    def reshape_coil(self):
        for key in self.coil:
            self.coil[key] = np.array(self.coil[key]).reshape(1, -1)

    def load_target(self, x, z, Nt=None, index=None):
        tshp = shape(x)  # store input shape of requested targets
        if Nt is None:
            Nt = np.ones((tshp.input_shape))
        self.target_index = index
        self.target = {}
        for key, value in zip(['x', 'z', 'Nt'], [x, z, Nt]):
            self.target[key] = tshp.shape(value, (-1, 1))
        self.nT = len(self.target['x'])

    def assemble(self):
        if not hasattr(self, 'target'):
            raise IndexError('points undefined: load_target')
        self.target_m = {}
        for key in self.target:
            self.target_m[key] = np.dot(self.target[key],
                                        np.ones((1, self.nC)))
        if not hasattr(self, 'coil'):
            raise IndexError('coil undefined: load_coilset')
        self.coil_m = {}
        for key in self.coil:
            self.coil_m[key] = np.dot(np.ones((self.nT, 1)), self.coil[key])
        self.offset()

    def grid(self, n, limit):
        '''
        specify evaluation points as 2D grid
        '''
        x, z = grid(n, limit)[:2]
        self.load_targets(x, z)
        self.assemble()

    def colocate(self):
        self.load_target(self.coil['x'], self.coil['z'], self.coil['Nt'],
                         index=self.subcoil_index)
        self.assemble()

    def offset(self):
        '''
        transform turn-trun offset to geometric mean
        '''
        self.dL = np.array([self.target_m['x'] - self.coil_m['x'],
                            self.target_m['z'] - self.coil_m['z']])
        self.Ro = np.exp((np.log(self.coil_m['x']) +
                          np.log(self.target_m['x'])) / 2)
        self.dL_mag = np.linalg.norm(self.dL, axis=0)
        iszero = np.isclose(self.dL_mag, 0)  # self index
        self.dL_norm = np.zeros((2, self.nT, self.nC))
        self.dL_norm[:, ~iszero] = self.dL[:, ~iszero] / self.dL_mag[~iszero]
        self.dL_norm[1, iszero] = 1
        # self inductance index
        dr = (self.coil_m['dx'] + self.coil_m['dz']) / 4  # mean turn radius
        idx = self.dL_mag < dr  # seperation < mean radius
        # mutual inductance
        if self.mutual:  # mutual inductance offset
            nx = abs(self.dL_mag / self.coil_m['dx'])
            nz = abs(self.dL_mag / self.coil_m['dz'])
            mutual_factor = self.gmr.evaluate(nx, nz)
            mutual_adjust = (mutual_factor - 1) / 2
            for i, key in enumerate(['x', 'z']):
                offset = mutual_adjust[~idx] * self.dL[i][~idx]
                self._apply_offset(key, offset, ~idx)
        # self-inductance offset
        factor = (1 - self.dL_mag[idx] / dr[idx]) / 2
        for i, key in enumerate(['x', 'z']):
            offset = factor * self.coil_m['ro'][idx] * self.dL_norm[i][idx]
            self._apply_offset(key, offset, idx)

    def _apply_offset(self, key, offset, index):
        if key == 'x':
            Ro_offset = np.exp(
                    (np.log(self.coil_m[key][index] - offset) +
                     np.log(self.target_m[key][index] + offset)) / 2)
            shift = self.Ro[index] - Ro_offset  # gmr shift
        else:
            shift = np.zeros(np.shape(offset))
        self.coil_m[key][index] -= offset - shift
        self.target_m[key][index] += offset + shift
        return shift

    def locate(self):
        xt, zt = self.target_m['x'], self.target_m['z']
        Nt = self.target_m['Nt']
        xc, zc = self.coil_m['x'], self.coil_m['z']
        Nc = self.coil_m['Nt']
        return xt, zt, Nt, xc, zc, Nc

    def flux_matrix(self):
        '''
        calculate subcoil flux (inductance) matrix
        '''
        xt, zt, Nt, xc, zc, Nc = self.locate()
        m = 4 * xt * xc / ((xt + xc)**2 + (zt - zc)**2)
        M = np.array((xt * xc)**0.5 * ((2 * m**-0.5 - m**0.5) *
                     ellipk(m) - 2 * m**-0.5 * ellipe(m)))
        M *= Nt * Nc  # turn-turn interaction, line-current
        return self.mu_o * M  # Wb

    def field_matrix(self):
        '''
        calculate subcoil field matrix
        '''
        field = np.zeros((2, self.nT, self.nC))
        xt, zt, Nt, xc, zc, Nc = self.locate()
        a = np.sqrt((xt + xc)**2 + (zt - zc)**2)
        m = 4 * xt * xc / a**2
        I1 = 4 / a * ellipk(m)
        I2 = 4 / a**3 * ellipe(m) / (1 - m)
        A = (zt - zc)**2 + xt**2 + xc**2
        B = -2 * xt * xc
        # xc / (2 * np.pi)
        field[0] = xc / 2 * (zt - zc) / B * (I1 - A * I2)
        field[1] = xc / 2 * ((xc + xt * A / B) * I2 - xt / B * I1)
        field *= Nt * Nc  # line-current
        return self.mu_o * field  # T

    def solve(self, field=False):
        Mo = self.flux_matrix()
        M = self.reduce(Mo)
        if field:
            Bo = self.field_matrix()
            Bx = self.reduce(Bo[0])
            Bz = self.reduce(Bo[1])
            return M, Bx, Bz
        else:
            return M

    def reduce(self, Mo):
        Mo = pd.DataFrame(Mo, index=self.target_index,
                          columns=self.subcoil_index, dtype=float)
        Mcol = pd.DataFrame(index=self.target_index,
                            columns=self.coil_index, dtype=float)
        M = pd.DataFrame(columns=self.coil_index, dtype=float)
        for name in self.coil_index:  # column reduction
            index = self.subindex[name]
            Mcol.loc[:, name] = Mo.loc[:, index].sum(axis=1)
        if self.target_index is not None:
            for name in self.coil_index:  # row reduction
                index = self.subindex[name]
                M.loc[name, :] = Mcol.loc[index, :].sum(axis=0)
        else:
            M = Mcol
        return M

    def calculate_inductance(self):
        self.colocate()  # set targets
        Mc = self.solve(field=False)  # line-current
        return Mc

    def calculate_interaction(self, coilset=None, grid=None):
        if coilset is not None:  # update coilset
            self.load_coilset(coilset)
        if grid is not None:  # update targets
            self.load_target(grid['x2d'], grid['z2d'])
        if coilset is not None or grid is not None:
            self.assemble()
        M, Bx, Bz = self.solve(field=True)  # line-current interaction
        return M, Bx, Bz


class self_inductance:
    '''
    self-inductance methods for single turn circular coil
    '''

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
        L = self.x * ((1 + 3 * gmr**2 / (16 * self.x**2)) *
                      np.log(8 * self.x / gmr) -
                      (2 + gmr**2 / (16 * self.x**2)))
        return biot_savart.mu_o * L  # Wb


if __name__ is '__main__':

    from nova.coil_class import CoilClass  # avoid cyclic import

    cc = CoilClass(dCoil=-1, turn_fraction=0.7)
    cc.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    cc.add_coil(1.6870, 5.4640, 0.7400, 2.093, Nt=554, name='CS3U', part='CS',
                cross_section='circle')
    cc.add_coil(1.6870, 3.2780, 0.7400, 2.093, Nt=554, name='CS2U', part='CS')
    cc.add_plasma(5, 2.5, 1.5, 1.5, It=5e6, cross_section='circle')
    cc.plot(label=True)

    bs = biot_savart(cc.coilset, mutual=False)
    Mc = bs.calculate_inductance()

    # plt.title(cc.coilset.matrix['inductance']['Mc'].CS3U)