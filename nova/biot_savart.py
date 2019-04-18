from scipy.special import ellipk, ellipe
from nova.inductance.geometric_mean_radius import geometric_mean_radius
from amigo.geom import shape, grid
import numpy as np
import pandas as pd


class biot_savart:

    mu_o = 4 * np.pi * 1e-7  # magnetic constant [Vs/Am]

    def __init__(self, coilset=None, **kwargs):
        self.gmr = geometric_mean_radius()  # mutual gmr factors
        self.mutual = kwargs.pop('mutual', False)  # mutual inductance offset
        if coilset is not None:
            self.load_coilset(coilset)

    def load_coilset(self, coilset):
        self.coilset = coilset
        self.gmr.calculate_self(self.coilset.subcoil)
        self.coil = {}
        for key in ['x', 'z', 'dx', 'dz', 'Nt']:
            self.coil[key] = self.coilset.subcoil.loc[:, key].values
        self.coil['ro'] = self.gmr.ro.values
        for key in self.coil:
            self.coil[key] = self.coil[key].reshape(1, -1)
        self.nC = self.coilset.subcoil.nC

    def load_target(self, x, z, Nt=None):
        self.tshp = shape(x)  # store input shape of requested targets
        if Nt is None:
            Nt = np.ones((self.tshp.input_shape))
        self.target = {}
        for key, value in zip(['x', 'z', 'Nt'], [x, z, Nt]):
            self.target[key] = self.tshp.shape(value, (-1, 1))
        self.nT = len(self.target['x'])

    def assemble(self):
        if not hasattr(self, 'target'):
            raise IndexError('points undefined: load_points')
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

    def colocate(self):
        self.load_target(self.coil['x'], self.coil['z'], self.coil['Nt'])
        self.assemble()

    def offset(self):
        '''
        transform turn-trun offset to geometric mean
        '''
        self.dL = np.array([self.target_m['x'] - self.coil_m['x'],
                            self.target_m['z'] - self.coil_m['z']])
        self.Xo = np.array([self.coil_m['x'], self.coil_m['z']]) + self.dL / 2
        self.dR = np.linalg.norm(self.dL, axis=0)
        # self inductance index
        idx = self.dR < self.coil['ro']  # seperation < self-self gmr
        # mutual inductance
        if self.mutual:  # mutual inductance offset
            nx = abs(self.dR / self.coil_m['dx'])
            nz = abs(self.dR / self.coil_m['dz'])
            mutual_factor = self.gmr.evaluate(nx, nz)
            mutual_adjust = (mutual_factor - 1) / 2
            for i, key in enumerate(['x', 'z']):
                self.target_m[key][~idx] -= \
                    mutual_adjust[~idx] * self.dL[i][~idx]
                self.coil_m[key][~idx] += \
                    mutual_adjust[~idx] * self.dL[i][~idx]
        # self inductance offset
        self.target_m['x'][idx] = self.Xo[0][idx] - self.coil_m['ro'][idx] / 2
        self.coil_m['x'][idx] = self.Xo[0][idx] + self.coil_m['ro'][idx] / 2

    def locate(self):
        xt, zt = self.target_m['x'], self.target_m['z']
        Nt = self.target_m['Nt']
        xc, zc = self.coil_m['x'], self.coil_m['z']
        Nc = self.coil_m['Nt']
        return xt, zt, Nt, xc, zc, Nc

    def flux(self):
        xt, zt, Nt, xc, zc, Nc = self.locate()
        m = 4 * xt * xc / ((xt + xc)**2 + (zt - zc)**2)
        M = np.array((xt * xc)**0.5 * ((2 * m**-0.5 - m**0.5) *
                     ellipk(m) - 2 * m**-0.5 * ellipe(m)))
        M *= Nt * Nc  # turn-turn interaction, line-current
        return self.mu_o * M  # Wb

    def inductance(self):
        self.colocate()  # set targets
        Msub = self.flux()  # calculate subcoil inductance matrix
        Msub = pd.DataFrame(Msub, index=self.coilset.subcoil.index,
                            columns=self.coilset.subcoil.index)
        Mrow = pd.DataFrame(index=self.coilset.coil.index,
                            columns=self.coilset.subcoil.index)
        Mc = pd.DataFrame(index=self.coilset.coil.index,
                          columns=self.coilset.coil.index)
        for name in self.coilset.coil.index:  # row reduction
            index = self.coilset.coil.subindex[name]
            Mrow.loc[name, :] = Msub.loc[index, :].sum(axis=0)
        for name in self.coilset.coil.index:  # column reduction
            index = self.coilset.coil.subindex[name]
            Mc.loc[:, name] = Mrow.loc[:, index].sum(axis=1)
        Nt = self.coilset.coil.Nt.values
        Nt = Nt.reshape(-1, 1) * Nt.reshape(1, -1)
        self.coilset.matrix['inductance']['Mc'] = Mc  # line-current
        self.coilset.matrix['inductance']['Mt'] = Mc / Nt  # amp-turn

    def field(self):
        field = np.zeros((2, self.nT, self.nC))
        xt, zt, __, xc, zc, Nc = self.locate()
        a = np.sqrt((xt + xc)**2 + (zt - zc)**2)
        m = 4 * xt * xc / a**2
        I1 = 4 / a * ellipk(m)
        I2 = 4 / a**3 * ellipe(m) / (1 - m)
        A = (zt - zc)**2 + xt**2 + xc**2
        B = -2 * xt * xc
        field[0] = xc / (2 * np.pi) * (zt - zc) / B * (I1 - A * I2)
        field[1] = xc / (2 * np.pi) * ((xc + xt * A / B) * I2 - xt / B * I1)
        field *= Nc  # line-current
        return self.mu_o * field  # T


if __name__ is '__main__':

    from nova.coil_class import CoilClass  # avoid cyclic import

    cc = CoilClass(dCoil=-1)
    #cc.add_coil(3.943, 7.564, 0.959, 0.984, Nt=248.64, name='PF1', part='PF')
    cc.add_coil(1.722, 5.313, 0.719, 2.2, Nt=554, name='CS3U', part='CS',
                cross_section='square')

    #cc.add_coil(1.722, 3.188, 0.719, 2.075, Nt=554, name='CS2U', part='CS')
    cc.plot()

    biot_savart(cc.coilset, mutual=False).inductance()

    plt.title(cc.coilset.matrix['inductance']['Mc'].CS3U)