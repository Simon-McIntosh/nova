from scipy.special import ellipk, ellipe
from nova.inductance.geometric_mean_radius import geometric_mean_radius
from amigo.geom import shape, grid
import numpy as np
from nep.coil_geom import PFgeom
import pandas as pd


class biot_savart:

    def __init__(self, coilset):
        self.gmr = geometric_mean_radius()  # mutual gmr factors
        self.load_coilset(coilset)

    def load_coilset(self, coilset):
        self.coilset = coilset
        self.gmr.calculate_self(self.coilset.subcoil)
        self.coil = {}
        for key in ['x', 'z', 'dx', 'dz']:
            self.coil[key] = self.coilset.subcoil.loc[:, key].values
        self.coil['ro'] = self.gmr.ro.values
        for key in self.coil:
            self.coil[key] = self.coil[key].reshape(1, -1)
        self.nC = self.coilset.subcoil.nC

    def load_targets(self, x, z, Nt=1):
        self.shp = shape(x)  # store input shape of requested targets
        self.target = {}
        for key, value in zip(['x', 'z'], [x, z]):
            self.target[key] = self.shp.shape(value, (-1, 1))
        self.target['Nt']
        self.nP = len(self.target['x'])
        self.assemble()

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
            self.coil_m[key] = np.dot(np.ones((self.nP, 1)), self.coil[key])
        self.offset()

    def grid(self, n, limit):
        '''
        specify evaluation points as 2D grid
        '''
        x, z = grid(n, limit)[:2]
        self.load_targets(x, z)

    def colocate(self):
        self.load_targets(self.coil['x'], self.coil['z'])

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
        nx = abs(self.dR / self.coil_m['dx'])
        nz = abs(self.dR / self.coil_m['dz'])
        mutual_factor = self.gmr.evaluate(nx, nz)
        mutual_adjust = (mutual_factor - 1) / 2
        for i, key in enumerate(['x', 'z']):
            self.target_m[key][~idx] -= mutual_adjust[~idx] * self.dL[i][~idx]
            self.coil_m[key][~idx] += mutual_adjust[~idx] * self.dL[i][~idx]
        # self inductance
        self.target_m['x'][idx] = self.Xo[0][idx] - self.coil_m['ro'][idx] / 2
        self.coil_m['x'][idx] = self.Xo[0][idx] + self.coil_m['ro'][idx] / 2

    def calculate_inductance_matrix(self):
        self.colocate()
        xt, zt = self.target_m['x'], self.target_m['z']
        xc, zc = self.coil_m['x'], self.coil_m['z']
        m = 4 * xt * xc / ((xt + xc)**2 + (zt - zc)**2)
        M = np.array((xt * xc)**0.5 * ((2 * m**-0.5 - m**0.5) *
                     ellipk(m) - 2 * m**-0.5 * ellipe(m)))

        M = pd.DataFrame(M, index=self.coilset.subcoil.index,
                         columns=self.coilset.subcoil.index)
        print(M)

if __name__ is '__main__':

    cc = PFgeom(VS=False, dCoil=0.5).cc
    bs = biot_savart(cc.coilset)

    #bs.grid(1e3, [3, 6, -3, 3])

    bs.calculate_inductance_matrix()

    cc.plot()
