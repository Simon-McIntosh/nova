import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import matplotlib.animation as manimation
from amigo.time import clock
from scipy.interpolate import RegularGridInterpolator as RGI
from io import StringIO
from nep.ITERcage import ITERcage
from collections import OrderedDict
from nep.invessel_coils import centreline
from itertools import count
from nova.coils import PF
from amigo.IO import readtxt

'''
ivc = centreline()
for part in ivc.loops:
    for loop in ivc.loops[part]:
        p = ivc.loops[part][loop]['points']
        pl.plot(np.sqrt(p[:, 0]**2 + p[:, 1]**2), p[:, 2], '-')
'''

filename = 'C:/Users/mcintos/Downloads/\
tor_cur_data_MD_DW_exp16ms_2010_Int.thic_34HAEW_v1_0.dat'

#data = pd.read_csv(filename)


class read_dina:

    def __init__(self, filename):
        self.read_file(filename)

    def read_file(self, filename):
        with readtxt(filename) as self.rt:
            self.read_coils()
            self.read_frames()

    def read_coils(self):
        self.rt.skiplines(5)  # skip header
        self.get_coils()
        self.get_filaments()

    def read_frames(self):
        frames = []
        self.rt.skiplines(6)
        while True:
            try:
                frames.append(dina.get_current())
            except:
                break
        self.timeframes = frames

    def get_coils(self):
        self.coil, coilnames = OrderedDict(), []
        self.rt.checkline('1.')
        self.rt.skiplines(1)
        coilnames = self.rt.readline(True)
        coilnames.extend(self.rt.readline(True))
        for name in coilnames:
            self.coil[name] = {}  # initalise as empty dict
        for var in ['x', 'z', 'dx', 'dz']:
            self.rt.skiplines(1)
            self.fill_coil(var, self.rt.readblock())
        self.set_coil()

        PF.plot_coil(self.coil)
        pl.axis('equal')
        pl.axis('off')

    def fill_coil(self, key, values):  # set key/value pairs in coil dict
        for name, value in zip(self.coil, values):
            self.coil[name][key] = value

    def set_coil(self):
        for name in self.coil:
            for var in ['dx', 'dz', 'x', 'z']:
                self.coil[name][var] *= 1e-2  # cm to meters
            self.coil[name]['rc'] = np.sqrt(self.coil[name]['dx']**2 +
                                            self.coil[name]['dz']**2) / 4
            self.coil[name]['I'] = 0

    def get_filaments(self, plot=False):
        self.rt.checkline('2.')
        self.rt.skiplines(1)
        self.nf = self.rt.readnumber()
        self.rt.skiplines(3)
        self.filaments = np.zeros(self.nf,
                                  dtype=[('r1', '2float'), ('z1', '2float'),
                                         ('r2', '2float'), ('z2', '2float'),
                                         ('turn', '2float'),
                                         ('n_turn', 'int')])
        for i in range(self.nf):
            n_turn = self.rt.readnumber()
            self.filaments[i]['n_turn'] = n_turn
            for j in range(n_turn):
                r1, z1, r2, z2, turn = [float(d) for d in
                                        self.rt.readline(True)]
                for var, value in zip(['r1', 'z1', 'r2', 'z2', 'turn'],
                                      [r1, z1, r2, z2, turn]):
                    self.filaments[i][var][j] = value
        self.store_filaments()
        PF.plot_coil(self.blanket_coil)
        PF.plot_coil(self.vessel_coil)
        if plot:
            self.plot_filaments()

    def plot_filaments(self):
        for f in self.filaments:
            for j in range(f['n_turn']):
                pl.plot([f['r1'][j], f['r2'][j]],
                        [f['z1'][j], f['z2'][j]], 'o-')
        pl.axis('equal')

    def store_filaments(self, dx=0.25, dz=0.25):
        rc = np.sqrt(dx**2 + dz**2) / 4
        nvv, nbl = count(0), count(0)
        self.blanket_coil = OrderedDict()
        self.vessel_coil = OrderedDict()
        for i, filament in enumerate(self.filaments):
            for j in range(filament['n_turn']):
                r1, r2 = filament['r1'][j], filament['r2'][j]
                z1, z2 = filament['z1'][j], filament['z2'][j]
                sign = filament['turn'][j]
                x = 1e-2*np.mean([r1, r2])
                z = 1e-2*np.mean([z1, z2])
                coil = {'I': 0, 'dx': dx, 'dz': dz, 'rc': rc,
                        'x': x, 'z': z, 'index': i, 'sign': sign}
                if filament['n_turn'] == 1:  # vessel
                    name = 'vv_{}'.format(next(nvv))
                    self.vessel_coil[name] = coil
                else:
                    name = 'bb_{}'.format(next(nbl))
                    self.blanket_coil[name] = coil

    def get_filament_current(self):
        self.rt.skiplines(1)
        filament = self.rt.readblock()
        return filament

    def get_plasma_current(self):
        self.rt.skiplines(3)
        plasma = self.rt.readblock()
        return plasma

    def get_coil_current(self):
        self.skiplines(1)
        coil = self.rt.readblock()
        return coil

    def get_current(self):
        self.rt.skiplines(1)
        t = self.rt.readnumber()
        filament = self.get_filament_current()
        plasma = self.get_plasma_current()
        coil = self.get_coil_current()
        return (t, filament, plasma, coil)

    def store(self, frames):
        nt = len(frames)
        self.timeseries = {'t': np.zeros(nt), }


cage = ITERcage()  # import ITER coil cage
dina = read_dina(filename)

