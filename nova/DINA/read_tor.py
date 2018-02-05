import numpy as np
import pylab as pl
import matplotlib.animation as manimation
from amigo.time import clock
from collections import OrderedDict
from itertools import count
from nova.coils import PF
from amigo.IO import readtxt


class read_tor:
    # read tor_cur_data*.dat file from DINA simulation
    # listing of toroidal currents

    def __init__(self, directory, subfolder=None):
        filename = locate(directory, 'tor_cur_data', subfolder=subfolder)
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
                frames.append(self.get_current())
            except ValueError:
                break

        self.nt = len(frames)
        self.t = np.zeros(self.nt)
        self.current = \
            np.zeros(self.nt, dtype=[('filament', '{}float'.format(self.nf)),
                                     ('coil', '{}float'.format(self.nC))])
        self.plasma_coil = []
        self.plasma_patch = []
        for i, frame in enumerate(frames):
            self.t[i] = frame[0]
            self.current['filament'][i] = frame[1]
            self.current['coil'][i] = frame[3]
            self.plasma_coil.append(self.plasma_filaments(frame))

    def plasma_filaments(self, frame, dx=0.1, dz=0.1):
        rc = np.sqrt(dx**2 + dz**2) / 4
        npl = count(0)
        plasma_coil = OrderedDict()
        xp = 1e-2*np.array(frame[2][0::3])
        zp = 1e-2*np.array(frame[2][1::3])
        Ip = 1e3*np.array(frame[2][2::3])
        for x, z, I in zip(xp, zp, Ip):
                name = 'pl_{}'.format(next(npl))
                plasma_coil[name] = {'I': I, 'dx': dx, 'dz': dz, 'rc': rc,
                                     'x': x, 'z': z}
        return plasma_coil

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
        self.nC = len(coilnames)
        self.set_coil()
        self.pf = PF()
        self.pf.coil = self.coil
        self.pf.categorize_coils()

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
        self.filaments = \
            np.zeros(self.nf, dtype=[('r1', '2float'), ('z1', '2float'),
                                     ('r2', '2float'), ('z2', '2float'),
                                     ('turn', '2float'), ('n_turn', 'int')])
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
        self.rt.skiplines(1)
        coil = self.rt.readblock()
        return coil

    def get_current(self):
        self.rt.skiplines(1)
        t = self.rt.readnumber()
        filament = self.get_filament_current()
        plasma = self.get_plasma_current()
        coil = self.get_coil_current()
        return (t, filament, plasma, coil)

    def plot(self, index, ax=None):
        if ax is None:
            ax = plt.subplots(figsize=(7, 10))[1]
        self.plot_coils()
        self.plot_plasma(index)
        plt.axis('off')

    def plot_coils(self):
        self.pf.plot()
        self.pf.plot_coil(self.blanket_coil, coil_color='C2')
        self.pf.plot_coil(self.vessel_coil, coil_color='C3')

    def plot_plasma(self, index):
        patch = self.pf.plot_coil(self.plasma_coil[index], coil_color='C4')
        self.plasma_patch.extend(patch)

    def clear_plasma(self):
        for patch in self.plasma_patch:
            patch.remove()
        self.plasma_patch = []  # reset patch list

    def movie(self, filename):
        fig, ax = plt.subplots(1, 1, figsize=(7, 10))
        moviename = '../Movies/{}'.format(filename)
        moviename += '.wmv'
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=20, bitrate=-1, codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        tick = clock(self.nt)
        self.plot_coils()
        plt.axis('off')
        with writer.saving(fig, moviename, 72):
            for i in range(int(self.nt/10)):
                self.clear_plasma()
                self.plot_plasma(i*10)
                writer.grab_frame()
                tick.tock()

if __name__ == '__main__':

    directory = join(class_dir(nep), '../Scenario_database/disruptions')
    folder = 'MD_UP_lin50'

    tor = read_tor(directory, folder)

    tor.plot(500)

    #tor.movie('tmp')

    '''
    i = 500
    plt.plot(1e-2*np.array(tor.timeframes[i][2][0::3]),
             1e-2*np.array(tor.timeframes[i][2][1::3]), '.')
    '''

