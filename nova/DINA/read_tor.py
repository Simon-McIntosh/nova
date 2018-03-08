import numpy as np
import matplotlib.animation as manimation
from amigo.time import clock
from collections import OrderedDict
from itertools import count
from nova.coils import PF
from amigo.IO import readtxt
from nep.DINA.read_dina import dina
from amigo.pyplot import plt
from nep.coil_geom import VSgeom


class read_tor:
    # read tor_cur_data*.dat file from DINA simulation
    # listing of toroidal currents

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)
        self.frame_index = 0

    def read_file(self, folder):
        filename = self.dina.locate_file('tor_cur', folder=folder)
        self.name = filename.split('\\')[-2]
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
        self.Ibar = np.zeros(self.nt, dtype=[('vv', float), ('pl', float)])
        self.current = \
            np.zeros(self.nt, dtype=[('filament', '{}float'.format(self.nf)),
                                     ('coil', '{}float'.format(self.nC))])
        self.plasma_coil = []
        self.plasma_patch = []
        self.nVV = len(self.vessel_coil)
        for i, frame in enumerate(frames):
            self.t[i] = 1e-3*frame[0]  # ms-s
            self.current['filament'][i] = -1e3*np.array(frame[1])  # -kA to A
            self.current['coil'][i] = -1e3*np.array(frame[3])  # -kA to A
            plasma_coil, Ipl = self.plasma_filaments(frame)
            self.plasma_coil.append(plasma_coil)
            self.Ibar['vv'][i] = np.mean(-1e3*np.array(frame[1])[:self.nVV])
            self.Ibar['pl'][i] = Ipl

    def plasma_filaments(self, frame, dx=0.15, dz=0.15):
        rc = np.sqrt(dx**2 + dz**2) / 4
        npl = count(0)
        plasma_coil = {}  # OrderedDict()
        xp = 1e-2*np.array(frame[2][0::3])
        zp = 1e-2*np.array(frame[2][1::3])
        Ip = -1e3*np.array(frame[2][2::3])  # -kA to A
        for x, z, Ic in zip(xp, zp, Ip):
            name = 'Plasma_{}'.format(next(npl))
            plasma_coil[name] = {'Ic': Ic, 'dx': dx, 'dz': dz, 'rc': rc,
                                 'x': x, 'z': z}
        return plasma_coil, np.sum(Ip)

    def get_coils(self):
        self.coil, coilnames = {}, []  # OrderedDict(), []
        self.rt.checkline('1.')
        self.rt.skiplines(1)
        coilnames = self.rt.readline(True, string=True)
        coilnames.extend(self.rt.readline(True, string=True))
        for name in coilnames:
            self.coil[name] = {}  # initalise as empty dict
        for var in ['x', 'z', 'dx', 'dz']:
            self.rt.skiplines(1)
            self.fill_coil(var, self.rt.readblock())
        self.nC = len(coilnames)
        self.set_coil()
        self.pf = PF()
        self.pf.coil = self.coil
        self.pf.index = {}
        self.pf.index['CS'] = {'name': [name for name in coilnames
                                        if 'CS' in name]}
        self.pf.index['PF'] = {'name': [name for name in coilnames
                                        if 'PF' in name]}

    def fill_coil(self, key, values):  # set key/value pairs in coil dict
        for name, value in zip(self.coil, values):
            self.coil[name][key] = value

    def set_coil(self):
        for name in self.coil:
            for var in ['dx', 'dz', 'x', 'z']:
                self.coil[name][var] *= 1e-2  # cm to meters
            self.coil[name]['rc'] = np.sqrt(self.coil[name]['dx']**2 +
                                            self.coil[name]['dz']**2) / 4
            self.coil[name]['Ic'] = 0

    def get_filaments(self, plot=False):
        self.rt.checkline('2.')
        self.rt.skiplines(1)
        self.nf = self.rt.readnumber()
        self.rt.skiplines(3)
        self.filaments = \
            np.zeros(self.nf, dtype=[('x1', '2float'), ('z1', '2float'),
                                     ('x2', '2float'), ('z2', '2float'),
                                     ('turn', '2float'), ('n_turn', 'int')])
        for i in range(self.nf):
            n_turn = self.rt.readnumber()
            self.filaments[i]['n_turn'] = n_turn
            for j in range(n_turn):
                x1, z1, x2, z2, turn = [float(d) for d in
                                        self.rt.readline(True)]
                for var, value in zip(['x1', 'z1', 'x2', 'z2', 'turn'],
                                      [x1, z1, x2, z2, turn]):
                    self.filaments[i][var][j] = value
        self.store_filaments()
        if plot:
            self.plot_filaments()

    def store_filaments(self, dx=0.15, dz=0.15, rho=0.8e-6, t=60e-3):
        rc = np.sqrt(dx**2 + dz**2) / 4
        nvv, nbl = count(0), count(0)
        self.blanket_coil = {}  # OrderedDict()
        self.vessel_coil = {}  # OrderedDict()
        vv = {'x': [], 'z': []}  # vv coils index
        for i, filament in enumerate(self.filaments):
            for j in range(filament['n_turn']):
                x1, x2 = 1e-2*filament['x1'][j], 1e-2*filament['x2'][j]
                z1, z2 = 1e-2*filament['z1'][j], 1e-2*filament['z2'][j]
                sign = filament['turn'][j]
                x = np.mean([x1, x2])
                z = np.mean([z1, z2])
                R = rho*2*np.pi*x / (t*np.sqrt((x2-x1)**2 + (z2-z1)**2))
                coil = {'Ic': 0, 'dx': dx, 'dz': dz, 'rc': rc,
                        'x': x, 'z': z, 'index': i, 'sign': sign,
                        'R': R}  # filament resistance
                if filament['n_turn'] == 1:  # vessel
                    name = 'vv_{}'.format(next(nvv))
                    self.vessel_coil[name] = coil
                    vv['x'].append(x)
                    vv['z'].append(z)
                else:
                    name = 'bb_{}'.format(next(nbl))
                    self.blanket_coil[name] = coil
        self.get_vv_vs_index(vv)

    def get_vv_vs_index(self, vv):
        vs_geom = VSgeom()
        self.vv_vs_index = np.zeros(2, dtype=int)
        for i, coil in enumerate(vs_geom.geom):
            vs_coil = vs_geom.geom[coil]
            self.vv_vs_index[i] = np.argmin(
                    (vs_coil['x']-np.array(vv['x']))**2 +
                    (vs_coil['z']-np.array(vv['z']))**2)

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

    def set_coil_current(self, frame_index):
        for name, Ic in zip(self.coil, self.current['coil'][frame_index]):
            self.coil[name]['Ic'] = Ic

    def set_filament_current(self, filament, frame_index):
        current = self.current['filament'][frame_index]
        for name in filament:
            turn_index = filament[name]['index']
            sign = filament[name]['sign']
            filament[name]['Ic'] = sign * current[turn_index]

    def set_current(self, frame_index):
        self.frame_index = frame_index
        self.set_coil_current(frame_index)
        self.set_filament_current(self.vessel_coil, frame_index)
        self.set_filament_current(self.blanket_coil, frame_index)

    def plot_filaments(self):
        for f in self.filaments:
            for j in range(f['n_turn']):
                plt.plot([f['x1'][j], f['x2'][j]],
                         [f['z1'][j], f['z2'][j]], 'o-')
        plt.axis('equal')

    def plot(self, index, ax=None):
        if ax is None:
            ax = plt.subplots(figsize=(7, 10))[1]
        self.set_current(index)
        self.plot_coils()
        # self.plot_plasma(index)
        plt.axis('off')

    def plot_coils(self):
        self.pf.patch_coil(self.coil)
        self.pf.patch_coil(self.blanket_coil)
        self.pf.patch_coil(self.vessel_coil)
        self.pf.patch_coil(self.plasma_coil[self.frame_index])
        self.pf.plot_patch(c='Jc')
        plt.axis('equal')
        plt.axis('off')

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


    tor = read_tor('disruptions')
    tor.read_file(3)


    tor.set_current(200)

    tor.plot(130)



    #tor.movie('tmp')

    '''
    i = 500
    plt.plot(1e-2*np.array(tor.timeframes[i][2][0::3]),
             1e-2*np.array(tor.timeframes[i][2][1::3]), '.')
    '''

