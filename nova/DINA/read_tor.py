import numpy as np
import matplotlib.animation as manimation
from amigo.time import clock
from itertools import count
from nova.coils import PF
from amigo.IO import readtxt, pythonIO
from nep.DINA.read_dina import dina
from amigo.pyplot import plt
from nep.coil_geom import VSgeom
from os import path
from collections import OrderedDict
import copy


class read_tor(pythonIO):
    # read tor_cur_data*.dat file from DINA simulation
    # listing of toroidal currents

    def __init__(self, database_folder='disruptions', Ip_scale=1,
                 read_txt=False):
        self.Ip_scale = Ip_scale
        self.read_txt = read_txt
        self.dina = dina(database_folder)
        self.frame_index = 0
        pythonIO.__init__(self)  # python read/write

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.dina.locate_file('tor_cur', folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.name = filepath.split(path.sep)[-2]
        if read_txt or not path.isfile(filepath + '.pk'):
            self.read_file(filepath)  # read txt file
            self.save_pickle(filepath, ['coilset', 'current', 'plasma_coil',
                                        't', 'nt'])
        else:
            self.load_pickle(filepath)
        self.pf = PF()
        self.pf.add_coilsets(self.coilset)

    def read_file(self, filepath):  # called by load_file
        self.coilset = {}
        with readtxt(filepath + '.dat') as self.rt:
            self.read_coils()
            self.read_frames()

    def read_coils(self):
        self.rt.skiplines(5)  # skip header
        self.get_coils()
        self.get_filaments()

    def read_frames(self):
        frames = []
        nCS, nPF = self.coilset['CS']['nC'], self.coilset['PF']['nC']
        self.rt.skiplines(6)
        while True:
            try:
                frames.append(self.get_current())
            except ValueError:
                break
        self.nt = len(frames)
        self.t = np.zeros(self.nt)
        self.Ibar = np.zeros(self.nt, dtype=[('vv', float), ('pl', float)])
        dtype = [('filament', '{}float'.format(self.nf)),
                 ('CS', '{}float'.format(nCS)), ('PF', '{}float'.format(nPF))]
        self.current = np.zeros(self.nt, dtype=dtype)
        self.plasma_coil = []
        self.plasma_patch = []
        self.nVV = self.coilset['vv_DINA']['nC']
        for i, frame in enumerate(frames):
            self.t[i] = 1e-3*frame[0]  # ms-s
            self.current['filament'][i] = -1e3*np.array(frame[1])  # -kA to A
            self.current['CS'][i] = -1e3*np.array(frame[3][:nCS])  # -kA to A
            self.current['PF'][i] = -1e3*np.array(frame[3][nCS:])  # -kA to A
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
        for x, z, If in zip(xp, zp, Ip):
            name = 'Plasma_{}'.format(next(npl))
            plasma_coil[name] = {'If': If, 'dx': dx, 'dz': dz, 'rc': rc,
                                 'x': x, 'z': z}
        return plasma_coil, np.sum(Ip)

    def get_coils(self, dCoil=0.25):
        coil, coilnames = {}, []
        self.rt.checkline('1.')
        self.rt.skiplines(1)
        coilnames = self.rt.readline(True, string=True)
        coilnames.extend(self.rt.readline(True, string=True))
        for name in coilnames:
            coil[name] = {}  # initalise as empty dict
        for var in ['x', 'z', 'dx', 'dz']:
            self.rt.skiplines(1)
            self.fill_coil(coil, var, self.rt.readblock())
        self.set_coil(coil)
        CS_coil = {name: coil[name] for name in coil if 'CS' in name}
        pf = PF(coil=CS_coil, label='CS')
        pf.mesh_coils(dCoil=dCoil)
        self.coilset['CS'] = pf.coilset
        PF_coil = {name: coil[name] for name in coil if 'PF' in name}
        pf = PF(coil=PF_coil, label='PF')
        pf.mesh_coils(dCoil=dCoil)
        self.coilset['PF'] = pf.coilset

    def fill_coil(self, coil, key, values):  # set key/value pairs in coil dict
        for name, value in zip(coil, values):
            coil[name][key] = value

    def set_coil(self, coil):
        for name in coil:
            for var in ['dx', 'dz', 'x', 'z']:
                coil[name][var] *= 1e-2  # cm to meters
            coil[name]['rc'] = np.sqrt(coil[name]['dx']**2 +
                                       coil[name]['dz']**2) / 4
            coil[name]['It'] = 0

    def get_filaments(self, dCoil=0.25, plot=False):
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
        self.store_filaments(dCoil=dCoil)
        if plot:
            self.plot_filaments()

    def store_filaments(self, dx=0.15, dz=0.15, rho=0.8e-6, t=60e-3,
                        dCoil=0.25):
        rc = np.sqrt(dx**2 + dz**2) / 4
        nvv, nbl = count(0), count(0)
        blanket_coil = OrderedDict()
        vessel_coil = OrderedDict()
        vv = {'x': [], 'z': []}  # vv coils index
        for i, filament in enumerate(self.filaments):
            for j in range(filament['n_turn']):
                x1, x2 = 1e-2*filament['x1'][j], 1e-2*filament['x2'][j]
                z1, z2 = 1e-2*filament['z1'][j], 1e-2*filament['z2'][j]
                sign = filament['turn'][j]
                x = np.mean([x1, x2])
                z = np.mean([z1, z2])
                R = rho*2*np.pi*x / (t*np.sqrt((x2-x1)**2 + (z2-z1)**2))
                coil = {'It': 0, 'dx': dx, 'dz': dz, 'rc': rc,
                        'x': x, 'z': z, 'index': i, 'sign': sign,
                        'R': R}  # filament resistance
                if filament['n_turn'] == 1:  # vessel
                    name = 'vv_{}'.format(next(nvv))
                    vessel_coil[name] = coil
                    vv['x'].append(x)
                    vv['z'].append(z)
                else:
                    name = 'bb_{}'.format(next(nbl))
                    blanket_coil[name] = coil
        pf = PF(coil=blanket_coil, label='bb_DINA')
        pf.mesh_coils(dCoil=dCoil)
        self.coilset['bb_DINA'] = pf.coilset
        pf = PF(coil=vessel_coil, label='vv_DINA')
        pf.mesh_coils(dCoil=dCoil)
        self.coilset['vv_DINA'] = pf.coilset
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

    def package_current(self, coil):
        It = {}
        for name in coil:
            It[name] = coil[name]['It']
        return It

    def set_coil_current(self, frame_index, Ip_scale):
        for label in ['CS', 'PF']:
            It = {}
            for name, I in zip(self.coilset[label]['coil'],
                               self.current[label][frame_index]):
                It[name] = Ip_scale * I
            self.pf.update_current(It, self.coilset[label])  # update coilset
            self.pf.update_current(It)  # update pf instance

    def set_filament_current(self, coilset, frame_index, Ip_scale):
        It = {}
        current = self.current['filament'][frame_index]
        for name in coilset['coil']:
            turn_index = coilset['coil'][name]['index']
            sign = coilset['coil'][name]['sign']
            It[name] = Ip_scale * sign * current[turn_index]
        self.pf.update_current(It, coilset)  # update coilset
        self.pf.update_current(It)  # update pf instance

    def set_plasma_current(self, frame_index, Ip_scale):
        self.pf.coilset['plasma'] =\
            copy.deepcopy(self.plasma_coil[frame_index])
        for name in self.pf.coilset['plasma']:
            self.pf.coilset['plasma'][name]['If'] *= Ip_scale
        self.coilset['PF']['plasma'] = self.pf.coilset['plasma']

    def set_current(self, frame_index, **kwargs):
        Ip_scale = kwargs.get('Ip_scale', self.Ip_scale)
        self.frame_index = frame_index
        self.set_coil_current(frame_index, Ip_scale)
        self.set_filament_current(self.coilset['vv_DINA'],
                                  frame_index, Ip_scale)
        self.set_filament_current(self.coilset['bb_DINA'],
                                  frame_index, Ip_scale)
        self.set_plasma_current(frame_index, Ip_scale)

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
        plt.axis('off')

    def plot_coils(self):
        self.pf.plot(subcoil=True, plasma=True)

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
                fig.clf()
                self.plot(i*10, ax=ax)
                writer.grab_frame()
                tick.tock()


if __name__ == '__main__':

    tor = read_tor('disruptions', read_txt=False, Ip_scale=1)
    #for folder in tor.dina.folders:
    #    tor.load_file(folder, read_txt=True)
    tor.load_file(3, read_txt=False)
    tor.plot(200)

    # tor.pf.plot(current=False, plasma=True, subcoil=True)
    # tor.movie('tmp')



