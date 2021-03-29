from os import path

import numpy as np
#import matplotlib.animation as manimation
from shapely.geometry import Polygon
import pandas as pd
from sklearn.neighbors import KDTree
from shapely.geometry import MultiPoint
from descartes import PolygonPatch

#from amigo.time import clock
#from amigo.IO import readtxt, pythonIO

from nova.electromagnetic.IO.read_waveform import read_dina
from nova.electromagnetic.frame import Frame
from nova.electromagnetic.machinedata import MachineData
from nova.utilities.pyplot import plt
from nova.utilities.IO import readtxt



class read_tor(read_dina):
    """
    Read tor_cur_data*.dat file from DINA simulation.

    Temperal listing of toroidal currents.

    """
    def __init__(self, database_folder='disruptions', read_txt=False):
        super().__init__(database_folder, read_txt)
        self.frame = Frame(delta=0.25, available=['section'])

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.locate_file('tor_cur', folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.name = filepath.split(path.sep)[-2]

        self.read_file(filepath)  # read txt file

        '''
        if read_txt or not path.isfile(filepath + '.pk'):
            self.read_file(filepath)  # read txt file
            self.save_coilset('coilset', directory=self.folder)
            self.save_pickle(filepath, ['frames'])
        else:
            self.load_coilset('coilset', directory=self.folder)
            self.load_pickle(filepath)
        '''


    def read_file(self, filepath):  # called by load_file
        with readtxt(filepath + '.dat') as self.rt:
            self.read_coils()
            #self.read_frames()

    def read_coils(self):
        self.rt.skiplines(5)  # skip header
        self.insert_coils()
        #self.get_filaments()
        #self.set_plasma()

    def insert_coils(self):
        self.rt.checkline('1.')
        self.rt.skiplines(1)
        index = self.rt.readline(True, string=True)
        index.extend(self.rt.readline(True, string=True))
        geom = np.zeros((4, len(index)))
        for i, var in enumerate(['x', 'z', 'dx', 'dz']):
            self.rt.skiplines(1)
            geom[i, :] = self.rt.readblock()
        geom *= 1e-2  # cm to meters
        part = ['CS' if 'CS' in name else 'PF' for name in index]
        self.frame.insert(*geom, name=index, part=part)

    def get_filaments(self, dt=60e-3, rho=0.8e-6, dCoil=0.25):
        self.rt.checkline('2.')
        self.rt.skiplines(1)
        self.nf = self.rt.readnumber()
        self.rt.skiplines(3)
        self.filaments = \
            np.zeros(self.nf, dtype=[('x1', '2float'), ('z1', '2float'),
                                     ('x2', '2float'), ('z2', '2float'),
                                     ('turn', '2float'), ('n_turn', 'int')])
        # read from file
        for i in range(self.nf):
            n_turn = self.rt.readnumber()
            self.filaments[i]['n_turn'] = n_turn
            for j in range(n_turn):
                x1, z1, x2, z2, turn = [float(d) for d in
                                        self.rt.readline(True)]
                for var, value in zip(['x1', 'z1', 'x2', 'z2', 'turn'],
                                      [x1, z1, x2, z2, turn]):
                    self.filaments[i][var][j] = value
        # add shells
        for i, filament in enumerate(self.filaments):
            for j in range(filament['n_turn']):
                x1, x2 = 1e-2*filament['x1'][j], 1e-2*filament['x2'][j]
                z1, z2 = 1e-2*filament['z1'][j], 1e-2*filament['z2'][j]
                part = 'vv' if filament['n_turn'] == 1 else 'bb'
                self.add_shell([x1, x2], [z1, z2], dt, part=part, rho=rho,
                               dShell=0, dCoil=dCoil)
                if j == 1:  # link blanket pairs
                    self.add_mpc(self.coil.index[-2:], factor=-1)

    def set_plasma(self, dPlasma=0.25):
        machine = MachineData()
        machine.load_data()
        fw = pd.concat((machine.data['firstwall'].iloc[::-1],
                        machine.data['divertor']))
        fw_poly = Polygon([(x, z) for x, z in zip(fw.x, fw.z)])
        limit = [fw.x.min(), fw.x.max(), fw.z.min(), fw.z.max()]
        self.add_plasma(np.mean(limit[:2]), np.mean(limit[2:]),
                  np.diff(limit[:2])[0], np.diff(limit[2:])[0],
                  dPlasma=dPlasma, Nt=0, name='plasma')  #polygon=fw_poly

    def read_frames(self):
        self.frames = []
        self.rt.skiplines(6)
        while True:
            try:
                self.frames.append(self.get_current())
            except ValueError:
                break

    @staticmethod
    def get_delta(value):
        diff = np.diff(value)
        diff, count = np.unique(diff[diff > 0], return_counts=True)
        delta = diff[np.argmax(count)]
        return delta

    def plot_plasma(self, index):
        frame = self.frames[index]

        xp = 1e-2*np.array(frame[2][0::3])
        zp = 1e-2*np.array(frame[2][1::3])
        Ip = -1e3*np.array(frame[2][2::3])  # -kA to A

        dx = self.get_delta(xp)
        dz = self.get_delta(zp)
        dA = dx*dz

        points = [(x, z) for x, z in zip(xp, zp)]
        separatrix = MultiPoint(points).buffer(np.max([dx, dz]) / 2,
                                cap_style=3)

        ax = plt.subplots(1, 1)[1]
        ax.add_patch(PolygonPatch(separatrix))
        print(separatrix)


        self.set_plasma(dPlasma=0.3)

        self.plasma = self.subset(self.coil.plasma)
        self.plasma.plot(passive=True)

        tree = KDTree(np.array([self.plasma.subcoil.x,
                                self.plasma.subcoil.z]).T)

        k =int((1 + 2*(np.ceil(dx / (2*self.dPlasma)))) *\
                        (1 + 2*(np.ceil(dz / (2*self.dPlasma)))))

        ind_o = tree.query(np.array([xp, zp]).T,
                           k=1, return_distance=False).flatten()

        print(np.shape(ind_o))
        ind = tree.query(np.array([self.plasma.subcoil.x[ind_o],
                                   self.plasma.subcoil.z[ind_o]]).T,
                         k=k, return_distance=False)
        Ip_sum = Ip.sum()
        offset = 38

        Np = np.zeros(len(self.Np))
        for i in range(1):
            plt.plot(xp[i+offset], zp[i+offset], 'C0o')
            plt.plot(self.plasma.subcoil.x[ind[i+offset]],
                     self.plasma.subcoil.z[ind[i+offset]], 'k.')
            x = xp[i+offset] + dx/2 * np.array([-1, -1, 1, 1, -1])
            z = zp[i+offset] + dz/2 * np.array([-1, 1, 1, -1, -1])
            plt.plot(x, z, 'C3')

        #self.plot()
        plt.plot(xp, zp, 'C3.')
        plt.axis('equal')

        '''
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
        '''

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

    '''
    def plasma_filaments(self, frame, dx=0.15, dz=0.15):
        rc = np.sqrt(dx**2 + dz**2) / 4
        npl = count(0)
        plasma_coil = {}
        xp = 1e-2*np.array(frame[2][0::3])
        zp = 1e-2*np.array(frame[2][1::3])
        Ip = -1e3*np.array(frame[2][2::3])  # -kA to A
        for x, z, If in zip(xp, zp, Ip):
            name = 'Plasma_{}'.format(next(npl))
            plasma_coil[name] = {'If': If, 'dx': dx, 'dz': dz, 'rc': rc,
                                 'x': x, 'z': z}
        return plasma_coil, np.sum(Ip)

    def get_vv_vs_index(self, vv):
        vs_geom = VSgeom()
        self.vv_vs_index = np.zeros(2, dtype=int)
        for i, coil in enumerate(vs_geom.geom):
            vs_coil = vs_geom.geom[coil]
            self.vv_vs_index[i] = np.argmin(
                    (vs_coil['x']-np.array(vv['x']))**2 +
                    (vs_coil['z']-np.array(vv['z']))**2)

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

    def plot(self, index, ax=None):
        if ax is None:
            ax = plt.subplots(figsize=(7, 10))[1]
        self.set_current(index)
        self.plot_coils()
        plt.axis('off')

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
    '''

if __name__ == '__main__':

    tor = read_tor('disruptions', read_txt=False)
    #for folder in tor.dina.folders:
    #    tor.load_file(folder, read_txt=True)
    tor.load_file(-1)

    #tor.plot_plasma(200)
    #tor.plot(200)

    # tor.pf.plot(current=False, plasma=True, subcoil=True)
    # tor.movie('tmp')



