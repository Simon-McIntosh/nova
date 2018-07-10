from nep.DINA.VDE_force import VDE_force
from nep.DINA.coupled_inductors import inductance
from nep.coil_geom import VSgeom
import nova.cross_coil as cc
from amigo.pyplot import plt
import numpy as np
from collections import OrderedDict
from amigo.time import clock
from os.path import split, join, isfile
from scipy.interpolate import interp1d
from amigo.png_tools import data_load
import nep
from amigo.IO import class_dir
import os
from scipy.optimize import minimize


class vs3_flux(VDE_force):

    def __init__(self):
        VDE_force.__init__(self, mode='control', discharge='DINA', Iscale=1)

    def load_psi(self, folder, plot=False, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.dina.locate_file('plasma', folder=folder)
        self.name = split(filepath)[-2]
        filepath = join(*split(filepath)[:-1], self.name, 'vs3_flux')
        if read_txt or not isfile(filepath + '.pk'):
            self.read_psi(folder, **kwargs)  # read txt file
            self.save_pickle(filepath, ['flux', 'time'])
        else:
            self.load_pickle(filepath)
        if plot:
            self.plot_flux()
        self.t = self.time
        vs3_trip = vs3.pl.Ivs3_single(folder)[0]
        self.t_trip = vs3_trip['t_trip']
        self.load_LTC()

    def load_LTC(self, plot=False, **kwargs):
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        points = data_load(path, 'VS3_discharge_main_report',
                           date='2018_06_25')[0]
        to, Io = points[0]['x'], points[0]['y']  # bare conductor
        io = np.append(np.diff(to) > 0, True)
        to, Io = to[io], Io[io]
        to -= to[0]
        td, Id = points[1]['x'], points[1]['y']  # jacket + vessel
        io = np.append(np.diff(td) > 0, True)
        td, Id = td[io], Id[io]
        td -= td[0]
        self.LTC = OrderedDict()
        self.LTC['LTC bare'] = \
            {'t': self.t, 'Ic': interp1d(to, Io, kind='cubic')(self.t)}
        self.LTC['LTC+vessel'] = \
            {'t': self.t, 'Ic': interp1d(td, Id, kind='cubic')(self.t)}
        if plot:
            self.plot_LTC(**kwargs)

    def plot_LTC(self, **kwargs):
        ax = kwargs.get('ax', None)
        if ax is None:
            ax = plt.subplots(1, 1)[0]
        for discharge in self.LTC:
            plt.plot(1e3*self.t, 1e-3*self.LTC[discharge]['Ic'],
                     label=discharge)
        plt.despine()
        plt.xlabel('$t$ ms')
        plt.ylabel('$I$ kA')
        plt.legend()

    def initalize(self, folder, **kwargs):
        mode = kwargs.get('mode', self.mode)
        discharge = kwargs.get('discharge', self.discharge)
        self.tor.load_file(folder)  # read toroidal strucutres
        self.load_vs3(folder, discharge=discharge)  # load vs3 currents
        self.frame_update(0)  # initalize timeseries
        self.vs3_update(mode=mode)  # initalize vs3 current
        vs_geom = VSgeom().geom
        self.flux = OrderedDict()
        nt = self.tor.nt
        self.time = self.tor.t
        for coil in vs_geom:
            x, z = vs_geom[coil]['x'], vs_geom[coil]['z']
            self.flux[coil] = {'x': x, 'z': z, 'psi_bg': np.zeros(nt)}
        x, y, z = 6.5525, 3.7831, -2.6136  # lower VS
        x = np.linalg.norm([x, y])
        self.flux['lowerVV'] = {'x': x, 'z': z, 'psi_bg': np.zeros(nt)}
        x, y, z = 5.7514, 1.0141, 5.0388  # upper VS
        x = np.linalg.norm([x, y])
        self.flux['upperVV'] = {'x': x, 'z': z, 'psi_bg': np.zeros(nt)}

    def read_psi(self, folder, **kwargs):
        self.load_file(folder)
        self.initalize(folder, **kwargs)
        x, z = np.zeros(len(self.flux)), np.zeros(len(self.flux))
        for i, coil in enumerate(self.flux):  # pack
            x[i] = self.flux[coil]['x']
            z[i] = self.flux[coil]['z']
        psi_bg = np.zeros((self.tor.nt, len(self.flux)))
        tick = clock(self.tor.nt, header='calculating coil flux history')
        for frame_index in range(self.tor.nt):
            # update coil currents and plasma position
            self.frame_update(frame_index)
            psi_bg[frame_index] = cc.get_coil_psi(x, z, self.pf)
            tick.tock()
        for i, coil in enumerate(self.flux):  # unpack
            self.flux[coil]['psi_bg'] = psi_bg[:, i]

    def plot_flux(self):
        for coil in self.flux:
            plt.plot(1e3*self.t, self.flux[coil]['psi_bg'], label=coil)
        plt.despine()
        plt.legend()
        plt.xlabel('$t$ ms')
        plt.ylabel('$\psi$ Weber rad$^{-1}$')

    def build_coilset(self, include_vessel=True):
        vs_geom = VSgeom()
        self.ind = inductance()
        nvs_o = self.ind.nC
        turns = np.append(np.ones(4), -np.ones(4))
        self.ind.add_pf_coil(vs_geom.pf.sub_coil, turns=turns)
        for i, index in enumerate(nvs_o+np.arange(1, 8)):  # vs3 loops
            self.ind.add_cp([nvs_o, index])  # link VS coils
        dx = dz = 60e-3
        if include_vessel:
            for vv in ['lowerVV', 'upperVV']:
                x, z = self.flux[vv]['x'], self.flux[vv]['z']
                self.ind.add_coil(x, z, dx, dz, 0, R=0, nt=1)
        self.ind.reduce()

    def calculate_background(self):
        self.flux['vs3'] = {'psi_bg': 4 * self.flux['lowerVS']['psi_bg'] -
                            4 * self.flux['upperVS']['psi_bg']}
        V, vfun = [], []
        V.append(-2*np.pi*np.gradient(self.flux['vs3']['psi_bg'], self.t))
        V.append(-2*np.pi*np.gradient(self.flux['lowerVV']['psi_bg'], self.t))
        V.append(-2*np.pi*np.gradient(self.flux['upperVV']['psi_bg'], self.t))
        for v in V:
            vfun.append(interp1d(self.t, v, fill_value=0, bounds_error=False))

        print(len(vfun))

        self.build_coilset(include_vessel=True)
        self.ind.Ic[0] = 60e3  # inital current
        self.ind.Rc[0] = 17.66e-3  # total vs3 resistance
        self.ind.M[0, 0] += 0.2e-3  # add busbar inductance
        self.ind.M[0, 0] += 0.3e-4  # adjust self inductance to aggree with LTC

        self.ind.Rc[1] = 2e-3  # lowerVV resistance
        self.ind.Rc[2] = 2e-4  # upperVV resistance
        vs3.get_Rvv()


        self.ind.Ic[0] = -60e3  # inital current
        plt.figure()
        print(self.ind.Rc)
        t = np.linspace(self.t_trip, self.t[-1], 500)
        t = np.linspace(0, 10*self.t[-1], 500)
        Iode = self.ind.solve(t)  # , vfun=vfun
        for i, name in enumerate(['vs3', 'lowerVV', 'upperVV']):
            plt.plot(t, Iode[i], label=name)
        plt.despine()
        plt.legend()

        print(1e-3*np.min((Iode[0])))

        '''
        self.build_coilset(include_vessel=False)
        self.ind.Ic[0] = -60e3  # inital current
        self.ind.Rc[0] = 17.66e-3  # total vs3 resistance
        self.ind.M[0, 0] += 0.2e-3  # add busbar inductance
        self.ind.M[0, 0] += 0.3e-4  # adjust self inductance to aggree with LTC
        Iode = self.ind.solve(t, vfun=vfun[:1])
        for i, name in enumerate(['vs3 bare']):
            plt.plot(t, Iode[i], label=name)
        plt.despine()
        plt.legend()

        print(1e-3*np.min((Iode[0])))
        '''

        '''



        t = np.linspace(0, self.t[-1], 200)

        Iode = self.ind.solve(t)
        tau = self.ind.M[0, 0] / self.ind.Rc[0]

        self.plot_LTC()

        plt.plot(1e3*t, 1e-3*self.ind.Ic[0]*np.exp(-t/tau), 'C0--')
        plt.plot(1e3*t, 1e-3*Iode[0], 'C1--')
        '''

    def fit_waveform(self, Rvv):
        Iref = self.LTC['LTC+vessel']['Ic']
        self.update_Rvv(Rvv)
        Iode = self.ind.solve(self.LTC['LTC+vessel']['t'])
        err = np.linalg.norm(Iode[0] - Iref) / np.linalg.norm(Iref)
        return err

    def update_Rvv(self, Rvv):
        self.ind.Rc[1] = np.max([Rvv[0], 1e-4])  # lowerVV resistance
        self.ind.Rc[2] = np.max([Rvv[1], 1e-4])   # upperVV resistance

    def get_Rvv(self):
        xo = [2e-4, 2e-4]  # sead resistance vector
        Rvv = minimize(vs3.fit_waveform, xo, method='Nelder-Mead').x
        self.update_Rvv(Rvv)
        t = self.LTC['LTC+vessel']['t']
        Iode = self.ind.solve(t)

        self.plot_LTC()
        plt.plot(1e3*t, 1e-3*self.ind.Ic[0]*np.exp(-t/tau), 'C0--')
        plt.plot(1e3*t, 1e-3*Iode[0], 'C1--')



if __name__ == '__main__':
    vs3 = vs3_flux()
    vs3.load_psi(3, plot=False, read_txt=False)
    vs3.calculate_background()




