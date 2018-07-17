from nep.DINA.VDE_force import VDE_force
from nep.DINA.coupled_inductors import inductance
from nep.coil_geom import VSgeom, VVcoils
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
            self.save_pickle(filepath, ['t', 'flux', 'Vbg', 'dVbg'])
        else:
            self.load_pickle(filepath)
        if plot:
            self.plot_profile()
        vs3_trip = self.pl.Ivs3_single(folder)[0]
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
        # coil_geom = VSgeom()
        coil_geom = VVcoils()
        self.flux = OrderedDict()
        nt = self.tor.nt
        self.time = self.tor.t
        for coil in coil_geom.pf.sub_coil:
            x = coil_geom.pf.sub_coil[coil]['x']
            z = coil_geom.pf.sub_coil[coil]['z']
            self.flux[coil] = {'x': x, 'z': z, 'psi_bg': np.zeros(nt)}

    def read_psi(self, folder, plot=False, **kwargs):
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
            self.frame_update(frame_index)  # utilises self.t for time instance
            psi_bg[frame_index] = cc.get_coil_psi(x, z, self.pf)
            tick.tock()
        self.t = self.time  # revert to time vector
        vs3_bg = np.zeros(self.tor.nt)
        for i, coil in enumerate(self.flux):  # unpack
            self.flux[coil]['psi_bg'] = psi_bg[:, i]
            if 'lowerVS' in coil:
                vs3_bg += psi_bg[:, i]
            elif 'upperVS' in coil:
                vs3_bg -= psi_bg[:, i]
        self.flux['vs3'] = {'psi_bg': vs3_bg}
        V = -2*np.pi*np.gradient(self.flux['vs3']['psi_bg'], self.t)
        dVdt = np.gradient(V, self.t)
        self.Vbg = interp1d(self.t, V, fill_value=0, bounds_error=False)
        self.dVbg = interp1d(self.t, dVdt, fill_value=0, bounds_error=False)

    def plot_profile(self):
        ax = plt.subplots(3, 1, sharex=True)[1]
        self.plot_flux(ax=ax[0])
        self.plot_background(ax=ax[1:])
        plt.detick(ax)

    def plot_flux(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]

        ax.plot(1e3*self.t, self.flux['upper_vv1_22_0']['psi_bg'], '-', lw=1,
                color='gray')
        for i, coil in enumerate(['lowerVS', 'upperVS']):
            for isc in range(4):
                subcoil = '{}_{}'.format(coil, isc)
                label = coil if isc == 0 else ''
                ax.plot(1e3*self.t, self.flux[subcoil]['psi_bg'], '-', lw=1,
                        label=label, color='C{}'.format(i+1))
        ax.plot(1e3*self.t, self.flux['vs3']['psi_bg'], '-',
                label='vs3 loop', color='C0')
        plt.despine()
        ax.legend()
        ax.set_xlabel('$t$ ms')
        ax.set_ylabel('$\psi$ Weber rad$^{-1}$')

    def plot_background(self, ax=None):
        if ax is None:
            ax = plt.subplots(2, 1)[1]
        ax[0].plot(1e3*self.t, np.zeros(len(self.t)), '-.', color='lightgray')
        ax[0].plot(1e3*self.t, 1e-3*self.Vbg(self.t), 'C3-')
        ax[1].plot(1e3*self.t, 1e-6*self.dVbg(self.t), 'C4-')
        plt.despine()
        plt.legend()
        ax[0].set_ylabel('$V_{bg}$ kV')
        ax[1].set_ylabel('$\dot{V}_{bg}$ MVt$^{-1}$')
        ax[1].set_xlabel('$t$ ms')


if __name__ == '__main__':
    vs3 = vs3_flux()
    vs3.load_psi(3, plot=True, read_txt=False)
    #vs3.plot_background()
    # vs3.calculate_background()




