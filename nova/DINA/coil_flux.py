from nep.coil_geom import VSgeom, VVcoils
import nova.cross_coil as cc
from amigo.pyplot import plt
import numpy as np
from collections import OrderedDict
from amigo.time import clock
from os.path import split, join, isfile
from scipy.interpolate import interp1d
from nep.DINA.read_tor import read_tor
from nep.DINA.read_plasma import read_plasma
from nep.DINA.read_dina import dina
from amigo.IO import pythonIO


class coil_flux(pythonIO):

    def __init__(self, Iscale=1, read_txt=False):
        self.read_txt = read_txt
        self.dina = dina('disruptions')
        self.pl = read_plasma('disruptions', Iscale=Iscale,
                              read_txt=read_txt)  # load plasma
        self.tor = read_tor('disruptions', Iscale=Iscale,
                            read_txt=read_txt)  # load currents
        pythonIO.__init__(self)  # python read/write

    def load_geometory(self, vessel=True):
        if vessel:
            self.coil_geom = VVcoils()
            # remove DINA vessel close to vs3 coils
            vv_remove = [0, 1] + list(np.arange(18, 23)) + \
                list(np.arange(57, 60)) + list(np.arange(91, 96)) +\
                list(np.arange(72, 76)) + list(np.arange(114, 116))
            for vv_index in vv_remove:
                self.tor.pf.remove_coil('vv_{}'.format(vv_index))
        else:
            self.coil_geom = VSgeom()
        self.flux = OrderedDict()
        for coil in self.coil_geom.pf.sub_coil:
            x = self.coil_geom.pf.sub_coil[coil]['x']
            z = self.coil_geom.pf.sub_coil[coil]['z']
            self.flux[coil] = {'x': x, 'z': z}

    def load_file(self, scenario, plot=False, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.dina.locate_file('plasma', folder=scenario)
        self.name = split(filepath)[-2]
        filepath = join(*split(filepath)[:-1], self.name, 'coil_flux')
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(scenario, **kwargs)  # read txt file
            self.save_pickle(filepath, ['t', 'flux', 'Vbg', 'dVbg'])
        else:
            self.load_pickle(filepath)
        if plot:
            self.plot_profile()
        vs3_trip = self.pl.Ivs3_single(scenario)[0]
        self.t_trip = vs3_trip['t_trip']

    def read_file(self, scenario, plot=False, **kwargs):
        self.tor.load_file(scenario)  # load toroidal scenario
        self.t = self.tor.t
        self.load_geometory()
        x, z = np.zeros(len(self.flux)), np.zeros(len(self.flux))
        for i, coil in enumerate(self.flux):  # pack
            x[i] = self.flux[coil]['x']
            z[i] = self.flux[coil]['z']
        psi_bg = np.zeros((self.tor.nt, len(self.flux)))
        tick = clock(self.tor.nt, header='calculating coil flux history')
        for index in range(self.tor.nt):
            self.tor.set_current(index)  # update coil currents and plasma
            psi_bg[index] = cc.get_coil_psi(x, z, self.tor.pf)
            tick.tock()
        vs3_bg = np.zeros(self.tor.nt)
        for i, coil in enumerate(self.flux):  # unpack
            nan_index = np.isnan(psi_bg[:, i])
            psi_bg[nan_index, i] = psi_bg[~nan_index, i][-1]
            self.flux[coil]['psi_bg'] = psi_bg[:, i]
            if 'jacket' not in coil:
                if 'lowerVS' in coil:
                    vs3_bg += psi_bg[:, i]
                elif 'upperVS' in coil:
                    vs3_bg -= psi_bg[:, i]
        self.flux['vs3'] = {'psi_bg': vs3_bg}
        dtype_array = '{}float'.format(self.tor.nt)
        bg = np.zeros(len(self.flux)-15,
                      dtype=[('V', dtype_array), ('dVdt', dtype_array)])
        bg['V'][0] = -2*np.pi*np.gradient(self.flux['vs3']['psi_bg'], self.t)
        bg['dVdt'][0] = np.gradient(bg['V'][0], self.t)
        bg['V'][1] = bg['V'][0]  # jacket turns
        bg['dVdt'][1] = bg['dVdt'][0]
        for i, coil in enumerate(self.flux):
            if i >= 16 and i < len(self.flux)-1:  # skip vs3 turns
                bg['V'][i-14] = -2*np.pi*np.gradient(self.flux[coil]['psi_bg'],
                                                     self.t)
                bg['dVdt'][i-14] = np.gradient(bg['V'][i-14], self.t)
        self.Vbg = interp1d(self.t, bg['V'], fill_value=0,
                            bounds_error=False)
        self.dVbg = interp1d(self.t, bg['dVdt'], fill_value=0,
                             bounds_error=False)

    def plot_profile(self):
        ax = plt.subplots(3, 1, sharex=True)[1]
        self.plot_flux(ax=ax[0])
        self.plot_background(ax=ax[1:])
        plt.detick(ax)

    def plot_flux(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
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
        ax[0].plot(1e3*self.t, 1e-3*self.Vbg(self.t)[0], 'C3-')
        ax[1].plot(1e3*self.t, 1e-6*self.dVbg(self.t)[0], 'C4-')
        plt.despine()
        plt.legend()
        ax[0].set_ylabel('$V_{bg}$ kV')
        ax[1].set_ylabel('$\dot{V}_{bg}$ MVt$^{-1}$')
        ax[1].set_xlabel('$t$ ms')


if __name__ == '__main__':
    cf = coil_flux()

    for i in range(12):
        cf.load_file(i, plot=True, read_txt=True)

    # vs3.plot_background()
    # vs3.calculate_background()




