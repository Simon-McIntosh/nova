from nep.DINA.read_tor import read_tor
from nep.coil_geom import elm_coils
from collections import OrderedDict
import nova.cross_coil as cc
from amigo.time import clock
import numpy as np
from amigo.IO import pythonIO
from os.path import split, join, isfile
from scipy.interpolate import interp1d
from nep.DINA.read_dina import dina
from amigo.pyplot import plt
from nep.DINA.scenario import scenario


class elm_flux(pythonIO):

    def __init__(self, read_txt=False, reverse_current=False):
        super().__init__()  # python read/write
        self.read_txt = read_txt
        self.reverse_current = reverse_current
        self.dina = dina('disruptions')
        self.tor = read_tor('disruptions', read_txt=False)
        self.initalize_flux()

    def load_file(self, folder, plot=False, **kwargs):
        self.folder = folder
        self.reverse_current = kwargs.get('reverse_current',
                                          self.reverse_current)
        kwargs.pop('reverse_current', None)
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.dina.locate_file('plasma', folder=self.folder)
        self.name = split(filepath)[-2]
        filename = 'elm_flux'
        if self.reverse_current:
            filename += '_reverse'
        filepath = join(*split(filepath)[:-1], self.name, filename)
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(self.folder, **kwargs)  # read txt file
            self.save_pickle(filepath, ['t', 'turn', 'flux', 'V', 'Vinterp',
                                        'names', 'reverse_current',
                                        'folder', 'name'])
        else:
            self.load_pickle(filepath)

    def initalize_flux(self):
        self.coilset = elm_coils().pf.coilset
        self.turn = OrderedDict()
        for name in self.coilset['coil']:
            x = self.coilset['coil'][name]['x']
            z = self.coilset['coil'][name]['z']
            torodal_fraction = self.coilset['coil'][name]['tf']
            self.turn[name] = {'x': x, 'z': z, 'tf': torodal_fraction}
            Nf = self.coilset['coil'][name]['Nf']
            for i in range(Nf):
                subcoil = f'{name}_{i}'
                x = self.coilset['subcoil'][subcoil]['x']
                z = self.coilset['subcoil'][subcoil]['z']
                self.turn[subcoil] = {'x': x, 'z': z, 'tf': torodal_fraction}

    def read_file(self, folder):
        self.tor.load_file(folder)  # load toroidal scenario
        self.t = self.tor.t
        self.name = self.tor.name
        self.folder = folder
        x = np.array([self.turn[name]['x'] for name in self.turn])
        z = np.array([self.turn[name]['z'] for name in self.turn])
        psi = np.zeros((self.tor.nt, len(self.turn)))
        tick = clock(self.tor.nt, header='calculating elm coil flux history')
        Ip_scale = -1 if self.reverse_current else 1
        for i in range(self.tor.nt):
            self.tor.set_current(i, Ip_scale=Ip_scale)
            psi[i] = cc.get_coil_psi(x, z, self.tor.pf.coilset['subcoil'],
                                     self.tor.pf.coilset['plasma'])
            tick.tock()
        for i, name in enumerate(self.turn):  # store turn flux
            self.turn[name]['psi'] = psi[:, i]
            self.turn[name]['psi'] *= self.turn[name]['tf']
        self.calculate()

    def calculate(self):
        self.names = np.unique(['_'.join(name.split('_')[:-1])
                               for name in self.coilset['coil']])
        self.flux = OrderedDict()
        self.V = OrderedDict()  # driving voltage
        self.Vinterp = OrderedDict()  # voltage interpolator
        for name in self.names:
            name_bg = f'{name}_bg'
            self.flux[name] = np.zeros(self.tor.nt)
            self.flux[name_bg] = np.zeros(self.tor.nt)  # background
            Nf = self.coilset['coil'][f'{name}_feed']['Nf']
            for sign, line in zip([1, -1], ['feed', 'return']):
                turn = f'{name}_{line}'
                self.flux[name_bg] += sign * self.turn[turn]['psi']
                for i in range(Nf):
                    subturn = f'{name}_{line}_{i}'
                    self.flux[name] += sign * self.turn[subturn]['psi']
            for n in [name, name_bg]:
                self.V[n] = -2*np.pi*np.gradient(self.flux[n], self.t)
                self.Vinterp[n] = interp1d(self.t, self.V[n], fill_value=0,
                                           bounds_error=False)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplots(2, 1)[1]
        names = kwargs.get('names', self.names)
        for name in names:
            label = name.replace('_', ' ')
            if self.reverse_current:
                label += ' (reverse)'
            ax[0].plot(self.t, 2*np.pi*self.flux[name])
            ax[1].plot(self.t, 1e-3*self.V[name], label=label)
        plt.despine()
        ax[0].set_ylabel('$\psi$ Wb')
        ax[1].set_xlabel('$t$ s')
        ax[1].set_ylabel('$V$ kA')
        ax[0].set_title(self.name)
        ax[1].legend()
        plt.detick(ax)

if __name__ is '__main__':

    elm = elm_flux(read_txt=False)

    ax = plt.subplots(2, 1)[1]
    elm.load_file(0, reverse_current=False)
    elm.plot(names=['lower_elm'], ax=ax)
    elm.load_file(0, reverse_current=True)
    elm.plot(names=['lower_elm'], ax=ax)

    #plt.plot(elm.t, elm.flux['lower_elm'])
    #plt.plot(elm.t, elm.flux['lower_elm_bg'])

    scn = scenario(read_txt=False)
    scn.load_boundary()
    tor = read_tor('disruptions', read_txt=False)
    tor.load_file(0)
    scn.pf = tor.pf  # link PF instance
    tor.set_current(320)
    scn.update_psi(n=1e3, plot=True)
    scn.sf.Bquiver()

    elm = elm_coils()
    elm.pf.plot(subcoil=True)


#plt.plot(tor.t, psi[:, 0] - psi[:, 1])
'''



#tor.set_current(20)

scn.update_psi(n=1e3, plot=True)


elm.plot()


#point = (8.23013, -0.54975)
psi = scn.get_flux()
'''
