import pandas as pd
from amigo.pyplot import plt
import numpy as np
from amigo.addtext import linelabel
from read_dina import dina
from scipy.interpolate import interp1d


class read_plasma:

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)

    def read_file(self, folder, dropnan=False):
        filename = self.dina.locate_file('plasma', folder=folder)
        self.name = filename.split('\\')[-2]
        self.data = pd.read_csv(filename, delim_whitespace=True, skiprows=40,
                                na_values='NAN')
        if dropnan:
            self.data = self.data.dropna()  # remove NaN values
        self.columns = {}
        for c in list(self.data):
            self.columns[c] = c.split('[')[0]
        self.data = self.data.rename(index=str, columns=self.columns)
        self.load_data()

    def load_data(self):
        data = {}  # convert units
        data['t'] = 1e-3*self.data['t']  # ms - s
        data['Ip'] = 1e3*self.data['Ip']  # kA - A
        data['Ivs_o'] = 1e3*self.data.loc[:, 'I_dw'] / 4  # kAturn - A

        dt_min = np.nanmin(np.diff(data['t']))
        t_max = np.nanmax(data['t'])
        self.nt = int(t_max / dt_min)
        self.t = np.linspace(0, t_max, self.nt)  # equal spaced time stencil
        for var in data:
            if var != 't':  # interpolate
                setattr(self, var, interp1d(data['t'], data[var])(self.t))

    def get_quench(self):  # locate current quench
        dIpdt = np.gradient(self.Ip, self.t)
        plt.plot(self.t, dIpdt)

    def plot_currents(self):
        ax = plt.subplots(2, 1, sharex=True)[1]
        ax[0].plot(1e3*self.t, 1e-6*self.Ip, 'C0')
        ax[0].set_ylabel('$I_p$ MA')
        ax[1].plot(1e3*self.t, 1e-3*self.Ivs_o, 'C1')
        ax[1].set_ylabel('$Ivs$ kA')

        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)

    def trip_vs3(self, eps=0.01):
        trip_current = 0.01*np.max(abs(self.Ivs_o))
        trip_index = next(i for i, Iind in enumerate(self.Ivs_o)
                          if abs(Iind) > trip_current)
        trip_t = self.t[trip_index]  # vs3 circuit open
        return trip_index, trip_t

    def get_vs3(self, Io=60, tau=None, plot=False):
        if tau is None:
            tau = 1.53/(17.66)*1e3  # vs3 discharge time constant
        self.tau = tau
        self.t = np.copy(self.data.loc[:, 't'])
        self.Iind = self.data.loc[:, 'I_dw'] / 4  # per turn
        self.vs_sign = np.sign(self.Iind[np.argmax(abs(self.Iind))])
        Io *= self.vs_sign  # induced current in same direction as pulse
        self.Io = Io*np.ones(len(self.t))  # inital current

        trip_index, trip_t = self.trip_vs3()
        self.Io[trip_index:] = Io*np.exp(-(self.t[trip_index:]-trip_t)/tau)
        self.Ivs = self.Iind+self.Io

        Itau = self.Ivs
        Iinf = self.Iind+Io
        self.Imax = {'tau': Itau[np.argmax(abs(Itau))],
                     'inf': Iinf[np.argmax(abs(Iinf))]}
        if plot:
            loc = 'max' if self.vs_sign > 0 else 'min'
            text = linelabel(loc=loc, postfix='kA')
            plt.plot(self.t, self.Iind+Io, '-', label=r'$\tau=\infty$ms')
            text.add('')
            plt.plot(self.t, self.Ivs, '-',
                     label=r'$\tau=${:1.0f}ms'.format(self.tau))
            text.add('')
            plt.plot(self.t, self.Io, '--', label='$I_{decay}$')
            plt.plot(self.t, self.Iind, '--', label='$I_{VDE}$')
            ax = plt.gca()
            ylim = ax.get_ylim()
            plt.plot(trip_t*np.ones(2), ylim, ':', color='gray')
            plt.ylabel(r'$I_{VS3}$ kA')
            plt.xlabel(r'$t$ ms')
            plt.text(0.5, 1, self.name, transform=plt.gca().transAxes,
                     weight='bold', ha='center')
            text.plot(Ralign=False)
            plt.legend(loc='best')
            plt.despine()


if __name__ == '__main__':

    pl = read_plasma('disruptions')


    pl.read_file(3)

    pl.get_quench()
    #pl.plot_currents()

    '''
    nF = len(folders)
    Itau, Iinf, vs_sign = np.ones(nF), np.ones(nF), np.ones(nF)
    for i, folder in enumerate(folders):
        plasma = read_plasma(directory, folder=folder)
        Itau[i] = plasma.Imax['tau']
        Iinf[i] = plasma.Imax['inf']
        vs_sign = plasma.vs_sign

    self.plot_plasma()
    '''

    '''
    plt.figure()
    x = range(len(folders))
    plt.bar(x, Iinf, width=0.8, label=r'$\tau=\infty$ms')
    plt.bar(x, Itau, width=0.6,
            label=r'$\tau=${:1.0f}ms'.format(plasma.tau))
    max_index = np.argmax(abs(Itau))
    va = 'top' if vs_sign > 0 else 'bottom'
    plt.text(max_index, Itau[max_index],
             '{:1.0f}kA'.format(Itau[max_index]), va=va, ha='center',
             color='k', weight='bold', rotation=90)
    plt.xticks(x, folders, rotation=70)
    plt.ylabel('$I_{vs}$ kA')
    plt.legend()
    plt.despine()

    plt.figure()
    max_index = 8
    plasma = read_plasma(directory, folder=folders[max_index])
    plasma.get_vs3(plot=True)
    '''