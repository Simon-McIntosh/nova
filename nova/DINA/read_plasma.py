import pandas as pd
from amigo.pyplot import plt
import numpy as np
from amigo.addtext import linelabel
from read_dina import dina, lowpass
from scipy.interpolate import interp1d
from read_dina import timeconstant
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from collections import OrderedDict


class read_plasma:

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)
        self.Ivs3_properties()

    def read_file(self, folder, dropnan=True):
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
        data['Ip'] = -1e3*self.data['Ip']  # -kA to A
        data['Ivs3_o'] = -1e3*self.data.loc[:, 'I_dw'] / 4  # -kAturn to A
        data['x'] = 1e-2*self.data['Rcur']  # cm - m
        data['z'] = 1e-2*self.data['Zcur']  # cm - m
        dt_min = np.nanmin(np.diff(data['t']))
        t_max = np.nanmax(data['t'])
        self.nt = int(t_max / dt_min)
        self.t = np.linspace(0, t_max, self.nt)  # equal spaced time stencil
        self.dt = t_max/(self.nt-1)
        for var in data:
            if var != 't':  # interpolate
                setattr(self, var, interp1d(data['t'], data[var])(self.t))
        self.zdir = np.sign(self.z[np.argmax(abs(self.z))])

    def plot_currents(self, ax=None):
        if ax is None:
            ax = plt.subplots(2, 1, sharex=True)[1]
        ax[0].plot(1e3*self.t, 1e-6*self.Ip, 'C0')
        ax[0].set_ylabel('$I_p$ MA')
        ax[1].plot(1e3*self.t, 1e-3*self.Ivs3_o, 'C1')
        ax[1].set_ylabel('$Ivs3_o$ kA')
        ax[1].set_xlabel('$t$ ms')
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)

    def Ivs3_properties(self, Rvs3=17.66e-3, Lvs3=1.52e-3, Io=60e3):
        self.Rvs3 = Rvs3
        self.Lvs3 = Lvs3
        self.tau_vs3 = self.Lvs3/self.Rvs3  # vs3 timeconstant
        self.tau_d = {}
        self.tau_d['DINA'] = {'alpha': [0.64754104,  0.13775032,  0.21254934],
                              'tau': [0.04390661,  0.01073328,  0.19240964]}
        self.tau_d['LTC'] = {'alpha': [0.0629583,  0.3059171,  0.63042031],
                             'tau': [0.00068959,  0.01501697,  0.09481686]}
        self.tau_d['ENP'] = {'alpha': [0.4003159,  0.36687173,  0.23344466],
                             'tau': [0.0283991,  0.12646493,  0.04291896]}
        self.Io = Io  # control spike current magnitude
        self.trip_dtype = [('name', '60U'),  # construct data strucutre
                           ('t_cq', float),  # current quench
                           ('i_cq', int),  # quench index
                           ('I_cq', float),  # plasma current
                           ('discharge_time', float),
                           ('discharge_type', '60U'),
                           ('zdir', int),  # VDE direction
                           ('t_dz', float), ('i_dz', int),
                           ('I_dz', float), ('z_dz', float),
                           ('i_trip', int), ('t_trip', float)]
        self.Ivs3_dtype = [('referance', '2float'),
                           ('control', '2float'),
                           ('error', '2float')]

    def get_vs3_trip(self, dIdt_trip=5e7, dz_trip=0.16, plot=False):
        Ip_lp = lowpass(self.Ip, self.dt, dt_window=0.001)  # plasma current
        dIpdt = np.gradient(Ip_lp, self.t)
        i_cq = next((i for i, dIdt in enumerate(dIpdt) if dIdt > dIdt_trip))
        t_cq = self.t[i_cq]  # current quench time
        tc = timeconstant(self.t[i_cq:], Ip_lp[i_cq:], trim_fraction=0.5)
        tdis, ttype, tfit, Ifit = tc.fit(plot=False, Io=-15e6)  # cq
        dZ = self.z - self.z[0]  # displacment trip
        i_dz = next((i for i, dz in enumerate(dZ) if abs(dz) > dz_trip))
        t_dz = self.t[i_dz]
        i_trip = i_dz if i_dz < i_cq else i_cq  # select first trigger
        t_trip = self.t[i_trip]  # trip time

        trip = np.zeros(1, dtype=self.trip_dtype)
        trip['name'] = self.name
        trip['t_cq'] = t_cq
        trip['i_cq'] = i_cq
        trip['I_cq'] = self.Ip[i_cq]
        trip['discharge_time'] = tdis
        trip['discharge_type'] = ttype
        trip['zdir'] = self.zdir
        trip['i_dz'] = i_dz
        trip['t_dz'] = t_dz
        trip['I_dz'] = self.Ip[i_dz]
        trip['z_dz'] = self.z[i_dz]
        trip['i_trip'] = i_trip
        trip['t_trip'] = t_trip

        if plot:
            txt = '{} discharge, t={:1.0f}ms'.format(ttype, 1e3*tdis)
            ax = plt.subplots(2, 1, sharex=True)[1]
            ax[0].plot(1e3*self.t, 1e-6*self.Ip)
            ax[0].plot(1e3*tfit, 1e-6*Ifit, label=txt)
            ax[0].set_ylabel('$Ip$ MA')
            ax[0].legend()

            ax[1].plot(1e3*self.t, 1e-6*dIpdt)
            txt_cq = '$t_{cq}$'+'={:1.1f}ms'.format(1e3*t_cq)
            ax[1].plot(1e3*t_cq, 1e-6*dIpdt[i_cq], '*', label=txt_cq,
                       color='gray')
            txt_dz = r'$t_{\Delta z}$'+'={:1.1f}ms'.format(1e3*t_dz)
            ax[1].plot(1e3*t_dz, 1e-6*dIpdt[i_dz], '^', label=txt_dz,
                       color='gray')

            ax[1].set_xlabel('$t$ ms')
            ax[1].set_ylabel('$dI_p/dt$ MAs$^{-1}$')
            ax[1].legend()
            plt.despine()
            plt.setp(ax[0].get_xticklabels(), visible=False)
            ax[0].text(0.5, 1.05, self.name,
                       transform=ax[0].transAxes,
                       bbox=dict(alpha=0.25), va='bottom', ha='center')
        return trip[0]

    def Ivs3_single(self, folder, plot=False):
        self.read_file(folder)  # load plasma file
        trip = self.get_vs3_trip()  # vs3 trigger
        Ivs3_data = {}  # VS3 system
        Ivs3_data['t'] = self.t
        Ivs3_data['Ireferance'] = self.Ivs3_o

        Ivs3_data['tpre'] = self.t[:trip['i_cq']]
        Ivs3_data['Ipre'] = self.Ivs3_o[:trip['i_cq']]
        Ivs3_data['Icontrol'] = self.Ioffset(trip['t_trip'], trip['zdir'])
        Ivs3_data['Ierror'] = self.Ioffset(trip['t_trip'], -trip['zdir'])
        Ivs3 = np.zeros(1, dtype=self.Ivs3_dtype)
        Ivs3_fun = OrderedDict()  # Ivs3 interpolator
        for mode in Ivs3.dtype.names:
            # max values
            Imode = 'I{}'.format(mode)
            index = np.argmax(abs(Ivs3_data[Imode]))
            Ivs3[mode] = np.array([Ivs3_data['t'][index],
                                  Ivs3_data[Imode][index]])
            Ivs3_fun[mode] = interp1d(Ivs3_data['t'], Ivs3_data[Imode],
                                      fill_value=(Ivs3_data[Imode][0],
                                                  Ivs3_data[Imode][-1]),
                                      bounds_error=False)
        if plot:
            for mode, color in zip(Ivs3_fun, ['gray', 'C0', 'C3']):
                plt.plot(1e3*self.t, 1e-3*Ivs3_fun[mode](self.t),
                         label=mode, color=color)
            plt.despine()
            plt.legend()
            plt.xlabel('$t$ ms')
            plt.ylabel('$I_{vs3}$ kA')
            plt.title(self.name)
        return trip, Ivs3_data, Ivs3[0], Ivs3_fun

    def Ivs3_ensemble(self, plot=False):
        self.trip = np.zeros(self.dina.nfolder, dtype=self.trip_dtype)
        self.Ivs3_data = [{} for _ in range(self.dina.nfolder)]  # Ivs data
        self.Ivs3 = np.zeros(self.dina.nfolder, dtype=self.Ivs3_dtype)

        for i in range(self.dina.nfolder):
            self.trip[i], self.Ivs3_data[i], self.Ivs3[i] \
                = self.Ivs3_single(i)[:3]
        if plot:
            self.plot_Ivs3_ensemble()

    def Ioffset(self, t_cq, zdir, discharge='DINA'):
        coef = self.tau_d[discharge]
        Id = np.zeros(self.nt)
        index = self.t > t_cq
        for alpha, tau in zip(coef['alpha'], coef['tau']):
            Id[index] += alpha*np.exp(-(self.t[index]-t_cq)/tau)
        Ic = self.Ivs3_o - zdir*self.Io*Id
        return Ic

    def get_color(self, index):
        t_cq = self.trip[index]['t_cq']
        label = ''
        if t_cq < 400e-3:  # MD
            iax = 0
            ic = 0
            label += 'MD '
        else:  # VDE
            iax = 1
            ic = 2
            label += 'VDE '
        if 'DW' in self.trip[index]['name']:
            color = 'C{}'.format(ic)
            label += 'down'
        else:
            color = 'C{}'.format(ic+1)
            label += 'up'
        return color, iax, label

    def plot_Ivs3_ensemble(self):
        ax = plt.subplots(2, 1)[1]
        text, loc = [], 'min'
        for i in range(2):
            text.append(linelabel(postfix='kA', value='1.1f',
                                  loc=loc, ax=ax[i]))
        for i in range(self.dina.nfolder):
            color, iax = self.get_color(i)[:2]
            ax[iax].plot(
              1e3*self.Ivs3_data[i]['t'][self.trip[i]['i_cq']:],
              1e-3*self.Ivs3_data[i]['Ireferance'][self.trip[i]['i_cq']:],
              color=color)
            text[iax].add('')
            ax[iax].plot(1e3*self.Ivs3_data[i]['tpre'],
                         1e-3*self.Ivs3_data[i]['Ipre'], '-',
                         color='gray', lw=0.5)
            ax[iax].plot(
                1e3*self.Ivs3_data[i]['t'][self.trip[i]['i_cq']],
                1e-3*self.Ivs3_data[i]['Ireferance'][self.trip[i]['i_cq']],
                '*', ms=10, color=color)
            ax[iax].plot(
                1e3*self.Ivs3_data[i]['t'][self.trip[i]['i_dz']],
                1e-3*self.Ivs3_data[i]['Ireferance'][self.trip[i]['i_dz']],
                '^', ms=7, color=color)
        plt.despine()
        for i, vde in enumerate(['MD', 'VDE']):
            self.add_vde_legend(ax[i], i, vde)
            ax[i].set_xlabel('$t$ ms')
            ax[i].set_ylabel('$I_{vs3}$ kA')
            text[i].plot()
        plt.tight_layout()

    def add_vde_legend(self, ax, i, vde):
        h_down = mlines.Line2D([], [], color='C{}'.format(2*i),
                               label='{} down'.format(vde))
        h_up = mlines.Line2D([], [], color='C{}'.format(2*i+1),
                             label='{} up'.format(vde))
        line_legend = ax.legend(handles=[h_down, h_up], loc=1)
        if i == 1:
            h_cq = mlines.Line2D([], [], color='gray', marker='*',
                                 linestyle='None',  label='cq-trip')
            h_dz = mlines.Line2D([], [], color='gray', marker='^',
                                 linestyle='None', label=r'$\Delta z$-trip')
            ax.legend(handles=[h_cq, h_dz], loc=4)
            ax.add_artist(line_legend)

    def plot_displacment(self):
        ax = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)[1]
        text = []
        for i in range(2):
            text.append(linelabel(postfix='', value='', ax=ax[i], Ndiv=20))
        for i in range(self.dina.nfolder):
            self.read_file(i)  # load file
            i_cq = self.trip[i]['i_cq']
            I_cq = self.trip[i]['I_cq']
            i_dz = self.trip[i]['i_dz']
            color, iax = self.get_color(i)[:2]
            ax[iax].plot(self.x, self.z, '-', color=color)
            ax[iax].plot(self.x[i_dz], self.z[i_dz], '^', color=color)
            ax[iax].plot(self.x[i_cq], self.z[i_cq], '*', color=color)
            text[iax].add('$I_{cq}$ '+'{:1.1f}MA'.format(1e-6*I_cq))
        plt.axis('equal')
        plt.despine()
        for i, vde in enumerate(['MD', 'VDE']):
            self.add_vde_legend(ax[i], i, vde)
            text[i].plot()
            ax[i].set_xlabel('$x$ m')
        ax[0].set_ylabel('$y$ m')
        plt.setp(ax[1].get_yticklabels(), visible=False)

    def plot_Ivs3_max(self, mode):
        plt.figure()
        X = range(self.dina.nfolder)
        for i, x in enumerate(X):
            color, iax = self.get_color(i)[:2]
            plt.bar(x, 1e-3*self.Ivs3[mode][i][1], color=color, width=0.8)
        for Idir in [-1, 1]:
            max_index = np.argmax(Idir*self.Ivs3[mode][:, 1])
            Imax = self.Ivs3[mode][max_index, 1]
            va = 'bottom' if Imax < 0 else 'top'
            plt.text(max_index, 1e-3*Imax, ' {:1.0f}kA'.format(1e-3*Imax),
                     va=va, ha='center', color='k',
                     weight='bold', rotation=90)

        h = []
        for i, label in enumerate(['MD down', 'MD up', 'VDE down', 'VDE up']):
            h.append(mpatches.Patch(color='C{}'.format(i), label=label))

        plt.legend(handles=h)
        plt.xticks(X, self.dina.folders, rotation=70)
        plt.ylabel('$I_{vs3}$ kA')
        plt.ylim([-110, 110])
        plt.despine()
        plt.title(mode)

    def dIdt_fun(self, I, t):
        # vs_fun = args[0]
        # g = -I*R/L
        g = (self.vs_fun(t) - I*self.Rvs3)/self.Lvs3
        return g


if __name__ == '__main__':

    pl = read_plasma('disruptions')

    pl.Ivs3_single(11, plot=True)
    pl.Ivs3_ensemble(plot=True)  # load Ivs3 current waveforms
    pl.read_file(3)
    pl.get_vs3_trip(plot=True)

    pl.plot_displacment()

    for mode in pl.Ivs3.dtype.names:
        pl.plot_Ivs3_max(mode)
